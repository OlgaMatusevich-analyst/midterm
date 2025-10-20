// data-loader.js
// ES module to parse local CSV, build a (pseudo) time-series dataset, and export tf.Tensors.
// Handles categorical encoding, numeric scaling, chronological split, and memory safety.

export class DataLoader {
  constructor(opts = {}) {
    this.log = opts.log || (() => {});
    this.headers = [];
    this.raw = [];
    this.labelKey = 'Attrition';
    this.attritionMap = { Yes: 1, No: 0 };
    this.numCols = new Set([
      'Age','DailyRate','DistanceFromHome','Education','EmployeeCount','EmployeeNumber',
      'EnvironmentSatisfaction','HourlyRate','JobInvolvement','JobLevel','JobSatisfaction',
      'MonthlyIncome','MonthlyRate','NumCompaniesWorked','PercentSalaryHike','PerformanceRating',
      'RelationshipSatisfaction','StandardHours','StockOptionLevel','TotalWorkingYears',
      'TrainingTimesLastYear','WorkLifeBalance','YearsAtCompany','YearsInCurrentRole',
      'YearsSinceLastPromotion','YearsWithCurrManager'
    ]);
    this.catCols = [];      // resolved after reading headers
    this.encoders = {};     // { colName: {categoryValue: index, ...}, ...}
    this.scalers = {};      // { colName: {mean, std} }
    this.featureOrder = []; // final expanded feature names (after one-hot)
  }

  async fromFile(file) {
    if (!file) throw new Error('No file selected.');
    const text = await this.#readFileText(file);
    const { headers, rows } = this.#parseCSV(text);
    this.headers = headers;

    if (!headers.includes(this.labelKey)) {
      throw new Error(`CSV is missing required label column: ${this.labelKey}`);
    }

    // Determine categorical columns (non-numeric or known categorical)
    const knownCategoricals = new Set([
      'BusinessTravel','Department','EducationField','Gender','JobRole','MaritalStatus','Over18','OverTime'
    ]);
    this.catCols = headers.filter(h => knownCategoricals.has(h));

    // Keep only required columns (ignore extras gracefully)
    const allowed = new Set([...this.numCols, ...this.catCols, this.labelKey]);
    const filteredRows = rows.map(r => {
      const obj = {};
      for (const k of headers) {
        if (allowed.has(k)) obj[k] = r[k];
      }
      return obj;
    });

    this.raw = filteredRows;
    this.log(`Loaded ${this.raw.length} rows, ${this.catCols.length} categorical, ${this.numCols.size} numeric.`);
    return this;
  }

  // Prepare tensors. seqLen>=1 creates sliding windows across rows sorted by a proxy "time".
  prepareTensors({ seqLen = 1, testSplit = 0.2 }) {
    if (!this.raw?.length) throw new Error('Dataset not loaded.');

    // Sort "chronologically": by YearsAtCompany asc, then EmployeeNumber asc (deterministic).
    const sorted = [...this.raw].sort((a, b) => {
      const ya = Number(a.YearsAtCompany ?? 0), yb = Number(b.YearsAtCompany ?? 0);
      if (ya !== yb) return ya - yb;
      return Number(a.EmployeeNumber ?? 0) - Number(b.EmployeeNumber ?? 0);
    });

    // Build encoders for categoricals
    this.#fitCategoricals(sorted);

    // Fit numeric scalers on train-only portion later. First transform all to feature vectors.
    const feats = [];
    const labels = [];
    for (const r of sorted) {
      const { v, y } = this.#rowToFeatures(r);
      feats.push(v);
      labels.push(y);
    }

    // Create sliding windows (time steps)
    const S = Math.max(1, Number(seqLen) | 0);
    const X_seq = [];
    const y_seq = [];
    for (let i = 0; i + S - 1 < feats.length; i++) {
      const window = feats.slice(i, i + S);
      // label is the last row's label (next outcome)
      const target = labels[i + S - 1];
      X_seq.push(window);
      y_seq.push([target]);
    }

    // Chronological split
    const n = X_seq.length;
    const nTest = Math.max(1, Math.floor(n * Math.min(Math.max(Number(testSplit) / 100 || testSplit, 0.05), 0.9)));
    const nTrain = n - nTest;

    const Xtr = X_seq.slice(0, nTrain);
    const ytr = y_seq.slice(0, nTrain);
    const Xte = X_seq.slice(nTrain);
    const yte = y_seq.slice(nTrain);

    // Fit scalers only on training portion (per feature dimension)
    this.#fitScalers(Xtr);

    // Apply scaling
    const XtrScaled = this.#applyScaling(Xtr);
    const XteScaled = this.#applyScaling(Xte);

    const xTrain = tf.tensor3d(XtrScaled);
    const yTrain = tf.tensor2d(ytr);
    const xTest = tf.tensor3d(XteScaled);
    const yTest = tf.tensor2d(yte);

    this.log(`Prepared tensors: xTrain ${xTrain.shape}, yTrain ${yTrain.shape}, xTest ${xTest.shape}, yTest ${yTest.shape}`);
    return { xTrain, yTrain, xTest, yTest, attritionMap: this.attritionMap, featureOrder: this.featureOrder.slice() };
  }

  // ---- internal helpers ----

  #readFileText(file) {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onerror = () => reject(new Error('Failed to read file.'));
      reader.onload = () => resolve(String(reader.result));
      reader.readAsText(file);
    });
  }

  #parseCSV(text) {
    // Simple CSV parser suitable for IBM Attrition dataset (no quoted commas in practice).
    const lines = text.replace(/\r\n/g, '\n').replace(/\r/g, '\n').split('\n').filter(Boolean);
    if (lines.length < 2) throw new Error('CSV has no data.');
    const headers = lines[0].split(',').map(h => h.trim());
    const rows = [];

    for (let i = 1; i < lines.length; i++) {
      const parts = lines[i].split(',');
      if (parts.length !== headers.length) continue; // skip malformed lines
      const obj = {};
      for (let j = 0; j < headers.length; j++) obj[headers[j]] = parts[j].trim();
      rows.push(obj);
    }
    return { headers, rows };
  }

  #fitCategoricals(rows) {
    for (const c of this.catCols) {
      const set = new Set();
      for (const r of rows) set.add(r[c] ?? '');
      const cats = Array.from(set.values()).sort();
      const enc = {};
      cats.forEach((v, i) => enc[v] = i);
      this.encoders[c] = enc;
    }

    // Build feature order (numeric + expanded categoricals)
    const order = [];
    for (const col of this.numCols) order.push(col);
    for (const c of this.catCols) {
      const enc = this.encoders[c];
      for (const v of Object.keys(enc)) order.push(`${c}__${v}`);
    }
    this.featureOrder = order;
  }

  #rowToFeatures(r) {
    // numeric
    const feat = [];
    for (const col of this.numCols) {
      const val = Number(r[col] ?? 0);
      feat.push(Number.isFinite(val) ? val : 0);
    }
    // one-hot categoricals
    for (const c of this.catCols) {
      const enc = this.encoders[c];
      const vec = new Array(Object.keys(enc).length).fill(0);
      const idx = enc[r[c] ?? ''];
      if (Number.isInteger(idx)) vec[idx] = 1;
      feat.push(...vec);
    }
    const y = this.attritionMap[String(r[this.labelKey] || 'No')] ?? 0;
    return { v: feat, y };
  }

  #fitScalers(X) {
    // X: array of [timesteps][featdim]
    const timesteps = X[0]?.length || 1;
    const dim = X[0]?.[0]?.length || 0;
    const all = [];
    for (const seq of X) for (const t of seq) all.push(t);
    const stats = [];
    for (let j = 0; j < dim; j++) {
      let s = 0, s2 = 0;
      for (let i = 0; i < all.length; i++) {
        const v = all[i][j];
        s += v; s2 += v*v;
      }
      const n = all.length || 1;
      const mean = s / n;
      const std = Math.sqrt(Math.max(1e-9, s2/n - mean*mean));
      stats.push({mean, std});
    }
    this.scalers = { dimStats: stats };
  }

  #applyScaling(X) {
    const dimStats = this.scalers.dimStats;
    return X.map(seq => seq.map(t => t.map((v, j) => (v - dimStats[j].mean) / (dimStats[j].std))));
  }
}
