// app.js
// Wire UI ↔ DataLoader ↔ GRUClassifier. Adds EDA text and charts, robust training flow, and clear logs.

export class App {
  constructor({ tf, Chart, dl, model, ui }) {
    this.tf = tf;
    this.Chart = Chart;
    this.dl = dl;
    this.model = model;
    this.ui = ui;

    this.dataset = null;
    this.charts = { balance:null, overtime:null, corr:null };

    ui.csvFile.addEventListener('change', () => this.#onCSV());
    ui.prepBtn.addEventListener('click', () => this.#prepare());
    ui.edaBtn.addEventListener('click', () => this.#runEDA());
    ui.buildBtn.addEventListener('click', () => this.#build());
    ui.trainBtn.addEventListener('click', () => this.#train());
    ui.evalBtn.addEventListener('click', () => this.#evaluate());
    ui.saveBtn.addEventListener('click', () => this.model.save());
    ui.loadBtn.addEventListener('click', () => this.model.load().then(()=>this.#toggleTrainButtons(true)));
    ui.resetBtn.addEventListener('click', () => this.#reset());
  }

  // ----- Data / EDA -----
  async #onCSV() {
    try {
      this.#status('dataStatus','Reading…','#fef3c7','#92400e');
      await this.dl.fromFile(this.ui.csvFile.files[0]);
      this.ui.prepBtn.disabled = false;
      this.ui.edaBtn.disabled = false;
      this.#status('dataStatus','Loaded','#dcfce7','#166534');
    } catch (e) {
      this.#status('dataStatus','Error','#fee2e2','#991b1b');
      alert(e.message || String(e));
    }
  }

  async #runEDA() {
    try {
      const { balance, topCorr, catRates } = this.dl.eda();
      // Text summary
      this.ui.edaText.innerHTML =
        `Class balance — Yes: <b>${balance.positive}</b>, No: <b>${balance.negative}</b> (rate ${(balance.rate*100).toFixed(1)}%).<br/>
         Top numeric correlations with Attrition: <b>${topCorr.map(([k,v])=>`${k} (${v.toFixed(2)})`).join(', ')}</b>.<br/>
         OverTime drives attrition (rates shown in chart).`;

      // Charts
      this.#renderChart('balance', this.ui.chartClassBalance, {
        labels: ['No','Yes'],
        data: [balance.negative, balance.positive],
        title: 'Class Balance'
      });

      if (catRates.OverTime) {
        this.#renderChart('overtime', this.ui.chartOvertime, {
          labels: catRates.OverTime.map(d=>d.k),
          data: catRates.OverTime.map(d=>+(d.rate*100).toFixed(2)),
          title: 'Attrition rate by OverTime (%)'
        });
      }

      const corr = topCorr;
      this.#renderChart('corr', this.ui.chartCorr, {
        labels: corr.map(d=>d[0]),
        data: corr.map(d=>+(d[1]).toFixed(3)),
        title: 'Top numeric correlations (Pearson)'
      });
    } catch (e) {
      alert(e.message || String(e));
    }
  }

  // ----- Tensors / Model -----
  async #prepare() {
    try {
      this.#progress(0);
      const testSplit = Number(this.ui.testSplit.value) / 100 || 0.2;

      this.dataset?.xTrain?.dispose?.();
      this.dataset?.yTrain?.dispose?.();
      this.dataset?.xTest?.dispose?.();
      this.dataset?.yTest?.dispose?.();

      // 2D tensors from DataLoader
      this.dataset = this.dl.prepareTensors({ testSplit });

      // Expand to 3D for GRU: [N, 1, F]
      const xTr3 = this.dataset.xTrain.expandDims(1);
      const xTe3 = this.dataset.xTest.expandDims(1);
      this.dataset.xTrain.dispose(); this.dataset.xTest.dispose();
      this.dataset.xTrain = xTr3; this.dataset.xTest = xTe3;

      this.ui.buildBtn.disabled = false;
      this.#log(`Dataset ready. Timesteps: ${this.dataset.xTrain.shape[1]}, Features: ${this.dataset.xTrain.shape[2]}`);
    } catch (e) {
      alert(e.message || String(e));
    }
  }

  #build() {
    try {
      const timesteps = this.dataset?.xTrain?.shape?.[1];   // = 1
      const features  = this.dataset?.xTrain?.shape?.[2];
      if (!timesteps || !features) throw new Error('Prepare dataset first.');
      const units = Math.max(8, Number(this.ui.units.value) | 0);
      const layers = Math.max(1, Number(this.ui.layers.value) | 0);
      const lr = Number(this.ui.lr.value) || 1e-3;
      this.model.build({ timesteps, features, units, layers, lr });
      this.#toggleTrainButtons(true);
    } catch (e) {
      alert(e.message || String(e));
    }
  }

  async #train() {
    try {
      this.#progress(0);
      const epochs = Math.max(1, Number(this.ui.epochs.value) | 0);
      const batchSize = Math.max(1, Number(this.ui.batchSize.value) | 0);

      await this.model.fit({
        xTrain: this.dataset.xTrain,
        yTrain: this.dataset.yTrain,
        epochs,
        batchSize,
        onEpoch: (epoch, logs) => this.#progress((epoch+1)/epochs)
      });
      this.ui.evalBtn.disabled = false;
      this.#progress(1);
    } catch (e) {
      alert(e.message || String(e));
    }
  }

  async #evaluate() {
    try {
      const res = await this.model.evaluate({
        xTest: this.dataset.xTest,
        yTest: this.dataset.yTest,
        threshold: 0.5
      });
      this.#renderMetrics(res);
    } catch (e) {
      alert(e.message || String(e));
    }
  }

  // ----- UI helpers -----
  #renderMetrics({ acc, prec, rec, f1, auc, cm }) {
    const fmt = v => Number.isFinite(v) ? v.toFixed(4) : '–';
    this.ui.elAcc.textContent = fmt(acc);
    this.ui.elPrec.textContent = fmt(prec);
    this.ui.elRec.textContent = fmt(rec);
    this.ui.elF1.textContent = fmt(f1);
    this.ui.elAUC.textContent = fmt(auc);

    this.ui.cmTN.textContent = cm.tn ?? '–';
    this.ui.cmFP.textContent = cm.fp ?? '–';
    this.ui.cmFN.textContent = cm.fn ?? '–';
    this.ui.cmTP.textContent = cm.tp ?? '–';
  }

  #renderChart(key, canvas, { labels, data, title }) {
    const ctx = canvas.getContext('2d');
    this.charts[key]?.destroy?.();
    this.charts[key] = new this.Chart(ctx, {
      type: 'bar',
      data: { labels, datasets: [{ label: title, data }] },
      options: { responsive: true, plugins: { legend: { display:false }, title: { display:true, text:title } } }
    });
  }

  #toggleTrainButtons(enable) {
    this.ui.buildBtn.disabled = !enable;
    this.ui.trainBtn.disabled = !enable;
    this.ui.evalBtn.disabled = true;
    this.ui.saveBtn.disabled = !enable;
  }

  #status(id, text, bg, color) {
    const node = document.getElementById(id);
    if (node){ node.textContent=text; node.style.background=bg; node.style.color=color; }
  }

  #progress(v){ this.ui.prog.value = Math.max(0, Math.min(1, v)); }
  #reset() {
    try {
      this.model.dispose();
      if (this.dataset) {
        this.dataset.xTrain.dispose(); this.dataset.yTrain.dispose();
        this.dataset.xTest.dispose();  this.dataset.yTest.dispose();
      }
      this.dataset = null;
      this.ui.prepBtn.disabled = true;
      this.ui.buildBtn.disabled = true;
      this.ui.trainBtn.disabled = true;
      this.ui.evalBtn.disabled = true;
      this.ui.saveBtn.disabled = true;
      this.#progress(0);
      this.#renderMetrics({ acc:NaN, prec:NaN, rec:NaN, f1:NaN, auc:NaN, cm:{tp:'–',tn:'–',fp:'–',fn:'–'} });
      this.#log('Reset complete.');
    } catch(_e) {}
  }

  #log(msg){ if (typeof window!=='undefined' && window?.console) console.log(msg); }
}
