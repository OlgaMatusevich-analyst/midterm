// app.js
// ES module wiring UI ↔ DataLoader ↔ GRUClassifier. Provides progress, evaluation, confusion matrix, and safety.

export class App {
  constructor({ tf, dl, model, ui }) {
    this.tf = tf;
    this.dl = dl;
    this.model = model;
    this.ui = ui;

    // State
    this.dataset = null;

    // Wire UI
    ui.csvFile.addEventListener('change', () => this.#onCSVSelected());
    ui.prepBtn.addEventListener('click', () => this.#prepare());
    ui.buildBtn.addEventListener('click', () => this.#build());
    ui.trainBtn.addEventListener('click', () => this.#train());
    ui.evalBtn.addEventListener('click', () => this.#evaluate());
    ui.saveBtn.addEventListener('click', () => this.model.save());
    ui.loadBtn.addEventListener('click', () => this.model.load().then(() => this.#toggleTrainButtons(true)));
    ui.resetBtn.addEventListener('click', () => this.#reset());
  }

  // ---- UI handlers ----

  async #onCSVSelected() {
    try {
      this.#setStatus('dataStatus', 'Reading…', '#fef3c7', '#92400e');
      await this.dl.fromFile(this.ui.csvFile.files[0]);
      this.ui.prepBtn.disabled = false;
      this.#setStatus('dataStatus', 'Loaded', '#dcfce7', '#166534');
    } catch (err) {
      this.#setStatus('dataStatus', 'Error', '#fee2e2', '#991b1b');
      alert(err.message || String(err));
    }
  }

  async #prepare() {
    try {
      this.#progress(0);
      const seqLen = Math.max(1, Number(this.ui.seqLen.value) | 0);
      const testSplit = Number(this.ui.testSplit.value) / 100 || 0.2;
      if (seqLen > 16) throw new Error('Sequence length too large.');

      this.dataset?.xTrain?.dispose?.();
      this.dataset?.yTrain?.dispose?.();
      this.dataset?.xTest?.dispose?.();
      this.dataset?.yTest?.dispose?.();

      this.dataset = this.dl.prepareTensors({ seqLen, testSplit });
      this.ui.buildBtn.disabled = false;
      this.#log(`Dataset ready. Features per step: ${this.dataset.xTrain.shape[2]}`);
    } catch (err) {
      alert(err.message || String(err));
    }
  }

  #build() {
    try {
      const timesteps = this.dataset?.xTrain?.shape?.[1];
      const features = this.dataset?.xTrain?.shape?.[2];
      if (!timesteps || !features) throw new Error('Dataset not prepared.');

      const units = Math.max(8, Number(this.ui.units.value) | 0);
      const layers = Math.max(1, Number(this.ui.layers.value) | 0);
      const lr = Number(this.ui.lr.value) || 1e-3;

      this.model.build({ timesteps, features, units, layers, lr });
      this.#toggleTrainButtons(true);
    } catch (err) {
      alert(err.message || String(err));
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
        onEpoch: (epoch, logs) => {
          this.#progress((epoch + 1) / epochs);
        }
      });
      this.ui.evalBtn.disabled = false;
      this.#progress(1);
    } catch (err) {
      alert(err.message || String(err));
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
    } catch (err) {
      alert(err.message || String(err));
    }
  }

  #renderMetrics({ acc, prec, rec, f1, auc, cm }) {
    const n = (v) => Number.isFinite(v) ? v.toFixed(4) : '–';
    this.ui.elAcc.textContent = n(acc);
    this.ui.elPrec.textContent = n(prec);
    this.ui.elRec.textContent = n(rec);
    this.ui.elF1.textContent = n(f1);
    this.ui.elAUC.textContent = n(auc);

    this.ui.cmTN.textContent = cm.tn ?? '–';
    this.ui.cmFP.textContent = cm.fp ?? '–';
    this.ui.cmFN.textContent = cm.fn ?? '–';
    this.ui.cmTP.textContent = cm.tp ?? '–';
  }

  // ---- helpers ----

  #toggleTrainButtons(enable) {
    this.ui.buildBtn.disabled = !enable;
    this.ui.trainBtn.disabled = !enable;
    this.ui.evalBtn.disabled = true;
    this.ui.saveBtn.disabled = !enable;
  }

  #setStatus(id, text, bg = '#eef2ff', color = '#3730a3') {
    const el = document.getElementById(id);
    el.textContent = text;
    el.style.background = bg;
    el.style.color = color;
  }

  #progress(v) {
    this.ui.prog.value = Math.max(0, Math.min(1, v));
  }

  #reset() {
    try {
      this.model.dispose();
      if (this.dataset) {
        this.dataset.xTrain.dispose();
        this.dataset.yTrain.dispose();
        this.dataset.xTest.dispose();
        this.dataset.yTest.dispose();
      }
      this.dataset = null;
      this.ui.prepBtn.disabled = true;
      this.ui.buildBtn.disabled = true;
      this.ui.trainBtn.disabled = true;
      this.ui.evalBtn.disabled = true;
      this.ui.saveBtn.disabled = true;
      this.#progress(0);
      this.#renderMetrics({ acc: NaN, prec: NaN, rec: NaN, f1: NaN, auc: NaN, cm: {tp:'–',tn:'–',fp:'–',fn:'–'} });
      this.#log('Reset complete.');
    } catch (e) { /* ignore */ }
  }

  #log(msg) {
    // index.html injects a global logger; this fallback is silent
    if (typeof window !== 'undefined' && window?.console) console.log(msg);
  }
}
