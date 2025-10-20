// gru.js
// ES module defining a GRU-based binary classifier with utilities for build, train, predict, evaluate, save, and load.

export class GRUClassifier {
  constructor(opts = {}) {
    this.log = opts.log || (() => {});
    this.model = null;
    this.inputShape = null; // [timesteps, features]
  }

  build({ timesteps, features, units = 64, layers = 2, lr = 1e-3 }) {
    if (!Number.isInteger(timesteps) || !Number.isInteger(features) || timesteps < 1 || features < 1) {
      throw new Error(`Invalid input shape timesteps=${timesteps}, features=${features}`);
    }
    this.dispose();
    this.inputShape = [timesteps, features];

    const m = tf.sequential();
    const gruUnits = Math.max(8, units|0);

    // Stacked GRU (first returnSequences=true if more than one layer)
    if (layers > 1) {
      m.add(tf.layers.gru({
        units: gruUnits,
        returnSequences: true,
        inputShape: this.inputShape,
        dropout: 0.1,
        recurrentDropout: 0.0,
        kernelInitializer: 'heNormal'
      }));
      for (let i = 0; i < layers - 2; i++) {
        m.add(tf.layers.gru({ units: gruUnits, returnSequences: true, dropout: 0.1 }));
      }
      m.add(tf.layers.gru({ units: gruUnits, returnSequences: false, dropout: 0.1 }));
    } else {
      m.add(tf.layers.gru({
        units: gruUnits,
        returnSequences: false,
        inputShape: this.inputShape,
        dropout: 0.1,
        kernelInitializer: 'heNormal'
      }));
    }

    m.add(tf.layers.dense({ units: 64, activation: 'relu', kernelInitializer: 'heNormal' }));
    m.add(tf.layers.dropout({ rate: 0.2 }));
    m.add(tf.layers.dense({ units: 1, activation: 'sigmoid' })); // binary output

    const optimizer = tf.train.adam(typeof lr === 'number' ? lr : 1e-3);
    m.compile({
      optimizer,
      loss: 'binaryCrossentropy',
      metrics: ['binaryAccuracy']
    });

    this.model = m;
    this.log(`Built GRU model: input [${timesteps}, ${features}], units=${gruUnits}, layers=${layers}`);
    return this.model;
  }

  async fit({ xTrain, yTrain, epochs = 20, batchSize = 32, onEpoch, onBatch }) {
    if (!this.model) throw new Error('Build the model before training.');
    const bs = Math.max(1, batchSize|0);
    const ep = Math.max(1, epochs|0);

    const cb = {
      onEpochEnd: async (epoch, logs) => {
        this.log(`Epoch ${epoch+1}/${ep} — loss=${logs.loss?.toFixed(4)} acc=${(logs.binaryAccuracy??0).toFixed(4)}`);
        if (onEpoch) onEpoch(epoch, logs);
        await tf.nextFrame();
      },
      onBatchEnd: async (batch, logs) => {
        if (onBatch) onBatch(batch, logs);
        await tf.nextFrame();
      }
    };

    return await this.model.fit(xTrain, yTrain, {
      epochs: ep,
      batchSize: bs,
      shuffle: true,
      callbacks: cb,
      validationSplit: 0.0
    });
  }

  predict(x) {
    if (!this.model) throw new Error('Model not built.');
    return tf.tidy(() => this.model.predict(x));
  }

  async evaluate({ xTest, yTest, threshold = 0.5 }) {
    if (!this.model) throw new Error('Model not built.');
    const yProb = this.predict(xTest);        // Tensor2D [N,1]
    const yPred = tf.tidy(() => yProb.greaterEqual(threshold).toInt());

    // Confusion matrix
    const yTrue = yTest.toInt();
    const tp = tf.tidy(() => tf.logicalAnd(yPred.equal(1), yTrue.equal(1)).sum().arraySync());
    const tn = tf.tidy(() => tf.logicalAnd(yPred.equal(0), yTrue.equal(0)).sum().arraySync());
    const fp = tf.tidy(() => tf.logicalAnd(yPred.equal(1), yTrue.equal(0)).sum().arraySync());
    const fn = tf.tidy(() => tf.logicalAnd(yPred.equal(0), yTrue.equal(1)).sum().arraySync());

    // Metrics
    const acc = (tp + tn) / Math.max(1, tp + tn + fp + fn);
    const prec = tp / Math.max(1, tp + fp);
    const rec = tp / Math.max(1, tp + fn);
    const f1 = 2 * prec * rec / Math.max(1e-9, (prec + rec));

    // ROC AUC (simple trapezoidal estimate over thresholds)
    const auc = await this.#rocAuc(yTest, yProb);

    yProb.dispose(); yPred.dispose(); yTrue.dispose();

    return { acc, prec, rec, f1, auc, cm: { tp, tn, fp, fn } };
  }

  async save(name = 'tfjs-hr-attrition-gru') {
    if (!this.model) throw new Error('Model not built.');
    await this.model.save(`indexeddb://${name}`);
    this.log(`Saved weights to indexedDB://${name}`);
  }

  async load(name = 'tfjs-hr-attrition-gru') {
    this.dispose();
    this.model = await tf.loadLayersModel(`indexeddb://${name}`);
    this.inputShape = this.model.inputs?.[0]?.shape?.slice(1);
    this.log(`Loaded model from indexedDB://${name} (input: ${this.inputShape})`);
    return this.model;
  }

  dispose() {
    if (this.model) {
      this.model.dispose();
      this.model = null;
    }
  }

  async #rocAuc(yTrue, yProb) {
    const yT = (await yTrue.array()).map(r => r[0]);
    const yP = (await yProb.array()).map(r => r[0]);

    const pairs = yP.map((p, i) => ({ p, y: yT[i] }));
    pairs.sort((a, b) => b.p - a.p);

    let tp = 0, fp = 0;
    const P = yT.reduce((s, v) => s + (v === 1 ? 1 : 0), 0);
    const N = yT.length - P;
    const roc = [{ fpr: 0, tpr: 0 }];

    for (const { y } of pairs) {
      if (y === 1) tp++; else fp++;
      const tpr = tp / Math.max(1, P);
      const fpr = fp / Math.max(1, N);
      roc.push({ fpr, tpr });
    }
    roc.push({ fpr: 1, tpr: 1 });

    let auc = 0;
    for (let i = 1; i < roc.length; i++) {
      const x1 = roc[i-1].fpr, y1 = roc[i-1].tpr;
      const x2 = roc[i].fpr,   y2 = roc[i].tpr;
      auc += (x2 - x1) * (y1 + y2) / 2;
    }
    return 1 - Math.min(1, Math.max(0, auc)); // convert to proper orientation
  }
}
