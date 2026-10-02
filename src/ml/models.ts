import type { TrainedModel, ModelType, FeatureMeta } from './types';
import { sigmoid } from './metrics';
import { DecisionTree } from './decisionTree';

function mulberry32(seed: number) {
  return () => {
    seed |= 0;
    seed = (seed + 0x6d2b79f5) | 0;
    let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

// ============ Naive Baseline ============
export function trainNaiveBaseline(
  X: number[][],
  y: number[],
  featureNames: string[],
  featureMeta: FeatureMeta[]
): TrainedModel {
  const p = y.reduce((a, b) => a + b, 0) / y.length;

  return {
    type: 'naive',
    name: 'Naive Baseline',
    predictProba: () => p,
    predict: () => (p >= 0.5 ? 1 : 0),
    featureNames,
    featureMeta,
    hyperparameters: { majorityClassRate: p },
  };
}

// ============ Logistic Regression ============
export function trainLogisticRegression(
  X: number[][],
  y: number[],
  featureNames: string[],
  featureMeta: FeatureMeta[],
  options: { lr?: number; l2?: number; epochs?: number } = {}
): TrainedModel & { weights: number[]; bias: number } {
  const lr = options.lr ?? 0.05;
  const l2 = options.l2 ?? 0.01;
  const epochs = options.epochs ?? 200;
  const n = X.length;
  const d = X[0].length;

  let weights = new Array(d).fill(0);
  let bias = 0;

  for (let epoch = 0; epoch < epochs; epoch++) {
    const gradW = new Array(d).fill(0);
    let gradB = 0;

    for (let i = 0; i < n; i++) {
      const z = bias + X[i].reduce((s, x, j) => s + x * weights[j], 0);
      const p = sigmoid(z);
      const err = p - y[i];
      for (let j = 0; j < d; j++) {
        gradW[j] += err * X[i][j];
      }
      gradB += err;
    }

    for (let j = 0; j < d; j++) {
      weights[j] -= lr * (gradW[j] / n + l2 * weights[j]);
    }
    bias -= lr * (gradB / n);
  }

  const weightsCopy = [...weights];
  const biasCopy = bias;

  return {
    type: 'logistic',
    name: 'Logistic Regression',
    predictProba: (features: Record<string, number>) => {
      let z = biasCopy;
      for (let j = 0; j < featureNames.length; j++) {
        z += (features[featureNames[j]] ?? 0) * weightsCopy[j];
      }
      return sigmoid(z);
    },
    predict: (features: Record<string, number>) => {
      return sigmoid(biasCopy + featureNames.reduce((s, fn, j) => s + (features[fn] ?? 0) * weightsCopy[j], 0)) >= 0.5 ? 1 : 0;
    },
    featureNames,
    featureMeta,
    hyperparameters: { lr, l2, epochs },
    weights: weightsCopy,
    bias: biasCopy,
  };
}

// ============ Random Forest ============
export function trainRandomForest(
  X: number[][],
  y: number[],
  featureNames: string[],
  featureMeta: FeatureMeta[],
  options: { nTrees?: number; maxDepth?: number; maxFeatures?: number; minSamplesSplit?: number; seed?: number } = {}
): TrainedModel & { trees: DecisionTree[]; featureImportances: number[] } {
  const nTrees = options.nTrees ?? 30;
  const maxDepth = options.maxDepth ?? 8;
  const maxFeatures = options.maxFeatures ?? Math.max(2, Math.floor(Math.sqrt(featureNames.length)));
  const minSamplesSplit = options.minSamplesSplit ?? 10;
  const seed = options.seed ?? 42;

  const rng = mulberry32(seed);
  const n = X.length;
  const trees: DecisionTree[] = [];
  const featureImportances = new Array(featureNames.length).fill(0);

  for (let t = 0; t < nTrees; t++) {
    // Bootstrap sampling
    const sampleIdx: number[] = [];
    for (let i = 0; i < n; i++) {
      sampleIdx.push(Math.floor(rng() * n));
    }
    const Xb = sampleIdx.map((i) => X[i]);
    const yb = sampleIdx.map((i) => y[i]);

    const tree = new DecisionTree({ maxDepth, maxFeatures, minSamplesSplit }, rng);
    tree.fit(Xb, yb);
    trees.push(tree);
  }

  return {
    type: 'randomForest',
    name: 'Random Forest',
    predictProba: (features: Record<string, number>) => {
      const x = featureNames.map((fn) => features[fn] ?? 0);
      let sum = 0;
      for (const tree of trees) {
        sum += tree.predictProbaOne(x);
      }
      return sum / trees.length;
    },
    predict: (features: Record<string, number>) => {
      const x = featureNames.map((fn) => features[fn] ?? 0);
      let sum = 0;
      for (const tree of trees) {
        sum += tree.predictProbaOne(x);
      }
      return sum / trees.length >= 0.5 ? 1 : 0;
    },
    featureNames,
    featureMeta,
    hyperparameters: { nTrees, maxDepth, maxFeatures, minSamplesSplit },
    trees,
    featureImportances,
  };
}

// ============ Gradient Boosting ============
export function trainGradientBoosting(
  X: number[][],
  y: number[],
  featureNames: string[],
  featureMeta: FeatureMeta[],
  options: { nTrees?: number; maxDepth?: number; learningRate?: number; seed?: number } = {}
): TrainedModel & { trees: DecisionTree[]; initPrediction: number; learningRate: number } {
  const nTrees = options.nTrees ?? 50;
  const maxDepth = options.maxDepth ?? 4;
  const learningRate = options.learningRate ?? 0.1;
  const seed = options.seed ?? 42;

  const rng = mulberry32(seed);
  const n = X.length;

  // Initial prediction: log-odds of positive class
  const posRate = y.reduce((a, b) => a + b, 0) / n;
  const initLogit = Math.log(posRate / (1 - posRate + 1e-10) + 1e-10);
  let currentLogits = new Array(n).fill(initLogit);

  const trees: DecisionTree[] = [];

  for (let t = 0; t < nTrees; t++) {
    // Compute gradients (negative gradient of log-loss = y - sigmoid(logit))
    const residuals = y.map((yi, i) => yi - sigmoid(currentLogits[i]));

    const tree = new DecisionTree({ maxDepth, minSamplesSplit: 5, maxFeatures: featureNames.length }, rng);
    tree.fit(X, residuals);
    trees.push(tree);

    // Update logits
    for (let i = 0; i < n; i++) {
      const leafPred = tree.predictProbaOne(X[i]);
      currentLogits[i] += learningRate * leafPred;
    }
  }

  const initLogitCopy = initLogit;
  const treesCopy = [...trees];
  const lrCopy = learningRate;

  return {
    type: 'gradientBoosting',
    name: 'Gradient Boosting',
    predictProba: (features: Record<string, number>) => {
      const x = featureNames.map((fn) => features[fn] ?? 0);
      let logit = initLogitCopy;
      for (const tree of treesCopy) {
        logit += lrCopy * tree.predictProbaOne(x);
      }
      return sigmoid(logit);
    },
    predict: (features: Record<string, number>) => {
      const x = featureNames.map((fn) => features[fn] ?? 0);
      let logit = initLogitCopy;
      for (const tree of treesCopy) {
        logit += lrCopy * tree.predictProbaOne(x);
      }
      return sigmoid(logit) >= 0.5 ? 1 : 0;
    },
    featureNames,
    featureMeta,
    hyperparameters: { nTrees, maxDepth, learningRate },
    trees: treesCopy,
    initPrediction: initLogitCopy,
    learningRate: lrCopy,
  };
}

export function computeTreeFeatureImportance(
  model: TrainedModel & { trees?: DecisionTree[] }
): { feature: string; importance: number }[] {
  if (!model.trees || model.trees.length === 0) return [];

  const importances = new Array(model.featureNames.length).fill(0);

  for (const tree of model.trees) {
    collectImportance(tree.root, importances);
  }

  const total = importances.reduce((a, b) => a + b, 0) || 1;
  return model.featureNames.map((f, i) => ({
    feature: f,
    importance: importances[i] / total,
  })).sort((a, b) => b.importance - a.importance);
}

function collectImportance(node: import('./decisionTree').TreeNode, importances: number[]): void {
  if (node.isLeaf || node.feature === undefined) return;
  // Approximate importance by number of samples that pass through this node
  importances[node.feature] += node.samples ?? 0;
  if (node.left) collectImportance(node.left, importances);
  if (node.right) collectImportance(node.right, importances);
}

export function getLogisticFeatureImportance(
  model: TrainedModel & { weights?: number[] }
): { feature: string; importance: number }[] {
  if (!model.weights) return [];
  const absW = model.weights.map((w) => Math.abs(w));
  const total = absW.reduce((a, b) => a + b, 0) || 1;
  return model.featureNames.map((f, i) => ({
    feature: f,
    importance: absW[i] / total,
  })).sort((a, b) => b.importance - a.importance);
}

export const modelTypeNames: Record<ModelType, string> = {
  naive: 'Naive Baseline',
  logistic: 'Logistic Regression',
  decisionTree: 'Decision Tree',
  randomForest: 'Random Forest',
  gradientBoosting: 'Gradient Boosting',
};
