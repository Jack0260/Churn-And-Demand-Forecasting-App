import type { Customer, ModelMetrics, FeatureMeta, TrainedModel, ModelType, ExperimentLogEntry } from './types';
import { preprocess, encodeCustomer, type PreprocessedData } from './preprocessing';
import {
  accuracy, precision, recall, f1Score, mae, brierScore,
  rocAuc, rocCurve, prAuc, prCurve, calibrationBins, optimalThreshold,
} from './metrics';
import {
  trainNaiveBaseline, trainLogisticRegression, trainRandomForest, trainGradientBoosting,
  computeTreeFeatureImportance, getLogisticFeatureImportance,
} from './models';

export interface TrainOptions {
  logistic: { lr: number; l2: number; epochs: number };
  randomForest: { nTrees: number; maxDepth: number; maxFeatures: number; minSamplesSplit: number };
  gradientBoosting: { nTrees: number; maxDepth: number; learningRate: number };
  cvFolds: number;
  testSize: number;
}

export const defaultTrainOptions: TrainOptions = {
  logistic: { lr: 0.05, l2: 0.01, epochs: 200 },
  randomForest: { nTrees: 30, maxDepth: 8, maxFeatures: 0, minSamplesSplit: 10 },
  gradientBoosting: { nTrees: 50, maxDepth: 4, learningRate: 0.1 },
  cvFolds: 5,
  testSize: 0.2,
};

export interface TrainResult {
  model: TrainedModel;
  metrics: ModelMetrics;
  preprocessed: PreprocessedData;
  testIndices: number[];
}

function stratifiedSplit(y: number[], testSize: number, rng: () => number = Math.random): { train: number[]; test: number[] } {
  const posIdx: number[] = [];
  const negIdx: number[] = [];
  y.forEach((label, i) => (label === 1 ? posIdx.push(i) : negIdx.push(i)));

  for (let i = posIdx.length - 1; i > 0; i--) {
    const j = Math.floor(rng() * (i + 1));
    [posIdx[i], posIdx[j]] = [posIdx[j], posIdx[i]];
  }
  for (let i = negIdx.length - 1; i > 0; i--) {
    const j = Math.floor(rng() * (i + 1));
    [negIdx[i], negIdx[j]] = [negIdx[j], negIdx[i]];
  }

  const nTestPos = Math.floor(posIdx.length * testSize);
  const nTestNeg = Math.floor(negIdx.length * testSize);

  const test = [...posIdx.slice(0, nTestPos), ...negIdx.slice(0, nTestNeg)];
  const train = [...posIdx.slice(nTestPos), ...negIdx.slice(nTestNeg)];
  return { train, test };
}

function stratifiedKFold(y: number[], k: number): { train: number[]; test: number[] }[] {
  const posIdx: number[] = [];
  const negIdx: number[] = [];
  y.forEach((label, i) => (label === 1 ? posIdx.push(i) : negIdx.push(i)));

  for (let i = posIdx.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [posIdx[i], posIdx[j]] = [posIdx[j], posIdx[i]];
  }
  for (let i = negIdx.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [negIdx[i], negIdx[j]] = [negIdx[j], negIdx[i]];
  }

  const folds: { train: number[]; test: number[] }[] = [];
  for (let f = 0; f < k; f++) {
    const posStart = Math.floor((f * posIdx.length) / k);
    const posEnd = Math.floor(((f + 1) * posIdx.length) / k);
    const negStart = Math.floor((f * negIdx.length) / k);
    const negEnd = Math.floor(((f + 1) * negIdx.length) / k);

    const test = [...posIdx.slice(posStart, posEnd), ...negIdx.slice(negStart, negEnd)];
    const train = [...posIdx.slice(0, posStart), ...posIdx.slice(posEnd), ...negIdx.slice(0, negStart), ...negIdx.slice(negEnd)];
    folds.push({ train, test });
  }
  return folds;
}

export function trainModel(
  type: ModelType,
  X: number[][],
  y: number[],
  featureNames: string[],
  featureMeta: FeatureMeta[],
  options: TrainOptions
): TrainedModel {
  switch (type) {
    case 'naive':
      return trainNaiveBaseline(X, y, featureNames, featureMeta);
    case 'logistic':
      return trainLogisticRegression(X, y, featureNames, featureMeta, options.logistic);
    case 'randomForest':
      return trainRandomForest(X, y, featureNames, featureMeta, {
        ...options.randomForest,
        maxFeatures: options.randomForest.maxFeatures > 0 ? options.randomForest.maxFeatures : Math.max(2, Math.floor(Math.sqrt(featureNames.length))),
      });
    case 'gradientBoosting':
      return trainGradientBoosting(X, y, featureNames, featureMeta, options.gradientBoosting);
    default:
      return trainNaiveBaseline(X, y, featureNames, featureMeta);
  }
}

export function trainAndEvaluate(
  customers: Customer[],
  featureMeta: FeatureMeta[],
  modelType: ModelType,
  options: TrainOptions
): TrainResult {
  const preprocessed = preprocess(customers, featureMeta);
  const { X, y, featureNames } = preprocessed;

  const { train, test } = stratifiedSplit(y, options.testSize);
  const XTrain = train.map((i) => X[i]);
  const yTrain = train.map((i) => y[i]);

  const start = performance.now();
  const model = trainModel(modelType, XTrain, yTrain, featureNames, featureMeta, options);
  const trainTimeMs = performance.now() - start;

  const XTest = test.map((i) => X[i]);
  const yTest = test.map((i) => y[i]);

  const yProb = model.predictProba ? XTest.map((x) => {
    const features: Record<string, number> = {};
    featureNames.forEach((fn, j) => { features[fn] = x[j]; });
    return model.predictProba(features);
  }) : XTest.map(() => 0.5);

  const threshold = optimalThreshold(yTest, yProb);
  const yPred = yProb.map((p) => (p >= threshold ? 1 : 0));

  // Cross-validation
  const folds = stratifiedKFold(y, options.cvFolds);
  const cvScores: number[] = [];
  for (const fold of folds) {
    const Xf = fold.train.map((i) => X[i]);
    const yf = fold.train.map((i) => y[i]);
    const Xv = fold.test.map((i) => X[i]);
    const yv = fold.test.map((i) => y[i]);
    const foldModel = trainModel(modelType, Xf, yf, featureNames, featureMeta, options);
    const foldProb = Xv.map((x) => {
      const features: Record<string, number> = {};
      featureNames.forEach((fn, j) => { features[fn] = x[j]; });
      return foldModel.predictProba(features);
    });
    cvScores.push(rocAuc(yv, foldProb));
  }

  const cvMean = cvScores.reduce((a, b) => a + b, 0) / cvScores.length;
  const cvStd = Math.sqrt(cvScores.reduce((a, b) => a + (b - cvMean) ** 2, 0) / cvScores.length);

  // Feature importance
  let featureImportance: { feature: string; importance: number }[] = [];
  if (modelType === 'logistic') {
    featureImportance = getLogisticFeatureImportance(model as TrainedModel & { weights?: number[] });
  } else if (modelType === 'randomForest' || modelType === 'gradientBoosting') {
    featureImportance = computeTreeFeatureImportance(model as TrainedModel & { trees?: import('./decisionTree').DecisionTree[] });
  }

  const metrics: ModelMetrics = {
    modelType,
    modelName: model.name,
    accuracy: accuracy(yTest, yPred),
    precision: precision(yTest, yPred),
    recall: recall(yTest, yPred),
    f1: f1Score(yTest, yPred),
    rocAuc: rocAuc(yTest, yProb),
    prAuc: prAuc(yTest, yProb),
    mae: mae(yTest, yProb),
    brierScore: brierScore(yTest, yProb),
    confusionMatrix: {
      tn: yTest.filter((y, i) => y === 0 && yPred[i] === 0).length,
      fp: yTest.filter((y, i) => y === 0 && yPred[i] === 1).length,
      fn: yTest.filter((y, i) => y === 1 && yPred[i] === 0).length,
      tp: yTest.filter((y, i) => y === 1 && yPred[i] === 1).length,
    },
    rocCurve: rocCurve(yTest, yProb),
    prCurve: prCurve(yTest, yProb),
    calibrationBins: calibrationBins(yTest, yProb),
    featureImportance,
    cvScores,
    cvMean,
    cvStd,
    beatsBaseline: true, // Updated by caller after comparing with naive
    trainTimeMs,
  };

  return { model, metrics, preprocessed, testIndices: test };
}

export function trainAllModels(
  customers: Customer[],
  featureMeta: FeatureMeta[],
  options: TrainOptions
): { results: Record<ModelType, TrainResult>; baselineMetrics: ModelMetrics; experimentEntries: ExperimentLogEntry[] } {
  const allTypes: ModelType[] = ['naive', 'logistic', 'randomForest', 'gradientBoosting'];
  const results: Record<string, TrainResult> = {};
  const experimentEntries: ExperimentLogEntry[] = [];

  for (const type of allTypes) {
    const result = trainAndEvaluate(customers, featureMeta, type, options);
    results[type] = result;

    if (type !== 'naive') {
      const naiveRocAuc = results['naive']?.metrics.rocAuc ?? 0.5;
      result.metrics.beatsBaseline = result.metrics.rocAuc > naiveRocAuc;
    }

    experimentEntries.push({
      id: `exp-${Date.now()}-${type}`,
      timestamp: new Date().toISOString(),
      modelType: type,
      modelName: result.model.name,
      hyperparameters: result.model.hyperparameters,
      metrics: {
        rocAuc: result.metrics.rocAuc,
        prAuc: result.metrics.prAuc,
        mae: result.metrics.mae,
        accuracy: result.metrics.accuracy,
        f1: result.metrics.f1,
        cvMean: result.metrics.cvMean,
        cvStd: result.metrics.cvStd,
      },
      datasetSize: customers.length,
      featureCount: result.model.featureNames.length,
    });
  }

  return {
    results: results as Record<ModelType, TrainResult>,
    baselineMetrics: results['naive'].metrics,
    experimentEntries,
  };
}

export function predictCustomer(customer: Customer, model: TrainedModel, preprocessed: PreprocessedData): number {
  const features = encodeCustomer(customer, preprocessed.encoders, preprocessed.featureMeta);
  return model.predictProba(features);
}
