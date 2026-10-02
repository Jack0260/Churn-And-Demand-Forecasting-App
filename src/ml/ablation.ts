import type { Customer, AblationResult, FeatureMeta, TrainedModel } from './types';
import { preprocess, type PreprocessedData } from './preprocessing';
import { rocAuc, prAuc, mae, accuracy, f1Score, optimalThreshold } from './metrics';
import { trainModel, type TrainOptions } from './training';

export function runAblation(
  customers: Customer[],
  featureMeta: FeatureMeta[],
  bestModelType: TrainedModel['type'],
  options: TrainOptions,
  onProgress?: (current: number, total: number, group: string) => void
): { baseline: AblationResult; ablations: AblationResult[] } {
  const groups = [...new Set(featureMeta.map((f) => f.group))];
  const fullPreprocessed = preprocess(customers, featureMeta);

  // Train baseline (full feature set) and evaluate on a held-out split
  const { train, test } = stratifiedSplit(fullPreprocessed.y, 0.2);
  const XTrain = train.map((i) => fullPreprocessed.X[i]);
  const yTrain = train.map((i) => fullPreprocessed.y[i]);
  const XTest = test.map((i) => fullPreprocessed.X[i]);
  const yTest = test.map((i) => fullPreprocessed.y[i]);

  const baselineModel = trainModel(bestModelType, XTrain, yTrain, fullPreprocessed.featureNames, fullPreprocessed.featureMeta, options);
  const baselineProb = XTest.map((x) => {
    const features: Record<string, number> = {};
    fullPreprocessed.featureNames.forEach((fn, j) => { features[fn] = x[j]; });
    return baselineModel.predictProba(features);
  });
  const baselineThreshold = optimalThreshold(yTest, baselineProb);
  const baselinePred = baselineProb.map((p) => (p >= baselineThreshold ? 1 : 0));

  const baselineRocAuc = rocAuc(yTest, baselineProb);
  const baselinePrAuc = prAuc(yTest, baselineProb);
  const baselineMae = mae(yTest, baselineProb);
  const baselineAcc = accuracy(yTest, baselinePred);
  const baselineF1 = f1Score(yTest, baselinePred);

  const baseline: AblationResult = {
    featureGroup: 'Full Model (all features)',
    featuresRemoved: [],
    rocAuc: baselineRocAuc,
    prAuc: baselinePrAuc,
    mae: baselineMae,
    accuracy: baselineAcc,
    f1: baselineF1,
    deltaRocAuc: 0,
    deltaPrAuc: 0,
  };

  const ablations: AblationResult[] = [];
  const total = groups.length;
  let current = 0;

  for (const group of groups) {
    if (onProgress) onProgress(current, total, group);
    current++;

    const featuresToRemove = featureMeta.filter((f) => f.group === group).map((f) => f.name);
    const reducedMeta = featureMeta.filter((f) => f.group !== group);
    const reducedPreprocessed = preprocess(customers, reducedMeta);

    const rTrain = train.map((_, i) => i).filter((i) => true); // same indices
    const rXTrain = rTrain.map((i) => reducedPreprocessed.X[i]);
    const ryTrain = rTrain.map((i) => reducedPreprocessed.y[i]);
    const rXTest = test.map((i) => reducedPreprocessed.X[i]);
    const ryTest = test.map((i) => reducedPreprocessed.y[i]);

    const reducedModel = trainModel(bestModelType, rXTrain, ryTrain, reducedPreprocessed.featureNames, reducedPreprocessed.featureMeta, options);
    const reducedProb = rXTest.map((x) => {
      const features: Record<string, number> = {};
      reducedPreprocessed.featureNames.forEach((fn, j) => { features[fn] = x[j]; });
      return reducedModel.predictProba(features);
    });
    const reducedThreshold = optimalThreshold(ryTest, reducedProb);
    const reducedPred = reducedProb.map((p) => (p >= reducedThreshold ? 1 : 0));

    const rRocAuc = rocAuc(ryTest, reducedProb);
    const rPrAuc = prAuc(ryTest, reducedProb);

    ablations.push({
      featureGroup: group,
      featuresRemoved: featuresToRemove,
      rocAuc: rRocAuc,
      prAuc: rPrAuc,
      mae: mae(ryTest, reducedProb),
      accuracy: accuracy(ryTest, reducedPred),
      f1: f1Score(ryTest, reducedPred),
      deltaRocAuc: baselineRocAuc - rRocAuc,
      deltaPrAuc: baselinePrAuc - rPrAuc,
    });
  }

  // Sort ablations by impact (most damage first)
  ablations.sort((a, b) => b.deltaRocAuc - a.deltaRocAuc);

  return { baseline, ablations };
}

function stratifiedSplit(y: number[], testSize: number): { train: number[]; test: number[] } {
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

  const nTestPos = Math.floor(posIdx.length * testSize);
  const nTestNeg = Math.floor(negIdx.length * testSize);

  const test = [...posIdx.slice(0, nTestPos), ...negIdx.slice(0, nTestNeg)];
  const train = [...posIdx.slice(nTestPos), ...negIdx.slice(nTestNeg)];
  return { train, test };
}
