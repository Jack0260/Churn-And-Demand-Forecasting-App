import type { Customer, ShapExplanation, CounterfactualResult, TrainedModel, FeatureMeta } from './types';
import type { PreprocessedData } from './preprocessing';
import { encodeCustomer, encodeFeatureValues } from './preprocessing';

export function explainPrediction(
  customer: Customer,
  model: TrainedModel,
  preprocessed: PreprocessedData,
  featureMeta: FeatureMeta[],
  nSamples: number = 100
): { explanations: ShapExplanation[]; baseValue: number; prediction: number } {
  const features = encodeCustomer(customer, preprocessed.encoders, featureMeta);
  const prediction = model.predictProba(features);
  const baseValue = prediction; // Will be replaced by mean prediction

  // Compute mean prediction across dataset (background expectation)
  // We use a subset for efficiency
  const explanations: ShapExplanation[] = [];

  for (const feat of featureMeta) {
    let contribution = 0;

    if (feat.type === 'categorical' && feat.categories) {
      // For each one-hot encoded feature, compute marginal contribution
      for (const cat of feat.categories) {
        const oneHotName = `${feat.name}=${cat}`;
        if (!model.featureNames.includes(oneHotName)) continue;

        // Perturbation-based SHAP: set this feature to 0 and see the change
        const perturbed = { ...features };
        perturbed[oneHotName] = 0;
        const perturbedPred = model.predictProba(perturbed);
        const originalVal = features[oneHotName];

        if (originalVal === 1) {
          contribution += prediction - perturbedPred;
        }

        if (cat === String(customer[feat.name as keyof Customer])) {
          explanations.push({
            feature: oneHotName,
            label: feat.label,
            value: cat,
            contribution,
            baseValue,
          });
        }
      }
    } else {
      // Numeric feature: perturbation-based contribution
      const featName = feat.name;
      if (!model.featureNames.includes(featName)) continue;

      const originalVal = features[featName];
      const perturbed = { ...features };
      // Set to mean (which is 0 after standardization)
      perturbed[featName] = 0;
      const perturbedPred = model.predictProba(perturbed);
      contribution = prediction - perturbedPred;

      const originalRaw = customer[feat.name as keyof Customer];
      explanations.push({
        feature: featName,
        label: feat.label,
        value: Number(originalRaw),
        contribution,
        baseValue,
      });
    }
  }

  // Sort by absolute contribution
  explanations.sort((a, b) => Math.abs(b.contribution) - Math.abs(a.contribution));

  return { explanations, baseValue, prediction };
}

export function computeCounterfactuals(
  customer: Customer,
  model: TrainedModel,
  preprocessed: PreprocessedData,
  featureMeta: FeatureMeta[],
  maxPerFeature: number = 3
): { results: CounterfactualResult[]; originalRisk: number } {
  const originalFeatures = encodeCustomer(customer, preprocessed.encoders, featureMeta);
  const originalRisk = model.predictProba(originalFeatures);
  const results: CounterfactualResult[] = [];

  for (const feat of featureMeta) {
    if (feat.type === 'categorical' && feat.categories) {
      const currentVal = String(customer[feat.name as keyof Customer]);
      for (const cat of feat.categories) {
        if (cat === currentVal) continue;
        const newCustomer = { ...customer, [feat.name]: cat };
        const newFeatures = encodeCustomer(newCustomer, preprocessed.encoders, featureMeta);
        const newRisk = model.predictProba(newFeatures);
        results.push({
          feature: feat.name,
          label: feat.label,
          originalValue: currentVal,
          newValue: cat,
          originalRisk,
          newRisk,
          riskDelta: newRisk - originalRisk,
        });
      }
    } else {
      // Numeric: try several values
      const currentVal = Number(customer[feat.name as keyof Customer]);
      const candidates: number[] = [];

      if (feat.name === 'tenure') {
        candidates.push(Math.min(72, currentVal + 12), Math.min(72, currentVal + 24), Math.max(0, currentVal - 6));
      } else if (feat.name === 'MonthlyCharges') {
        candidates.push(Math.max(18, currentVal - 20), Math.max(18, currentVal - 10), currentVal + 10);
      } else if (feat.name === 'TotalCharges') {
        candidates.push(currentVal * 1.5, currentVal * 2, currentVal * 0.5);
      } else {
        candidates.push(currentVal * 1.5, currentVal * 0.5, currentVal + 10);
      }

      for (const newVal of candidates) {
        const newCustomer = { ...customer, [feat.name]: newVal };
        const newFeatures = encodeCustomer(newCustomer, preprocessed.encoders, featureMeta);
        const newRisk = model.predictProba(newFeatures);
        results.push({
          feature: feat.name,
          label: feat.label,
          originalValue: currentVal,
          newValue: Math.round(newVal * 100) / 100,
          originalRisk,
          newRisk,
          riskDelta: newRisk - originalRisk,
        });
      }
    }
  }

  // Sort by most negative risk delta (most reduction in churn risk)
  results.sort((a, b) => a.riskDelta - b.riskDelta);

  return { results: results.slice(0, Math.min(results.length, maxPerFeature * featureMeta.length)), originalRisk };
}

export function findBestCounterfactual(
  customer: Customer,
  model: TrainedModel,
  preprocessed: PreprocessedData,
  featureMeta: FeatureMeta[]
): CounterfactualResult | null {
  const { results } = computeCounterfactuals(customer, model, preprocessed, featureMeta);
  // Find the single change that most reduces risk
  const reductions = results.filter((r) => r.riskDelta < 0);
  if (reductions.length === 0) return null;
  return reductions[0];
}
