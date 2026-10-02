import type { Customer, FeatureMeta } from './types';

export interface PreprocessedData {
  X: number[][];
  y: number[];
  featureNames: string[];
  featureMeta: FeatureMeta[];
  encoders: Record<string, { type: 'categorical'; categories: string[] } | { type: 'numeric'; mean: number; std: number }>;
}

export function preprocess(
  customers: Customer[],
  featureMeta: FeatureMeta[],
  excludedFeatures: string[] = []
): PreprocessedData {
  const activeFeatures = featureMeta.filter((f) => !excludedFeatures.includes(f.name));
  const featureNames: string[] = [];
  const encoders: PreprocessedData['encoders'] = {};

  // Build encoders
  for (const feat of activeFeatures) {
    if (feat.type === 'categorical' && feat.categories) {
      for (const cat of feat.categories) {
        featureNames.push(`${feat.name}=${cat}`);
      }
      encoders[feat.name] = { type: 'categorical', categories: feat.categories };
    } else {
      featureNames.push(feat.name);
      const values = customers.map((c) => Number(c[feat.name as keyof Customer]));
      const mean = values.reduce((a, b) => a + b, 0) / values.length;
      const std = Math.sqrt(values.reduce((a, b) => a + (b - mean) ** 2, 0) / values.length) || 1;
      encoders[feat.name] = { type: 'numeric', mean, std: std || 1 };
    }
  }

  // Encode
  const X = customers.map((c) => {
    const row: number[] = [];
    for (const feat of activeFeatures) {
      const val = c[feat.name as keyof Customer];
      if (feat.type === 'categorical' && feat.categories) {
        for (const cat of feat.categories) {
          row.push(String(val) === cat ? 1 : 0);
        }
      } else {
        const enc = encoders[feat.name];
        if (enc.type === 'numeric') {
          row.push((Number(val) - enc.mean) / enc.std);
        }
      }
    }
    return row;
  });

  const y = customers.map((c) => (c.Churn === 'Yes' ? 1 : 0));

  return { X, y, featureNames, featureMeta: activeFeatures, encoders };
}

export function encodeCustomer(
  customer: Customer,
  encoders: PreprocessedData['encoders'],
  featureMeta: FeatureMeta[]
): Record<string, number> {
  const features: Record<string, number> = {};
  for (const feat of featureMeta) {
    const val = customer[feat.name as keyof Customer];
    if (feat.type === 'categorical' && feat.categories) {
      for (const cat of feat.categories) {
        features[`${feat.name}=${cat}`] = String(val) === cat ? 1 : 0;
      }
    } else {
      const enc = encoders[feat.name];
      if (enc && enc.type === 'numeric') {
        features[feat.name] = (Number(val) - enc.mean) / enc.std;
      } else {
        features[feat.name] = Number(val);
      }
    }
  }
  return features;
}

export function encodeFeatureValues(
  values: Record<string, string | number>,
  encoders: PreprocessedData['encoders'],
  featureMeta: FeatureMeta[]
): Record<string, number> {
  const features: Record<string, number> = {};
  for (const feat of featureMeta) {
    const val = values[feat.name];
    if (feat.type === 'categorical' && feat.categories) {
      for (const cat of feat.categories) {
        features[`${feat.name}=${cat}`] = String(val) === cat ? 1 : 0;
      }
    } else {
      const enc = encoders[feat.name];
      if (enc && enc.type === 'numeric') {
        features[feat.name] = (Number(val) - enc.mean) / enc.std;
      } else {
        features[feat.name] = Number(val);
      }
    }
  }
  return features;
}
