export type ModelType = 'naive' | 'logistic' | 'decisionTree' | 'randomForest' | 'gradientBoosting';

export interface Customer {
  customerID: string;
  gender: 'Male' | 'Female';
  SeniorCitizen: 0 | 1;
  Partner: 'Yes' | 'No';
  Dependents: 'Yes' | 'No';
  tenure: number;
  PhoneService: 'Yes' | 'No';
  MultipleLines: 'Yes' | 'No' | 'No phone service';
  InternetService: 'DSL' | 'Fiber optic' | 'No';
  OnlineSecurity: 'Yes' | 'No' | 'No internet service';
  OnlineBackup: 'Yes' | 'No' | 'No internet service';
  DeviceProtection: 'Yes' | 'No' | 'No internet service';
  TechSupport: 'Yes' | 'No' | 'No internet service';
  StreamingTV: 'Yes' | 'No' | 'No internet service';
  StreamingMovies: 'Yes' | 'No' | 'No internet service';
  Contract: 'Month-to-month' | 'One year' | 'Two year';
  PaperlessBilling: 'Yes' | 'No';
  PaymentMethod: 'Electronic check' | 'Mailed check' | 'Bank transfer (automatic)' | 'Credit card (automatic)';
  MonthlyCharges: number;
  TotalCharges: number;
  Churn: 'Yes' | 'No';
}

export interface FeatureMeta {
  name: string;
  label: string;
  type: 'numeric' | 'categorical';
  categories?: string[];
  group: string;
  defaultValue: string | number;
}

export interface ModelMetrics {
  modelType: ModelType;
  modelName: string;
  accuracy: number;
  precision: number;
  recall: number;
  f1: number;
  rocAuc: number;
  prAuc: number;
  mae: number;
  brierScore: number;
  confusionMatrix: { tn: number; fp: number; fn: number; tp: number };
  rocCurve: { fpr: number; tpr: number }[];
  prCurve: { recall: number; precision: number }[];
  calibrationBins: { binStart: number; binEnd: number; meanPredicted: number; meanObserved: number; count: number }[];
  featureImportance: { feature: string; importance: number }[];
  cvScores: number[];
  cvMean: number;
  cvStd: number;
  beatsBaseline: boolean;
  trainTimeMs: number;
}

export interface ExperimentLogEntry {
  id: string;
  timestamp: string;
  modelType: ModelType;
  modelName: string;
  hyperparameters: Record<string, number | string | boolean>;
  metrics: {
    rocAuc: number;
    prAuc: number;
    mae: number;
    accuracy: number;
    f1: number;
    cvMean: number;
    cvStd: number;
  };
  datasetSize: number;
  featureCount: number;
}

export interface AblationResult {
  featureGroup: string;
  featuresRemoved: string[];
  rocAuc: number;
  prAuc: number;
  mae: number;
  accuracy: number;
  f1: number;
  deltaRocAuc: number;
  deltaPrAuc: number;
}

export interface ShapExplanation {
  feature: string;
  label: string;
  value: string | number;
  contribution: number;
  baseValue: number;
}

export interface CounterfactualResult {
  feature: string;
  label: string;
  originalValue: string | number;
  newValue: string | number;
  originalRisk: number;
  newRisk: number;
  riskDelta: number;
}

export interface TrainedModel {
  type: ModelType;
  name: string;
  predictProba: (features: Record<string, number>) => number;
  predict: (features: Record<string, number>) => 0 | 1;
  featureNames: string[];
  featureMeta: FeatureMeta[];
  hyperparameters: Record<string, number | string | boolean>;
}
