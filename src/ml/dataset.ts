import type { Customer, FeatureMeta } from './types';

const featureMeta: FeatureMeta[] = [
  { name: 'gender', label: 'Gender', type: 'categorical', categories: ['Male', 'Female'], group: 'Demographics', defaultValue: 'Male' },
  { name: 'SeniorCitizen', label: 'Senior Citizen', type: 'categorical', categories: ['0', '1'], group: 'Demographics', defaultValue: '0' },
  { name: 'Partner', label: 'Partner', type: 'categorical', categories: ['Yes', 'No'], group: 'Demographics', defaultValue: 'No' },
  { name: 'Dependents', label: 'Dependents', type: 'categorical', categories: ['Yes', 'No'], group: 'Demographics', defaultValue: 'No' },
  { name: 'tenure', label: 'Tenure (months)', type: 'numeric', group: 'Account', defaultValue: 12 },
  { name: 'PhoneService', label: 'Phone Service', type: 'categorical', categories: ['Yes', 'No'], group: 'Services', defaultValue: 'Yes' },
  { name: 'MultipleLines', label: 'Multiple Lines', type: 'categorical', categories: ['Yes', 'No', 'No phone service'], group: 'Services', defaultValue: 'No' },
  { name: 'InternetService', label: 'Internet Service', type: 'categorical', categories: ['DSL', 'Fiber optic', 'No'], group: 'Services', defaultValue: 'DSL' },
  { name: 'OnlineSecurity', label: 'Online Security', type: 'categorical', categories: ['Yes', 'No', 'No internet service'], group: 'Services', defaultValue: 'No' },
  { name: 'OnlineBackup', label: 'Online Backup', type: 'categorical', categories: ['Yes', 'No', 'No internet service'], group: 'Services', defaultValue: 'No' },
  { name: 'DeviceProtection', label: 'Device Protection', type: 'categorical', categories: ['Yes', 'No', 'No internet service'], group: 'Services', defaultValue: 'No' },
  { name: 'TechSupport', label: 'Tech Support', type: 'categorical', categories: ['Yes', 'No', 'No internet service'], group: 'Services', defaultValue: 'No' },
  { name: 'StreamingTV', label: 'Streaming TV', type: 'categorical', categories: ['Yes', 'No', 'No internet service'], group: 'Services', defaultValue: 'No' },
  { name: 'StreamingMovies', label: 'Streaming Movies', type: 'categorical', categories: ['Yes', 'No', 'No internet service'], group: 'Services', defaultValue: 'No' },
  { name: 'Contract', label: 'Contract Type', type: 'categorical', categories: ['Month-to-month', 'One year', 'Two year'], group: 'Account', defaultValue: 'Month-to-month' },
  { name: 'PaperlessBilling', label: 'Paperless Billing', type: 'categorical', categories: ['Yes', 'No'], group: 'Account', defaultValue: 'Yes' },
  { name: 'PaymentMethod', label: 'Payment Method', type: 'categorical', categories: ['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'], group: 'Account', defaultValue: 'Electronic check' },
  { name: 'MonthlyCharges', label: 'Monthly Charges ($)', type: 'numeric', group: 'Billing', defaultValue: 50 },
  { name: 'TotalCharges', label: 'Total Charges ($)', type: 'numeric', group: 'Billing', defaultValue: 600 },
];

export function getFeatureMeta(): FeatureMeta[] {
  return featureMeta;
}

export function getFeatureGroups(): string[] {
  return [...new Set(featureMeta.map((f) => f.group))];
}

function mulberry32(seed: number) {
  return () => {
    seed |= 0;
    seed = (seed + 0x6d2b79f5) | 0;
    let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function pick<T>(rng: () => number, arr: T[]): T {
  return arr[Math.floor(rng() * arr.length)];
}

function randInt(rng: () => number, min: number, max: number): number {
  return Math.floor(rng() * (max - min + 1)) + min;
}

function sigmoid(x: number): number {
  return 1 / (1 + Math.exp(-x));
}

export function generateDataset(n: number = 5000, seed: number = 42): Customer[] {
  const rng = mulberry32(seed);
  const customers: Customer[] = [];

  for (let i = 0; i < n; i++) {
    const gender = pick(rng, ['Male', 'Female'] as const);
    const SeniorCitizen = rng() < 0.16 ? 1 : 0;
    const Partner = pick(rng, ['Yes', 'No'] as const);
    const Dependents = rng() < 0.3 ? 'Yes' : 'No';
    const tenure = randInt(rng, 0, 72);
    const PhoneService = rng() < 0.9 ? 'Yes' : 'No';
    const MultipleLines = PhoneService === 'No' ? 'No phone service' : pick(rng, ['Yes', 'No'] as const);
    const InternetService = pick(rng, ['DSL', 'Fiber optic', 'No'] as const);
    const hasInternet = InternetService !== 'No';

    const onlineAddons = hasInternet ? pick(rng, ['Yes', 'No'] as const) : 'No internet service';
    const OnlineSecurity = hasInternet ? (rng() < 0.36 ? 'Yes' : 'No') : 'No internet service';
    const OnlineBackup = hasInternet ? (rng() < 0.34 ? 'Yes' : 'No') : 'No internet service';
    const DeviceProtection = hasInternet ? (rng() < 0.35 ? 'Yes' : 'No') : 'No internet service';
    const TechSupport = hasInternet ? (rng() < 0.37 ? 'Yes' : 'No') : 'No internet service';
    const StreamingTV = hasInternet ? (rng() < 0.38 ? 'Yes' : 'No') : 'No internet service';
    const StreamingMovies = hasInternet ? (rng() < 0.39 ? 'Yes' : 'No') : 'No internet service';

    const Contract = pick(rng, ['Month-to-month', 'One year', 'Two year'] as const);
    const PaperlessBilling = pick(rng, ['Yes', 'No'] as const);
    const PaymentMethod = pick(rng, ['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'] as const);

    let MonthlyCharges = 18;
    if (PhoneService === 'Yes') MonthlyCharges += 20;
    if (hasInternet) MonthlyCharges += InternetService === 'Fiber optic' ? 70 : 40;
    if (OnlineSecurity === 'Yes') MonthlyCharges += 5;
    if (OnlineBackup === 'Yes') MonthlyCharges += 5;
    if (DeviceProtection === 'Yes') MonthlyCharges += 4;
    if (TechSupport === 'Yes') MonthlyCharges += 5;
    if (StreamingTV === 'Yes') MonthlyCharges += 8;
    if (StreamingMovies === 'Yes') MonthlyCharges += 8;
    if (MultipleLines === 'Yes') MonthlyCharges += 10;
    MonthlyCharges += rng() * 10 - 5;
    MonthlyCharges = Math.max(18, Math.round(MonthlyCharges * 100) / 100);

    const TotalCharges = Math.round(tenure * MonthlyCharges * 100) / 100;

    // Churn probability model - realistic relationships
    let logit = -2.0;

    // Tenure effect: longer tenure -> less churn (strong)
    logit += -0.04 * tenure;

    // Contract effect: month-to-month churns more
    if (Contract === 'Month-to-month') logit += 1.2;
    else if (Contract === 'One year') logit += 0.3;
    // Two year: baseline (no add)

    // Internet service: fiber optic churns more
    if (InternetService === 'Fiber optic') logit += 0.6;
    else if (InternetService === 'DSL') logit -= 0.1;

    // Tech support reduces churn
    if (TechSupport === 'Yes') logit -= 0.5;

    // Online security reduces churn
    if (OnlineSecurity === 'Yes') logit -= 0.3;

    // Payment method: electronic check churns more
    if (PaymentMethod === 'Electronic check') logit += 0.5;
    else if (PaymentMethod === 'Mailed check') logit += 0.2;

    // High monthly charges increase churn
    if (MonthlyCharges > 80) logit += 0.4;
    else if (MonthlyCharges > 60) logit += 0.15;

    // Senior citizens churn slightly more
    if (SeniorCitizen === 1) logit += 0.2;

    // No partner/dependents -> slightly more churn
    if (Partner === 'No') logit += 0.15;
    if (Dependents === 'No') logit += 0.1;

    // Paperless billing slight increase
    if (PaperlessBilling === 'Yes') logit += 0.1;

    const churnProb = sigmoid(logit);
    const Churn = rng() < churnProb ? 'Yes' : 'No';

    customers.push({
      customerID: `C-${String(i + 1).padStart(4, '0')}`,
      gender,
      SeniorCitizen,
      Partner,
      Dependents,
      tenure,
      PhoneService,
      MultipleLines,
      InternetService,
      OnlineSecurity,
      OnlineBackup,
      DeviceProtection,
      TechSupport,
      StreamingTV,
      StreamingMovies,
      Contract,
      PaperlessBilling,
      PaymentMethod,
      MonthlyCharges,
      TotalCharges,
      Churn,
    });
  }

  return customers;
}
