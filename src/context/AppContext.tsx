import { createContext, useContext, useState, useEffect, useCallback, type ReactNode } from 'react';
import type { Customer, ModelMetrics, ExperimentLogEntry, AblationResult, TrainedModel, ModelType } from '@/ml/types';
import { generateDataset, getFeatureMeta } from '@/ml/dataset';
import { trainAllModels, type TrainOptions, type TrainResult } from '@/ml/training';
import { runAblation } from '@/ml/ablation';

interface AppState {
  customers: Customer[];
  featureMeta: ReturnType<typeof getFeatureMeta>;
  isTraining: boolean;
  trainingProgress: string;
  trainResults: Record<ModelType, TrainResult> | null;
  bestModelType: ModelType | null;
  experimentLog: ExperimentLogEntry[];
  ablationResults: { baseline: AblationResult; ablations: AblationResult[] } | null;
  isRunningAblation: boolean;
  ablationProgress: string;
  train: (options: TrainOptions) => Promise<void>;
  runAblationStudy: (modelType: ModelType, options: TrainOptions) => Promise<void>;
  clearExperimentLog: () => void;
  bestMetrics: ModelMetrics | null;
}

const AppContext = createContext<AppState | null>(null);

const STORAGE_KEY = 'churn_lab_experiments';

export function AppProvider({ children }: { children: ReactNode }) {
  const [customers] = useState<Customer[]>(() => generateDataset(5000, 42));
  const featureMeta = useState(() => getFeatureMeta())[0];
  const [isTraining, setIsTraining] = useState(false);
  const [trainingProgress, setTrainingProgress] = useState('');
  const [trainResults, setTrainResults] = useState<Record<ModelType, TrainResult> | null>(null);
  const [bestModelType, setBestModelType] = useState<ModelType | null>(null);
  const [experimentLog, setExperimentLog] = useState<ExperimentLogEntry[]>([]);
  const [ablationResults, setAblationResults] = useState<{ baseline: AblationResult; ablations: AblationResult[] } | null>(null);
  const [isRunningAblation, setIsRunningAblation] = useState(false);
  const [ablationProgress, setAblationProgress] = useState('');

  useEffect(() => {
    try {
      const stored = localStorage.getItem(STORAGE_KEY);
      if (stored) {
        setExperimentLog(JSON.parse(stored));
      }
    } catch {
      // ignore
    }
  }, []);

  const persistLog = useCallback((entries: ExperimentLogEntry[]) => {
    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(entries.slice(-200)));
    } catch {
      // ignore
    }
  }, []);

  const train = useCallback(async (options: TrainOptions) => {
    setIsTraining(true);
    setTrainingProgress('Training models...');

    // Yield to let UI update
    await new Promise((r) => setTimeout(r, 50));

    const { results, experimentEntries } = trainAllModels(customers, featureMeta, options);

    setTrainResults(results);

    // Find best model (highest ROC-AUC, must beat naive)
    const modelTypes: ModelType[] = ['logistic', 'randomForest', 'gradientBoosting'];
    let best: ModelType = 'logistic';
    let bestAuc = -1;
    for (const mt of modelTypes) {
      if (results[mt].metrics.rocAuc > bestAuc) {
        bestAuc = results[mt].metrics.rocAuc;
        best = mt;
      }
    }
    setBestModelType(best);

    setExperimentLog((prev) => {
      const updated = [...prev, ...experimentEntries];
      persistLog(updated);
      return updated;
    });

    setTrainingProgress('');
    setIsTraining(false);
  }, [customers, featureMeta, persistLog]);

  const runAblationStudy = useCallback(async (modelType: ModelType, options: TrainOptions) => {
    setIsRunningAblation(true);
    setAblationProgress('Starting ablation study...');

    await new Promise((r) => setTimeout(r, 50));

    const result = runAblation(
      customers,
      featureMeta,
      modelType,
      options,
      (current, total, group) => {
        setAblationProgress(`Ablation ${current + 1}/${total}: Removing "${group}" features...`);
      }
    );

    setAblationResults(result);
    setAblationProgress('');
    setIsRunningAblation(false);
  }, [customers, featureMeta]);

  const clearExperimentLog = useCallback(() => {
    setExperimentLog([]);
    persistLog([]);
  }, [persistLog]);

  const bestMetrics = bestModelType && trainResults ? trainResults[bestModelType].metrics : null;

  return (
    <AppContext.Provider
      value={{
        customers,
        featureMeta,
        isTraining,
        trainingProgress,
        trainResults,
        bestModelType,
        experimentLog,
        ablationResults,
        isRunningAblation,
        ablationProgress,
        train,
        runAblationStudy,
        clearExperimentLog,
        bestMetrics,
      }}
    >
      {children}
    </AppContext.Provider>
  );
}

export function useApp() {
  const ctx = useContext(AppContext);
  if (!ctx) throw new Error('useApp must be used within AppProvider');
  return ctx;
}
