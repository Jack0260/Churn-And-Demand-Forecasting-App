import { createContext, useContext, useState, useEffect, useCallback, type ReactNode } from 'react';
import type { Customer, ModelMetrics, ExperimentLogEntry, AblationResult, TrainedModel, ModelType } from '@/ml/types';
import { generateDataset, getFeatureMeta } from '@/ml/dataset';
import { trainAllModels, type TrainOptions, type TrainResult } from '@/ml/training';
import { runAblation } from '@/ml/ablation';
import { supabase } from '@/lib/supabase';

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
  dbConnected: boolean;
}

const AppContext = createContext<AppState | null>(null);

interface DbExperimentRow {
  id: string;
  run_id: string;
  timestamp: string;
  model_type: string;
  model_name: string;
  hyperparameters: Record<string, number | string | boolean>;
  roc_auc: number;
  pr_auc: number;
  mae: number;
  accuracy: number;
  f1: number;
  cv_mean: number;
  cv_std: number;
  dataset_size: number;
  feature_count: number;
}

interface DbAblationRow {
  id: string;
  run_id: string;
  feature_group: string;
  features_removed: string[];
  is_baseline: boolean;
  roc_auc: number;
  pr_auc: number;
  mae: number;
  accuracy: number;
  f1: number;
  delta_roc_auc: number;
  delta_pr_auc: number;
}

function dbRowToExperiment(row: DbExperimentRow): ExperimentLogEntry {
  return {
    id: row.id,
    timestamp: row.timestamp,
    modelType: row.model_type as ModelType,
    modelName: row.model_name,
    hyperparameters: row.hyperparameters,
    metrics: {
      rocAuc: row.roc_auc,
      prAuc: row.pr_auc,
      mae: row.mae,
      accuracy: row.accuracy,
      f1: row.f1,
      cvMean: row.cv_mean,
      cvStd: row.cv_std,
    },
    datasetSize: row.dataset_size,
    featureCount: row.feature_count,
  };
}

function dbRowToAblation(row: DbAblationRow): AblationResult {
  return {
    featureGroup: row.feature_group,
    featuresRemoved: row.features_removed,
    rocAuc: row.roc_auc,
    prAuc: row.pr_auc,
    mae: row.mae,
    accuracy: row.accuracy,
    f1: row.f1,
    deltaRocAuc: row.delta_roc_auc,
    deltaPrAuc: row.delta_pr_auc,
  };
}

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
  const [dbConnected, setDbConnected] = useState(false);

  // Load experiment log from Supabase on mount
  useEffect(() => {
    (async () => {
      try {
        const { data, error } = await supabase
          .from('experiments')
          .select('*')
          .order('timestamp', { ascending: false })
          .limit(200);

        if (error) {
          console.error('Failed to load experiments from database:', error.message);
          setDbConnected(false);
          return;
        }

        if (data) {
          const entries = (data as DbExperimentRow[]).map(dbRowToExperiment);
          setExperimentLog(entries.reverse()); // reverse to get chronological order
        }
        setDbConnected(true);
      } catch (err) {
        console.error('Database connection failed:', err);
        setDbConnected(false);
      }
    })();
  }, []);

  // Load latest ablation results from Supabase on mount
  useEffect(() => {
    (async () => {
      try {
        const { data, error } = await supabase
          .from('ablation_results')
          .select('*')
          .order('created_at', { ascending: false })
          .limit(20);

        if (error || !data || data.length === 0) return;

        const rows = data as DbAblationRow[];
        const latestRunId = rows[0].run_id;
        const runRows = rows.filter((r) => r.run_id === latestRunId);
        const baselineRow = runRows.find((r) => r.is_baseline);
        const ablationRows = runRows.filter((r) => !r.is_baseline);

        if (baselineRow) {
          setAblationResults({
            baseline: dbRowToAblation(baselineRow),
            ablations: ablationRows.map(dbRowToAblation),
          });
        }
      } catch {
        // ignore
      }
    })();
  }, []);

  const train = useCallback(async (options: TrainOptions) => {
    setIsTraining(true);
    setTrainingProgress('Training models...');

    await new Promise((r) => setTimeout(r, 50));

    const { results, experimentEntries } = trainAllModels(customers, featureMeta, options);

    setTrainResults(results);

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

    // Persist experiment entries to Supabase
    const runId = `run-${Date.now()}`;
    const dbRows = experimentEntries.map((e) => ({
      id: e.id,
      run_id: runId,
      timestamp: e.timestamp,
      model_type: e.modelType,
      model_name: e.modelName,
      hyperparameters: e.hyperparameters,
      roc_auc: e.metrics.rocAuc,
      pr_auc: e.metrics.prAuc,
      mae: e.metrics.mae,
      accuracy: e.metrics.accuracy,
      f1: e.metrics.f1,
      cv_mean: e.metrics.cvMean,
      cv_std: e.metrics.cvStd,
      dataset_size: e.datasetSize,
      feature_count: e.featureCount,
    }));

    try {
      const { error } = await supabase.from('experiments').insert(dbRows);
      if (error) {
        console.error('Failed to save experiments to database:', error.message);
      } else {
        setDbConnected(true);
      }
    } catch (err) {
      console.error('Failed to save experiments:', err);
    }

    setExperimentLog((prev) => [...prev, ...experimentEntries]);

    setTrainingProgress('');
    setIsTraining(false);
  }, [customers, featureMeta]);

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

    // Persist ablation results to Supabase
    const runId = `ablation-${Date.now()}`;
    const allRows = [
      {
        run_id: runId,
        feature_group: result.baseline.featureGroup,
        features_removed: result.baseline.featuresRemoved,
        is_baseline: true,
        roc_auc: result.baseline.rocAuc,
        pr_auc: result.baseline.prAuc,
        mae: result.baseline.mae,
        accuracy: result.baseline.accuracy,
        f1: result.baseline.f1,
        delta_roc_auc: result.baseline.deltaRocAuc,
        delta_pr_auc: result.baseline.deltaPrAuc,
      },
      ...result.ablations.map((a) => ({
        run_id: runId,
        feature_group: a.featureGroup,
        features_removed: a.featuresRemoved,
        is_baseline: false,
        roc_auc: a.rocAuc,
        pr_auc: a.prAuc,
        mae: a.mae,
        accuracy: a.accuracy,
        f1: a.f1,
        delta_roc_auc: a.deltaRocAuc,
        delta_pr_auc: a.deltaPrAuc,
      })),
    ];

    try {
      const { error } = await supabase.from('ablation_results').insert(allRows);
      if (error) {
        console.error('Failed to save ablation results to database:', error.message);
      }
    } catch (err) {
      console.error('Failed to save ablation results:', err);
    }

    setAblationProgress('');
    setIsRunningAblation(false);
  }, [customers, featureMeta]);

  const clearExperimentLog = useCallback(async () => {
    setExperimentLog([]);
    try {
      await supabase.from('experiments').delete().neq('id', 'impossible');
    } catch (err) {
      console.error('Failed to clear experiments from database:', err);
    }
  }, []);

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
        dbConnected,
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
