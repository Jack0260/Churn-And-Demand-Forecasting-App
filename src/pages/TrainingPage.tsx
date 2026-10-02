import { useState } from 'react';
import { useApp } from '@/context/AppContext';
import { Card, CardHeader, PageHeader, Button, Slider, Select, Spinner, Badge } from '@/components/ui';
import { Cpu, Play, Settings, CheckCircle2, XCircle, Clock } from 'lucide-react';
import { defaultTrainOptions, type TrainOptions } from '@/ml/training';
import type { ModelType } from '@/ml/types';
import type { PageId } from '@/components/Sidebar';

const modelInfo: { type: ModelType; name: string; description: string }[] = [
  { type: 'naive', name: 'Naive Baseline', description: 'Always predicts majority class — the yardstick every model must beat' },
  { type: 'logistic', name: 'Logistic Regression', description: 'Linear model with L2 regularization and gradient descent' },
  { type: 'randomForest', name: 'Random Forest', description: 'Ensemble of decision trees with bootstrap sampling and feature subsampling' },
  { type: 'gradientBoosting', name: 'Gradient Boosting', description: 'Sequential tree building with gradient-based residuals (XGBoost-style)' },
];

export function TrainingPage({ onNavigate }: { onNavigate: (page: PageId) => void }) {
  const { isTraining, trainingProgress, train, trainResults } = useApp();
  const [options, setOptions] = useState<TrainOptions>(defaultTrainOptions);
  const [showAdvanced, setShowAdvanced] = useState(false);

  const handleTrain = async () => {
    await train(options);
  };

  const updateLogistic = (key: keyof TrainOptions['logistic'], value: number) => {
    setOptions({ ...options, logistic: { ...options.logistic, [key]: value } });
  };
  const updateRF = (key: keyof TrainOptions['randomForest'], value: number) => {
    setOptions({ ...options, randomForest: { ...options.randomForest, [key]: value } });
  };
  const updateGB = (key: keyof TrainOptions['gradientBoosting'], value: number) => {
    setOptions({ ...options, gradientBoosting: { ...options.gradientBoosting, [key]: value } });
  };

  return (
    <div className="animate-fade-in">
      <PageHeader
        title="Train Models"
        subtitle="Train, cross-validate, and compare 4 models against a naive baseline"
        icon={<Cpu size={28} />}
      />

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Model cards */}
        <div className="lg:col-span-2 space-y-4">
          {modelInfo.map((m) => {
            const result = trainResults?.[m.type];
            return (
              <Card key={m.type} className="p-5">
                <div className="flex items-start justify-between">
                  <div className="flex-1">
                    <div className="flex items-center gap-2 mb-1">
                      <h3 className="text-sm font-semibold text-ink-100">{m.name}</h3>
                      {m.type !== 'naive' && result && (
                        result.metrics.beatsBaseline ? (
                          <Badge color="emerald"><CheckCircle2 size={12} /> beats baseline</Badge>
                        ) : (
                          <Badge color="rose"><XCircle size={12} /> below baseline</Badge>
                        )
                      )}
                    </div>
                    <p className="text-xs text-ink-400 mb-3">{m.description}</p>
                    {result && (
                      <div className="grid grid-cols-4 gap-2">
                        {[
                          { label: 'ROC-AUC', value: result.metrics.rocAuc },
                          { label: 'PR-AUC', value: result.metrics.prAuc },
                          { label: 'MAE', value: result.metrics.mae },
                          { label: 'F1', value: result.metrics.f1 },
                        ].map((metric) => (
                          <div key={metric.label} className="bg-ink-800/50 rounded-lg px-2.5 py-1.5">
                            <div className="text-xs text-ink-500">{metric.label}</div>
                            <div className="text-sm font-mono text-ink-100">{metric.value.toFixed(4)}</div>
                          </div>
                        ))}
                      </div>
                    )}
                    {result && (
                      <div className="flex items-center gap-3 mt-3 text-xs text-ink-500">
                        <span className="flex items-center gap-1"><Clock size={12} /> {result.metrics.trainTimeMs.toFixed(0)}ms</span>
                        <span>CV: {result.metrics.cvMean.toFixed(4)} +/- {result.metrics.cvStd.toFixed(4)}</span>
                      </div>
                    )}
                  </div>
                </div>
              </Card>
            );
          })}
        </div>

        {/* Controls */}
        <div className="space-y-4">
          <Card>
            <CardHeader title="Training Configuration" subtitle="Hyperparameters & CV settings" icon={<Settings size={18} />} />
            <div className="p-5 space-y-4">
              <Slider label="CV Folds" value={options.cvFolds} min={2} max={10} step={1} onChange={(v) => setOptions({ ...options, cvFolds: v })} />
              <Slider label="Test Set Size" value={options.testSize * 100} min={10} max={40} step={5} unit="%" onChange={(v) => setOptions({ ...options, testSize: v / 100 })} />

              <button
                onClick={() => setShowAdvanced(!showAdvanced)}
                className="text-xs text-teal-400 hover:text-teal-300 transition-colors"
              >
                {showAdvanced ? 'Hide' : 'Show'} advanced hyperparameters
              </button>

              {showAdvanced && (
                <div className="space-y-4 pt-2 border-t border-ink-700">
                  <div className="text-xs font-semibold text-ink-300 pt-2">Logistic Regression</div>
                  <Slider label="Learning Rate" value={options.logistic.lr} min={0.01} max={0.5} step={0.01} onChange={(v) => updateLogistic('lr', v)} />
                  <Slider label="L2 Regularization" value={options.logistic.l2} min={0} max={0.1} step={0.005} onChange={(v) => updateLogistic('l2', v)} />
                  <Slider label="Epochs" value={options.logistic.epochs} min={50} max={500} step={50} onChange={(v) => updateLogistic('epochs', v)} />

                  <div className="text-xs font-semibold text-ink-300 pt-2 border-t border-ink-700">Random Forest</div>
                  <Slider label="Number of Trees" value={options.randomForest.nTrees} min={10} max={100} step={5} onChange={(v) => updateRF('nTrees', v)} />
                  <Slider label="Max Depth" value={options.randomForest.maxDepth} min={2} max={15} step={1} onChange={(v) => updateRF('maxDepth', v)} />
                  <Slider label="Min Samples Split" value={options.randomForest.minSamplesSplit} min={2} max={50} step={2} onChange={(v) => updateRF('minSamplesSplit', v)} />

                  <div className="text-xs font-semibold text-ink-300 pt-2 border-t border-ink-700">Gradient Boosting</div>
                  <Slider label="Number of Trees" value={options.gradientBoosting.nTrees} min={20} max={150} step={10} onChange={(v) => updateGB('nTrees', v)} />
                  <Slider label="Max Depth" value={options.gradientBoosting.maxDepth} min={2} max={8} step={1} onChange={(v) => updateGB('maxDepth', v)} />
                  <Slider label="Learning Rate" value={options.gradientBoosting.learningRate} min={0.01} max={0.5} step={0.01} onChange={(v) => updateGB('learningRate', v)} />
                </div>
              )}

              <Button onClick={handleTrain} disabled={isTraining} size="lg" className="w-full">
                {isTraining ? (
                  <span className="flex items-center justify-center gap-2">
                    <Spinner size="sm" /> Training...
                  </span>
                ) : (
                  <span className="flex items-center justify-center gap-2">
                    <Play size={16} /> Train All Models
                  </span>
                )}
              </Button>
            </div>
          </Card>

          {trainResults && !isTraining && (
            <Card className="p-5">
              <div className="text-sm text-ink-200 mb-3">Models trained successfully!</div>
              <div className="space-y-2">
                <Button variant="secondary" size="sm" className="w-full" onClick={() => onNavigate('comparison')}>
                  View Comparison
                </Button>
                <Button variant="secondary" size="sm" className="w-full" onClick={() => onNavigate('calibration')}>
                  Calibration Plots
                </Button>
                <Button variant="secondary" size="sm" className="w-full" onClick={() => onNavigate('explanations')}>
                  SHAP Explanations
                </Button>
              </div>
            </Card>
          )}
        </div>
      </div>

      {isTraining && trainingProgress && (
        <div className="fixed bottom-4 right-4 bg-ink-800 border border-teal-500/30 rounded-xl p-4 shadow-xl flex items-center gap-3">
          <Spinner size="sm" />
          <span className="text-sm text-ink-200">{trainingProgress}</span>
        </div>
      )}
    </div>
  );
}
