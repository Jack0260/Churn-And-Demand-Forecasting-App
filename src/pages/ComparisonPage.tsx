import { useApp } from '@/context/AppContext';
import { Card, CardHeader, PageHeader, EmptyState, Button, Badge } from '@/components/ui';
import { BarChart3, ArrowRight, Trophy, AlertTriangle } from 'lucide-react';
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer,
  BarChart, Bar, Cell,
} from 'recharts';
import type { PageId } from '@/components/Sidebar';
import type { ModelType, ModelMetrics } from '@/ml/types';

const modelColors: Record<ModelType, string> = {
  naive: '#64748b',
  logistic: '#14b8a6',
  decisionTree: '#fbbf24',
  randomForest: '#34d399',
  gradientBoosting: '#fb7185',
};

const modelOrder: ModelType[] = ['naive', 'logistic', 'randomForest', 'gradientBoosting'];
const modelNames: Record<ModelType, string> = {
  naive: 'Naive Baseline', logistic: 'Logistic Regression', decisionTree: 'Decision Tree',
  randomForest: 'Random Forest', gradientBoosting: 'Gradient Boosting',
};

export function ComparisonPage({ onNavigate }: { onNavigate: (page: PageId) => void }) {
  const { trainResults, bestModelType } = useApp();

  if (!trainResults) {
    return (
      <div className="animate-fade-in">
        <PageHeader title="Model Comparison" subtitle="Compare all models on ROC-AUC, PR-AUC, MAE, and more" icon={<BarChart3 size={28} />} />
        <Card className="p-8">
          <EmptyState
            icon={<BarChart3 size={48} />}
            title="No models trained yet"
            subtitle="Train models first to see the comparison."
            action={<Button onClick={() => onNavigate('training')}>Go to Training <ArrowRight size={16} className="inline ml-2" /></Button>}
          />
        </Card>
      </div>
    );
  }

  const allMetrics: ModelMetrics[] = modelOrder.map((mt) => trainResults[mt].metrics);

  // ROC curves
  const rocData: { fpr: number; [key: string]: number }[] = (() => {
    const maxPoints = 100;
    const result: { fpr: number; [key: string]: number }[] = [];
    for (let i = 0; i <= maxPoints; i++) {
      const fpr = i / maxPoints;
      const row: { fpr: number; [key: string]: number } = { fpr };
      for (const mt of modelOrder) {
        const curve = trainResults[mt].metrics.rocCurve;
        // Interpolate tpr at this fpr
        let tpr = 0;
        for (let j = 0; j < curve.length; j++) {
          if (curve[j].fpr >= fpr) {
            if (j === 0) { tpr = curve[0].tpr; break; }
            const prev = curve[j - 1];
            const curr = curve[j];
            const ratio = (fpr - prev.fpr) / (curr.fpr - prev.fpr || 1);
            tpr = prev.tpr + ratio * (curr.tpr - prev.tpr);
            break;
          }
          tpr = curve[curve.length - 1].tpr;
        }
        row[modelNames[mt]] = tpr;
      }
      result.push(row);
    }
    return result;
  })();

  // PR curves
  const prData: { recall: number; [key: string]: number }[] = (() => {
    const maxPoints = 100;
    const result: { recall: number; [key: string]: number }[] = [];
    for (let i = 0; i <= maxPoints; i++) {
      const recall = i / maxPoints;
      const row: { recall: number; [key: string]: number } = { recall };
      for (const mt of modelOrder) {
        const curve = trainResults[mt].metrics.prCurve;
        let precision = curve[0]?.precision || 1;
        for (let j = 0; j < curve.length; j++) {
          if (curve[j].recall >= recall) {
            if (j === 0) { precision = curve[0].precision; break; }
            const prev = curve[j - 1];
            const curr = curve[j];
            const ratio = (recall - prev.recall) / (curr.recall - prev.recall || 1);
            precision = prev.precision + ratio * (curr.precision - prev.precision);
            break;
          }
          precision = curve[curve.length - 1].precision;
        }
        row[modelNames[mt]] = precision;
      }
      result.push(row);
    }
    return result;
  })();

  // Metrics table data
  const metricRows = ['rocAuc', 'prAuc', 'mae', 'accuracy', 'precision', 'recall', 'f1', 'brierScore', 'cvMean'];
  const metricLabels: Record<string, string> = {
    rocAuc: 'ROC-AUC', prAuc: 'PR-AUC', mae: 'MAE', accuracy: 'Accuracy', precision: 'Precision',
    recall: 'Recall', f1: 'F1 Score', brierScore: 'Brier Score', cvMean: 'CV Mean',
  };

  // Feature importance comparison
  const bestMetrics = bestModelType ? trainResults[bestModelType].metrics : null;
  const topFeatures = bestMetrics?.featureImportance.slice(0, 12) || [];

  // Find best metric value per row
  const bestPerMetric: Record<string, number> = {};
  for (const row of metricRows) {
    const values = allMetrics.map((m) => m[row as keyof ModelMetrics] as number);
    bestPerMetric[row] = row === 'mae' || row === 'brierScore' ? Math.min(...values) : Math.max(...values);
  }

  // CV scores bar data
  const cvBarData = modelOrder.map((mt) => ({
    name: modelNames[mt].split(' ')[0],
    mean: trainResults[mt].metrics.cvMean,
    std: trainResults[mt].metrics.cvStd,
  }));

  return (
    <div className="animate-fade-in">
      <PageHeader
        title="Model Comparison"
        subtitle="Cross-validated performance across ROC-AUC, PR-AUC, MAE, and more"
        icon={<BarChart3 size={28} />}
      />

      {/* Metrics table */}
      <Card className="mb-6 overflow-x-auto">
        <CardHeader title="Performance Metrics" subtitle="Best value per metric highlighted in teal" icon={<Trophy size={18} />} />
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead className="bg-ink-800/50 text-ink-400 text-left">
              <tr>
                <th className="px-4 py-3">Metric</th>
                {modelOrder.map((mt) => (
                  <th key={mt} className="px-4 py-3">
                    <div className="flex items-center gap-2">
                      <span className="w-2 h-2 rounded-full" style={{ background: modelColors[mt] }} />
                      {modelNames[mt]}
                    </div>
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {metricRows.map((row) => (
                <tr key={row} className="border-t border-ink-700/50 hover:bg-ink-800/30">
                  <td className="px-4 py-2.5 text-ink-200 font-medium">{metricLabels[row]}</td>
                  {allMetrics.map((m) => {
                    const val = m[row as keyof ModelMetrics] as number;
                    const isBest = val === bestPerMetric[row];
                    return (
                      <td key={m.modelType} className="px-4 py-2.5">
                        <span className={isBest ? 'text-teal-300 font-mono font-semibold' : 'text-ink-300 font-mono'}>
                          {val.toFixed(4)}
                        </span>
                      </td>
                    );
                  })}
                </tr>
              ))}
              <tr className="border-t border-ink-700/50">
                <td className="px-4 py-2.5 text-ink-200 font-medium">Beats Baseline?</td>
                {modelOrder.map((mt) => (
                  <td key={mt} className="px-4 py-2.5">
                    {mt === 'naive' ? <span className="text-ink-500 text-xs">baseline</span> :
                      trainResults[mt].metrics.beatsBaseline ?
                        <Badge color="emerald">Yes</Badge> :
                        <Badge color="rose">No</Badge>
                    }
                  </td>
                ))}
              </tr>
            </tbody>
          </table>
        </div>
      </Card>

      {/* ROC and PR curves */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-6">
        <Card>
          <CardHeader title="ROC Curves" subtitle="True positive rate vs. false positive rate" />
          <div className="p-5">
            <ResponsiveContainer width="100%" height={280}>
              <LineChart data={rocData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                <XAxis dataKey="fpr" type="number" domain={[0, 1]} tick={{ fill: '#94a3b8', fontSize: 11 }} label={{ value: 'False Positive Rate', position: 'insideBottom', offset: -5, fill: '#64748b', fontSize: 11 }} />
                <YAxis domain={[0, 1]} tick={{ fill: '#94a3b8', fontSize: 11 }} label={{ value: 'True Positive Rate', angle: -90, position: 'insideLeft', fill: '#64748b', fontSize: 11 }} />
                <Tooltip contentStyle={{ background: '#0f172a', border: '1px solid #334155', borderRadius: '8px' }} />
                <Legend wrapperStyle={{ fontSize: '11px' }} />
                {modelOrder.map((mt) => (
                  <Line key={mt} type="monotone" dataKey={modelNames[mt]} stroke={modelColors[mt]} strokeWidth={mt === bestModelType ? 2.5 : 1.5} dot={false} strokeDasharray={mt === 'naive' ? '5 5' : undefined} />
                ))}
              </LineChart>
            </ResponsiveContainer>
          </div>
        </Card>

        <Card>
          <CardHeader title="PR Curves" subtitle="Precision vs. recall" />
          <div className="p-5">
            <ResponsiveContainer width="100%" height={280}>
              <LineChart data={prData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                <XAxis dataKey="recall" type="number" domain={[0, 1]} tick={{ fill: '#94a3b8', fontSize: 11 }} label={{ value: 'Recall', position: 'insideBottom', offset: -5, fill: '#64748b', fontSize: 11 }} />
                <YAxis domain={[0, 1]} tick={{ fill: '#94a3b8', fontSize: 11 }} label={{ value: 'Precision', angle: -90, position: 'insideLeft', fill: '#64748b', fontSize: 11 }} />
                <Tooltip contentStyle={{ background: '#0f172a', border: '1px solid #334155', borderRadius: '8px' }} />
                <Legend wrapperStyle={{ fontSize: '11px' }} />
                {modelOrder.map((mt) => (
                  <Line key={mt} type="monotone" dataKey={modelNames[mt]} stroke={modelColors[mt]} strokeWidth={mt === bestModelType ? 2.5 : 1.5} dot={false} strokeDasharray={mt === 'naive' ? '5 5' : undefined} />
                ))}
              </LineChart>
            </ResponsiveContainer>
          </div>
        </Card>
      </div>

      {/* CV scores and confusion matrices */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-6">
        <Card>
          <CardHeader title="Cross-Validation Scores" subtitle="ROC-AUC across folds with mean and std" />
          <div className="p-5">
            <ResponsiveContainer width="100%" height={240}>
              <BarChart data={cvBarData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                <XAxis dataKey="name" tick={{ fill: '#94a3b8', fontSize: 11 }} />
                <YAxis domain={[0.4, 1]} tick={{ fill: '#94a3b8', fontSize: 11 }} />
                <Tooltip contentStyle={{ background: '#0f172a', border: '1px solid #334155', borderRadius: '8px' }} />
                <Bar dataKey="mean" name="CV Mean ROC-AUC" radius={[4, 4, 0, 0]}>
                  {modelOrder.map((mt, i) => (
                    <Cell key={i} fill={modelColors[mt]} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        </Card>

        <Card>
          <CardHeader title="Confusion Matrices" subtitle="True vs. predicted labels (test set)" />
          <div className="p-5 grid grid-cols-2 gap-4">
            {modelOrder.filter((mt) => mt !== 'naive').map((mt) => {
              const cm = trainResults[mt].metrics.confusionMatrix;
              return (
                <div key={mt} className="bg-ink-800/50 rounded-lg p-3">
                  <div className="text-xs text-ink-300 font-medium mb-2">{modelNames[mt]}</div>
                  <div className="grid grid-cols-2 gap-1 text-center text-xs">
                    <div className="bg-emerald-500/10 rounded p-2">
                      <div className="text-ink-500">TN</div>
                      <div className="text-emerald-300 font-mono font-bold">{cm.tn}</div>
                    </div>
                    <div className="bg-rose-500/10 rounded p-2">
                      <div className="text-ink-500">FP</div>
                      <div className="text-rose-300 font-mono font-bold">{cm.fp}</div>
                    </div>
                    <div className="bg-amber-500/10 rounded p-2">
                      <div className="text-ink-500">FN</div>
                      <div className="text-amber-300 font-mono font-bold">{cm.fn}</div>
                    </div>
                    <div className="bg-teal-500/10 rounded p-2">
                      <div className="text-ink-500">TP</div>
                      <div className="text-teal-300 font-mono font-bold">{cm.tp}</div>
                    </div>
                  </div>
                </div>
              );
            })}
          </div>
        </Card>
      </div>

      {/* Feature importance */}
      {bestMetrics && topFeatures.length > 0 && (
        <Card>
          <CardHeader
            title={`Feature Importance — ${bestModelType ? modelNames[bestModelType] : ''}`}
            subtitle="Top features driving model predictions"
            icon={<AlertTriangle size={18} />}
          />
          <div className="p-5">
            <ResponsiveContainer width="100%" height={300}>
              <BarChart data={topFeatures} layout="vertical" margin={{ left: 100 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                <XAxis type="number" tick={{ fill: '#94a3b8', fontSize: 11 }} />
                <YAxis type="category" dataKey="feature" tick={{ fill: '#94a3b8', fontSize: 10 }} width={100} />
                <Tooltip contentStyle={{ background: '#0f172a', border: '1px solid #334155', borderRadius: '8px' }} />
                <Bar dataKey="importance" fill="#14b8a6" radius={[0, 4, 4, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </Card>
      )}
    </div>
  );
}
