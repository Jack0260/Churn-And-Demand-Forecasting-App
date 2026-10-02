import { useApp } from '@/context/AppContext';
import { Card, CardHeader, PageHeader, EmptyState, Button, Badge, Spinner } from '@/components/ui';
import { FlaskConical, ArrowRight, Play, TrendingDown, TrendingUp } from 'lucide-react';
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell,
  ReferenceLine,
} from 'recharts';
import type { PageId } from '@/components/Sidebar';

export function AblationReportPage({ onNavigate }: { onNavigate: (page: PageId) => void }) {
  const { ablationResults, isRunningAblation, ablationProgress, runAblationStudy, bestModelType, trainResults } = useApp();

  if (!trainResults) {
    return (
      <div className="animate-fade-in">
        <PageHeader title="Ablation Report" subtitle="What happens when each feature group is removed?" icon={<FlaskConical size={28} />} />
        <Card className="p-8">
          <EmptyState
            icon={<FlaskConical size={48} />}
            title="No models trained yet"
            subtitle="Train models first, then run an ablation study to see which feature groups matter most."
            action={<Button onClick={() => onNavigate('training')}>Go to Training <ArrowRight size={16} className="inline ml-2" /></Button>}
          />
        </Card>
      </div>
    );
  }

  const modelNames: Record<string, string> = {
    naive: 'Naive Baseline', logistic: 'Logistic Regression', randomForest: 'Random Forest', gradientBoosting: 'Gradient Boosting',
  };

  const handleRunAblation = async () => {
    if (!bestModelType) return;
    // Use default options from the training
    await runAblationStudy(bestModelType, {
      logistic: { lr: 0.05, l2: 0.01, epochs: 200 },
      randomForest: { nTrees: 30, maxDepth: 8, maxFeatures: 0, minSamplesSplit: 10 },
      gradientBoosting: { nTrees: 50, maxDepth: 4, learningRate: 0.1 },
      cvFolds: 5,
      testSize: 0.2,
    });
  };

  if (isRunningAblation) {
    return (
      <div className="animate-fade-in">
        <PageHeader title="Ablation Report" subtitle="Running ablation study..." icon={<FlaskConical size={28} />} />
        <Card className="p-12">
          <div className="flex flex-col items-center gap-4">
            <Spinner size="lg" />
            <div className="text-sm text-ink-300">{ablationProgress}</div>
          </div>
        </Card>
      </div>
    );
  }

  if (!ablationResults) {
    return (
      <div className="animate-fade-in">
        <PageHeader title="Ablation Report" subtitle="What happens when each feature group is removed?" icon={<FlaskConical size={28} />} />
        <Card className="p-8">
          <EmptyState
            icon={<FlaskConical size={48} />}
            title="Ablation study not run yet"
            subtitle={`Run an ablation study using the best model (${bestModelType ? modelNames[bestModelType] : 'none'}) to see how removing each feature group impacts performance.`}
            action={
              <Button onClick={handleRunAblation} disabled={!bestModelType} size="lg">
                <Play size={16} className="inline mr-2" /> Run Ablation Study
              </Button>
            }
          />
        </Card>
      </div>
    );
  }

  const { baseline, ablations } = ablationResults;

  const rocAucData = [
    { name: 'Full Model', rocAuc: baseline.rocAuc, isBaseline: true },
    ...ablations.map((a) => ({ name: a.featureGroup, rocAuc: a.rocAuc, isBaseline: false })),
  ];

  const deltaData = ablations.map((a) => ({
    name: a.featureGroup.length > 15 ? a.featureGroup.slice(0, 13) + '...' : a.featureGroup,
    deltaRocAuc: a.deltaRocAuc,
    deltaPrAuc: a.deltaPrAuc,
  }));

  return (
    <div className="animate-fade-in">
      <div className="flex items-center justify-between mb-6">
        <PageHeader
          title="Ablation Report"
          subtitle={`Impact of removing each feature group from ${bestModelType ? modelNames[bestModelType] : 'best model'}`}
          icon={<FlaskConical size={28} />}
        />
        <Button variant="secondary" onClick={handleRunAblation}>
          <Play size={14} className="inline mr-1" /> Re-run
        </Button>
      </div>

      {/* Baseline metrics */}
      <Card className="mb-6">
        <CardHeader title="Baseline (Full Model)" subtitle="All features included" />
        <div className="p-5 grid grid-cols-2 md:grid-cols-5 gap-3">
          {[
            { label: 'ROC-AUC', value: baseline.rocAuc },
            { label: 'PR-AUC', value: baseline.prAuc },
            { label: 'MAE', value: baseline.mae },
            { label: 'Accuracy', value: baseline.accuracy },
            { label: 'F1', value: baseline.f1 },
          ].map((m) => (
            <div key={m.label} className="bg-ink-800/50 rounded-lg p-3 text-center">
              <div className="text-xs text-ink-400">{m.label}</div>
              <div className="text-lg font-bold text-teal-300">{m.value.toFixed(4)}</div>
            </div>
          ))}
        </div>
      </Card>

      {/* ROC-AUC comparison bar chart */}
      <Card className="mb-6">
        <CardHeader title="ROC-AUC: Full Model vs. Ablations" subtitle="Lower bars = that feature group was important" />
        <div className="p-5">
          <ResponsiveContainer width="100%" height={300}>
            <BarChart data={rocAucData} margin={{ left: 20, bottom: 60 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
              <XAxis dataKey="name" tick={{ fill: '#94a3b8', fontSize: 10 }} angle={-25} textAnchor="end" height={70} interval={0} />
              <YAxis domain={[0.5, 1]} tick={{ fill: '#94a3b8', fontSize: 11 }} />
              <Tooltip contentStyle={{ background: '#0f172a', border: '1px solid #334155', borderRadius: '8px' }} />
              <ReferenceLine y={baseline.rocAuc} stroke="#14b8a6" strokeDasharray="5 5" label={{ value: 'Baseline', fill: '#14b8a6', fontSize: 10 }} />
              <Bar dataKey="rocAuc" radius={[4, 4, 0, 0]}>
                {rocAucData.map((d, i) => (
                  <Cell key={i} fill={d.isBaseline ? '#14b8a6' : d.rocAuc < baseline.rocAuc - 0.02 ? '#f43f5e' : '#475569'} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
      </Card>

      {/* Delta chart */}
      <Card className="mb-6">
        <CardHeader title="Performance Drop When Feature Group Removed" subtitle="Positive = removing that group hurt the model (it was important)" />
        <div className="p-5">
          <ResponsiveContainer width="100%" height={280}>
            <BarChart data={deltaData} layout="vertical" margin={{ left: 80 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
              <XAxis type="number" tick={{ fill: '#94a3b8', fontSize: 10 }} />
              <YAxis type="category" dataKey="name" tick={{ fill: '#94a3b8', fontSize: 10 }} width={80} />
              <Tooltip contentStyle={{ background: '#0f172a', border: '1px solid #334155', borderRadius: '8px' }} />
              <ReferenceLine x={0} stroke="#475569" />
              <Bar dataKey="deltaRocAuc" name="ROC-AUC Drop" radius={[0, 4, 4, 0]}>
                {deltaData.map((d, i) => (
                  <Cell key={i} fill={d.deltaRocAuc > 0 ? '#f43f5e' : '#10b981'} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
      </Card>

      {/* Detailed table */}
      <Card>
        <CardHeader title="Detailed Ablation Results" subtitle="All metrics for each feature group removal" />
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead className="bg-ink-800/50 text-ink-400 text-left">
              <tr>
                <th className="px-4 py-3">Feature Group</th>
                <th className="px-4 py-3">Features Removed</th>
                <th className="px-4 py-3">ROC-AUC</th>
                <th className="px-4 py-3">PR-AUC</th>
                <th className="px-4 py-3">MAE</th>
                <th className="px-4 py-3">Accuracy</th>
                <th className="px-4 py-3">F1</th>
                <th className="px-4 py-3">Impact</th>
              </tr>
            </thead>
            <tbody>
              <tr className="border-t border-ink-700/50 bg-teal-500/5">
                <td className="px-4 py-3 text-ink-200 font-medium">Full Model</td>
                <td className="px-4 py-3 text-ink-500 text-xs">— (baseline)</td>
                <td className="px-4 py-3 font-mono text-xs text-teal-300 font-semibold">{baseline.rocAuc.toFixed(4)}</td>
                <td className="px-4 py-3 font-mono text-xs text-teal-300">{baseline.prAuc.toFixed(4)}</td>
                <td className="px-4 py-3 font-mono text-xs text-ink-300">{baseline.mae.toFixed(4)}</td>
                <td className="px-4 py-3 font-mono text-xs text-ink-300">{baseline.accuracy.toFixed(4)}</td>
                <td className="px-4 py-3 font-mono text-xs text-ink-300">{baseline.f1.toFixed(4)}</td>
                <td className="px-4 py-3"><Badge color="teal">baseline</Badge></td>
              </tr>
              {ablations.map((a) => {
                const impact = a.deltaRocAuc;
                return (
                  <tr key={a.featureGroup} className="border-t border-ink-700/50 hover:bg-ink-800/30">
                    <td className="px-4 py-3 text-ink-200 font-medium">{a.featureGroup}</td>
                    <td className="px-4 py-3 text-ink-500 text-xs">{a.featuresRemoved.join(', ')}</td>
                    <td className="px-4 py-3 font-mono text-xs text-ink-300">{a.rocAuc.toFixed(4)}</td>
                    <td className="px-4 py-3 font-mono text-xs text-ink-300">{a.prAuc.toFixed(4)}</td>
                    <td className="px-4 py-3 font-mono text-xs text-ink-300">{a.mae.toFixed(4)}</td>
                    <td className="px-4 py-3 font-mono text-xs text-ink-300">{a.accuracy.toFixed(4)}</td>
                    <td className="px-4 py-3 font-mono text-xs text-ink-300">{a.f1.toFixed(4)}</td>
                    <td className="px-4 py-3">
                      {impact > 0.02 ? (
                        <Badge color="rose"><TrendingDown size={12} /> -{(impact * 100).toFixed(1)}%</Badge>
                      ) : impact > 0.005 ? (
                        <Badge color="amber">-{(impact * 100).toFixed(1)}%</Badge>
                      ) : (
                        <Badge color="emerald"><TrendingUp size={12} /> minimal</Badge>
                      )}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </Card>
    </div>
  );
}
