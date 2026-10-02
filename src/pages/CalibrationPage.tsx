import { useApp } from '@/context/AppContext';
import { Card, CardHeader, PageHeader, EmptyState, Button, Badge } from '@/components/ui';
import { Gauge, ArrowRight } from 'lucide-react';
import {
  ScatterChart, Scatter, XAxis, YAxis, CartesianGrid, Tooltip, Legend,
  ResponsiveContainer, ReferenceLine, Line, ComposedChart, Bar, Cell,
} from 'recharts';
import type { PageId } from '@/components/Sidebar';
import type { ModelType } from '@/ml/types';

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

export function CalibrationPage({ onNavigate }: { onNavigate: (page: PageId) => void }) {
  const { trainResults } = useApp();

  if (!trainResults) {
    return (
      <div className="animate-fade-in">
        <PageHeader title="Calibration" subtitle="Are predicted probabilities trustworthy?" icon={<Gauge size={28} />} />
        <Card className="p-8">
          <EmptyState
            icon={<Gauge size={48} />}
            title="No models trained yet"
            subtitle="Train models first to see calibration plots."
            action={<Button onClick={() => onNavigate('training')}>Go to Training <ArrowRight size={16} className="inline ml-2" /></Button>}
          />
        </Card>
      </div>
    );
  }

  // Build calibration scatter data for each model
  const calibrationData = modelOrder.map((mt) => ({
    modelType: mt,
    name: modelNames[mt],
    color: modelColors[mt],
    bins: trainResults[mt].metrics.calibrationBins.filter((b) => b.count > 0),
  }));

  // Brier score comparison
  const brierData = modelOrder.map((mt) => ({
    name: modelNames[mt].split(' ')[0],
    brier: trainResults[mt].metrics.brierScore,
    color: modelColors[mt],
  }));

  return (
    <div className="animate-fade-in">
      <PageHeader
        title="Calibration Plots"
        subtitle="Reliability diagrams — predicted probability vs. observed churn rate"
        icon={<Gauge size={28} />}
      />

      {/* Main calibration plot - all models overlaid */}
      <Card className="mb-6">
        <CardHeader title="Reliability Diagram" subtitle="Points on the diagonal = perfectly calibrated. Deviations show over/under-confidence." />
        <div className="p-5">
          <ResponsiveContainer width="100%" height={350}>
            <ComposedChart margin={{ top: 20, right: 30, bottom: 20, left: 20 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
              <XAxis type="number" dataKey="meanPredicted" domain={[0, 1]} tick={{ fill: '#94a3b8', fontSize: 11 }} label={{ value: 'Mean Predicted Probability', position: 'insideBottom', offset: -10, fill: '#64748b', fontSize: 11 }} />
              <YAxis type="number" dataKey="meanObserved" domain={[0, 1]} tick={{ fill: '#94a3b8', fontSize: 11 }} label={{ value: 'Observed Churn Rate', angle: -90, position: 'insideLeft', fill: '#64748b', fontSize: 11 }} />
              <Tooltip
                contentStyle={{ background: '#0f172a', border: '1px solid #334155', borderRadius: '8px' }}
                formatter={(value: number, name: string) => [value.toFixed(3), name]}
              />
              <Legend wrapperStyle={{ fontSize: '11px' }} />
              <ReferenceLine segment={[{ x: 0, y: 0 }, { x: 1, y: 1 }]} stroke="#475569" strokeDasharray="5 5" label={{ value: 'Perfect Calibration', fill: '#64748b', fontSize: 10, position: 'insideTopRight' }} />

              {calibrationData.map((cd) => (
                <Scatter
                  key={cd.modelType}
                  name={cd.name}
                  data={cd.bins}
                  fill={cd.color}
                  shape={cd.modelType === 'naive' ? 'diamond' : 'circle'}
                />
              ))}
            </ComposedChart>
          </ResponsiveContainer>
        </div>
      </Card>

      {/* Per-model calibration plots */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-6">
        {modelOrder.map((mt) => {
          const bins = trainResults[mt].metrics.calibrationBins.filter((b) => b.count > 0);
          const brier = trainResults[mt].metrics.brierScore;
          return (
            <Card key={mt}>
              <CardHeader
                title={modelNames[mt]}
                subtitle={`Brier Score: ${brier.toFixed(4)} (lower = better)`}
              />
              <div className="p-5">
                <ResponsiveContainer width="100%" height={200}>
                  <ComposedChart>
                    <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                    <XAxis type="number" dataKey="meanPredicted" domain={[0, 1]} tick={{ fill: '#94a3b8', fontSize: 10 }} />
                    <YAxis type="number" dataKey="meanObserved" domain={[0, 1]} tick={{ fill: '#94a3b8', fontSize: 10 }} />
                    <Tooltip contentStyle={{ background: '#0f172a', border: '1px solid #334155', borderRadius: '8px' }} />
                    <ReferenceLine segment={[{ x: 0, y: 0 }, { x: 1, y: 1 }]} stroke="#475569" strokeDasharray="5 5" />
                    <Scatter data={bins} fill={modelColors[mt]} />
                    <Line dataKey="meanObserved" data={bins} stroke={modelColors[mt]} strokeWidth={1} dot={false} />
                  </ComposedChart>
                </ResponsiveContainer>
                <div className="mt-3 flex items-center gap-2">
                  <Badge color={brier < 0.15 ? 'emerald' : brier < 0.25 ? 'amber' : 'rose'}>
                    {brier < 0.15 ? 'Well calibrated' : brier < 0.25 ? 'Moderate' : 'Poorly calibrated'}
                  </Badge>
                  <span className="text-xs text-ink-400">{bins.length} bins with data</span>
                </div>
              </div>
            </Card>
          );
        })}
      </div>

      {/* Brier score comparison */}
      <Card>
        <CardHeader title="Brier Score Comparison" subtitle="Mean squared difference between predicted probability and actual outcome" />
        <div className="p-5">
          <ResponsiveContainer width="100%" height={200}>
            <ComposedChart data={brierData}>
              <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
              <XAxis dataKey="name" tick={{ fill: '#94a3b8', fontSize: 11 }} />
              <YAxis tick={{ fill: '#94a3b8', fontSize: 11 }} />
              <Tooltip contentStyle={{ background: '#0f172a', border: '1px solid #334155', borderRadius: '8px' }} />
              <Bar dataKey="brier" name="Brier Score" radius={[4, 4, 0, 0]}>
                {brierData.map((d, i) => (
                  <Cell key={i} fill={d.color} />
                ))}
              </Bar>
            </ComposedChart>
          </ResponsiveContainer>
        </div>
      </Card>
    </div>
  );
}
