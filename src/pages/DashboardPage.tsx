import { useApp } from '@/context/AppContext';
import { Card, CardHeader, MetricBadge, PageHeader, EmptyState, Button } from '@/components/ui';
import { LayoutDashboard, Cpu, TrendingUp, Database, Activity, CheckCircle2, XCircle, ArrowRight, BarChart3 } from 'lucide-react';
import {
  PieChart, Pie, Cell, ResponsiveContainer, BarChart, Bar, XAxis, YAxis,
  CartesianGrid, Tooltip, Legend,
} from 'recharts';
import type { PageId } from '@/components/Sidebar';

export function DashboardPage({ onNavigate }: { onNavigate: (page: PageId) => void }) {
  const { customers, trainResults, bestModelType, experimentLog, isTraining, bestMetrics } = useApp();

  const churned = customers.filter((c) => c.Churn === 'Yes').length;
  const retained = customers.length - churned;
  const churnRate = (churned / customers.length) * 100;

  const pieData = [
    { name: 'Retained', value: retained, color: '#14b8a6' },
    { name: 'Churned', value: churned, color: '#f43f5e' },
  ];

  const modelTypes = ['naive', 'logistic', 'randomForest', 'gradientBoosting'] as const;
  const modelNames: Record<string, string> = {
    naive: 'Naive Baseline', logistic: 'Logistic Regression', randomForest: 'Random Forest', gradientBoosting: 'Gradient Boosting',
  };
  const barData = trainResults
    ? modelTypes.map((mt) => ({ name: modelNames[mt], rocAuc: trainResults[mt].metrics.rocAuc, prAuc: trainResults[mt].metrics.prAuc }))
    : [];

  return (
    <div className="animate-fade-in">
      <PageHeader
        title="Dashboard"
        subtitle="Overview of your churn prediction modeling pipeline"
        icon={<LayoutDashboard size={28} />}
      />

      {/* Quick stats */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
        <Card className="p-4">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 bg-blue-500/10 rounded-lg flex items-center justify-center">
              <Database size={20} className="text-blue-400" />
            </div>
            <div>
              <div className="text-xs text-ink-400">Dataset Size</div>
              <div className="text-xl font-bold text-ink-100">{customers.length.toLocaleString()}</div>
            </div>
          </div>
        </Card>
        <Card className="p-4">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 bg-rose-500/10 rounded-lg flex items-center justify-center">
              <TrendingUp size={20} className="text-rose-400" />
            </div>
            <div>
              <div className="text-xs text-ink-400">Churn Rate</div>
              <div className="text-xl font-bold text-ink-100">{churnRate.toFixed(1)}%</div>
            </div>
          </div>
        </Card>
        <Card className="p-4">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 bg-teal-500/10 rounded-lg flex items-center justify-center">
              <Activity size={20} className="text-teal-400" />
            </div>
            <div>
              <div className="text-xs text-ink-400">Models Trained</div>
              <div className="text-xl font-bold text-ink-100">{trainResults ? 4 : 0}</div>
            </div>
          </div>
        </Card>
        <Card className="p-4">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 bg-amber-500/10 rounded-lg flex items-center justify-center">
              <Cpu size={20} className="text-amber-400" />
            </div>
            <div>
              <div className="text-xs text-ink-400">Experiments Logged</div>
              <div className="text-xl font-bold text-ink-100">{experimentLog.length}</div>
            </div>
          </div>
        </Card>
      </div>

      {!trainResults ? (
        <Card className="p-8">
          <EmptyState
            icon={<Cpu size={48} />}
            title="No models trained yet"
            subtitle="Train your first set of models to see performance comparisons, calibration plots, SHAP explanations, and ablation studies."
            action={
              <Button onClick={() => onNavigate('training')} size="lg">
                Go to Training <ArrowRight size={16} className="inline ml-2" />
              </Button>
            }
          />
        </Card>
      ) : (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Class distribution */}
          <Card>
            <CardHeader title="Class Distribution" subtitle="Churned vs. retained customers" icon={<Database size={18} />} />
            <div className="p-5">
              <ResponsiveContainer width="100%" height={220}>
                <PieChart>
                  <Pie data={pieData} dataKey="value" nameKey="name" cx="50%" cy="50%" innerRadius={50} outerRadius={80} paddingAngle={3}>
                    {pieData.map((entry) => (
                      <Cell key={entry.name} fill={entry.color} />
                    ))}
                  </Pie>
                  <Tooltip
                    contentStyle={{ background: '#0f172a', border: '1px solid #334155', borderRadius: '8px', color: '#e2e8f0' }}
                  />
                  <Legend wrapperStyle={{ fontSize: '12px', color: '#94a3b8' }} />
                </PieChart>
              </ResponsiveContainer>
            </div>
          </Card>

          {/* Model comparison bar chart */}
          <Card>
            <CardHeader title="Model ROC-AUC Comparison" subtitle="All models vs. naive baseline" icon={<BarChart3 size={18} />} />
            <div className="p-5">
              <ResponsiveContainer width="100%" height={220}>
                <BarChart data={barData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                  <XAxis dataKey="name" tick={{ fill: '#94a3b8', fontSize: 10 }} interval={0} angle={-15} textAnchor="end" height={60} />
                  <YAxis domain={[0.4, 1]} tick={{ fill: '#94a3b8', fontSize: 11 }} />
                  <Tooltip contentStyle={{ background: '#0f172a', border: '1px solid #334155', borderRadius: '8px' }} />
                  <Legend wrapperStyle={{ fontSize: '12px' }} />
                  <Bar dataKey="rocAuc" name="ROC-AUC" fill="#14b8a6" radius={[4, 4, 0, 0]} />
                  <Bar dataKey="prAuc" name="PR-AUC" fill="#fbbf24" radius={[4, 4, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </Card>

          {/* Best model metrics */}
          {bestMetrics && bestModelType && (
            <Card className="lg:col-span-2">
              <CardHeader
                title={`Best Model: ${modelNames[bestModelType]}`}
                subtitle="Selected by highest ROC-AUC among models that beat the naive baseline"
                icon={<CheckCircle2 size={18} />}
              />
              <div className="p-5 grid grid-cols-2 md:grid-cols-4 lg:grid-cols-7 gap-3">
                <MetricBadge label="ROC-AUC" value={bestMetrics.rocAuc} color="teal" />
                <MetricBadge label="PR-AUC" value={bestMetrics.prAuc} color="amber" />
                <MetricBadge label="MAE" value={bestMetrics.mae} color="blue" />
                <MetricBadge label="Accuracy" value={bestMetrics.accuracy} color="emerald" />
                <MetricBadge label="F1" value={bestMetrics.f1} color="teal" />
                <MetricBadge label="CV Mean" value={bestMetrics.cvMean} color="emerald" />
                <MetricBadge label="CV Std" value={bestMetrics.cvStd} color="rose" />
              </div>
              <div className="px-5 pb-5 mt-3">
                <div className="flex items-center gap-2 text-sm">
                  {bestMetrics.beatsBaseline ? (
                    <>
                      <CheckCircle2 size={16} className="text-emerald-400" />
                      <span className="text-emerald-300">Beats naive baseline (ROC-AUC: {trainResults.naive.metrics.rocAuc.toFixed(4)})</span>
                    </>
                  ) : (
                    <>
                      <XCircle size={16} className="text-rose-400" />
                      <span className="text-rose-300">Does not beat naive baseline</span>
                    </>
                  )}
                </div>
              </div>
            </Card>
          )}

          {/* Navigation cards */}
          <Card className="p-5 cursor-pointer hover:border-teal-500/30 transition-colors" >
            <div onClick={() => onNavigate('counterfactual')}>
              <div className="flex items-center justify-between mb-2">
                <h3 className="text-sm font-semibold text-ink-100">Why This Customer?</h3>
                <ArrowRight size={16} className="text-ink-500" />
              </div>
              <p className="text-xs text-ink-400">Explore counterfactual explanations — see what would change a customer's churn risk.</p>
            </div>
          </Card>
          <Card className="p-5 cursor-pointer hover:border-teal-500/30 transition-colors">
            <div onClick={() => onNavigate('ablation')}>
              <div className="flex items-center justify-between mb-2">
                <h3 className="text-sm font-semibold text-ink-100">Ablation Report</h3>
                <ArrowRight size={16} className="text-ink-500" />
              </div>
              <p className="text-xs text-ink-400">See what happens to model performance when each feature group is removed.</p>
            </div>
          </Card>
        </div>
      )}

      {isTraining && (
        <div className="fixed bottom-4 right-4 bg-ink-800 border border-teal-500/30 rounded-xl p-4 shadow-xl flex items-center gap-3">
          <div className="w-5 h-5 border-2 border-ink-600 border-t-teal-500 rounded-full animate-spin" />
          <span className="text-sm text-ink-200">Training models...</span>
        </div>
      )}
    </div>
  );
}
