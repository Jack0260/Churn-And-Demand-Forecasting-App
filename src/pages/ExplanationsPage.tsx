import { useState, useMemo } from 'react';
import { useApp } from '@/context/AppContext';
import { Card, CardHeader, PageHeader, EmptyState, Button, Badge, Select } from '@/components/ui';
import { Sparkles, ArrowRight, Search, User } from 'lucide-react';
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell,
  ReferenceLine,
} from 'recharts';
import type { PageId } from '@/components/Sidebar';
import type { ModelType } from '@/ml/types';
import { explainPrediction } from '@/ml/explanations';

const modelOptions: { value: ModelType; label: string }[] = [
  { value: 'logistic', label: 'Logistic Regression' },
  { value: 'randomForest', label: 'Random Forest' },
  { value: 'gradientBoosting', label: 'Gradient Boosting' },
];

export function ExplanationsPage({ onNavigate }: { onNavigate: (page: PageId) => void }) {
  const { trainResults, customers, featureMeta } = useApp();
  const [selectedModel, setSelectedModel] = useState<ModelType>('gradientBoosting');
  const [customerIdx, setCustomerIdx] = useState(0);
  const [search, setSearch] = useState('');

  const trainResult = trainResults?.[selectedModel];

  // Filter customers for search
  const filteredIndices = useMemo(() => {
    if (!search) return customers.map((_, i) => i).slice(0, 200);
    const q = search.toLowerCase();
    return customers
      .map((c, i) => ({ c, i }))
      .filter(({ c }) =>
        c.customerID.toLowerCase().includes(q) ||
        String(c.Contract).toLowerCase().includes(q) ||
        String(c.InternetService).toLowerCase().includes(q)
      )
      .map(({ i }) => i)
      .slice(0, 200);
  }, [customers, search]);

  const explanation = useMemo(() => {
    if (!trainResult) return null;
    const customer = customers[customerIdx];
    return explainPrediction(customer, trainResult.model, trainResult.preprocessed, trainResult.preprocessed.featureMeta);
  }, [trainResult, customers, customerIdx]);

  if (!trainResults) {
    return (
      <div className="animate-fade-in">
        <PageHeader title="SHAP Explanations" subtitle="Per-prediction feature contributions" icon={<Sparkles size={28} />} />
        <Card className="p-8">
          <EmptyState
            icon={<Sparkles size={48} />}
            title="No models trained yet"
            subtitle="Train models first to see SHAP-style explanations."
            action={<Button onClick={() => onNavigate('training')}>Go to Training <ArrowRight size={16} className="inline ml-2" /></Button>}
          />
        </Card>
      </div>
    );
  }

  const customer = customers[customerIdx];
  const chartData = explanation
    ? explanation.explanations.slice(0, 15).map((e) => ({
        feature: e.label.length > 20 ? e.label.slice(0, 18) + '...' : e.label,
        contribution: e.contribution,
        rawValue: e.value,
      }))
    : [];

  return (
    <div className="animate-fade-in">
      <PageHeader
        title="SHAP Explanations"
        subtitle="Understand which features drive each prediction — model-agnostic per-feature contributions"
        icon={<Sparkles size={28} />}
      />

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Customer selector */}
        <div className="space-y-4">
          <Card>
            <CardHeader title="Model Selection" icon={<Sparkles size={18} />} />
            <div className="p-5">
              <Select
                label="Model"
                value={selectedModel}
                options={modelOptions}
                onChange={(v) => setSelectedModel(v as ModelType)}
              />
            </div>
          </Card>

          <Card>
            <CardHeader title="Customer Search" subtitle="Find a customer to explain" icon={<Search size={18} />} />
            <div className="p-5">
              <div className="relative mb-3">
                <Search size={14} className="absolute left-3 top-1/2 -translate-y-1/2 text-ink-500" />
                <input
                  type="text"
                  placeholder="Search by ID, contract..."
                  value={search}
                  onChange={(e) => setSearch(e.target.value)}
                  className="w-full bg-ink-800 border border-ink-600 rounded-lg pl-9 pr-3 py-2 text-sm text-ink-100 focus:border-teal-500 focus:outline-none"
                />
              </div>
              <div className="max-h-[300px] overflow-auto space-y-1">
                {filteredIndices.slice(0, 100).map((idx) => {
                  const c = customers[idx];
                  return (
                    <button
                      key={idx}
                      onClick={() => setCustomerIdx(idx)}
                      className={`w-full text-left px-3 py-2 rounded-lg transition-colors ${
                        idx === customerIdx
                          ? 'bg-teal-500/10 border border-teal-500/20'
                          : 'hover:bg-ink-800 border border-transparent'
                      }`}
                    >
                      <div className="flex items-center justify-between">
                        <span className="text-xs font-mono text-ink-300">{c.customerID}</span>
                        <Badge color={c.Churn === 'Yes' ? 'rose' : 'emerald'}>{c.Churn}</Badge>
                      </div>
                      <div className="text-xs text-ink-500 mt-0.5">
                        {c.tenure}mo | {c.Contract} | ${c.MonthlyCharges}/mo
                      </div>
                    </button>
                  );
                })}
              </div>
            </div>
          </Card>

          {/* Customer profile */}
          <Card>
            <CardHeader title="Customer Profile" icon={<User size={18} />} />
            <div className="p-5">
              <div className="space-y-1.5 text-sm">
                {featureMeta.map((f) => (
                  <div key={f.name} className="flex justify-between">
                    <span className="text-ink-400 text-xs">{f.label}</span>
                    <span className="text-ink-200 text-xs font-mono">
                      {String(customer[f.name as keyof typeof customer])}
                    </span>
                  </div>
                ))}
              </div>
            </div>
          </Card>
        </div>

        {/* Explanation chart */}
        <div className="lg:col-span-2 space-y-4">
          {explanation && (
            <>
              <Card>
                <CardHeader
                  title={`Prediction: ${(explanation.prediction * 100).toFixed(1)}% churn risk`}
                  subtitle={explanation.prediction >= 0.5 ? 'Model predicts: CHURN' : 'Model predicts: RETAIN'}
                />
                <div className="p-5">
                  {/* Risk gauge */}
                  <div className="relative h-8 bg-ink-800 rounded-full overflow-hidden mb-2">
                    <div
                      className={`h-full rounded-full transition-all duration-500 ${
                        explanation.prediction >= 0.5 ? 'bg-rose-500' : 'bg-teal-500'
                      }`}
                      style={{ width: `${explanation.prediction * 100}%` }}
                    />
                    <div className="absolute inset-0 flex items-center justify-center text-xs font-bold text-ink-100">
                      {(explanation.prediction * 100).toFixed(1)}%
                    </div>
                  </div>
                  <div className="flex justify-between text-xs text-ink-500">
                    <span>0% (Retain)</span>
                    <span>50% threshold</span>
                    <span>100% (Churn)</span>
                  </div>
                </div>
              </Card>

              <Card>
                <CardHeader
                  title="SHAP-Style Feature Contributions"
                  subtitle="Red bars increase churn risk, teal bars decrease it"
                  icon={<Sparkles size={18} />}
                />
                <div className="p-5">
                  <ResponsiveContainer width="100%" height={400}>
                    <BarChart data={chartData} layout="vertical" margin={{ left: 80, right: 20 }}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                      <XAxis type="number" tick={{ fill: '#94a3b8', fontSize: 10 }} label={{ value: 'Contribution to prediction', position: 'insideBottom', offset: -5, fill: '#64748b', fontSize: 10 }} />
                      <YAxis type="category" dataKey="feature" tick={{ fill: '#94a3b8', fontSize: 10 }} width={80} />
                      <Tooltip
                        contentStyle={{ background: '#0f172a', border: '1px solid #334155', borderRadius: '8px' }}
                        formatter={(value: number) => [value.toFixed(4), 'Contribution']}
                      />
                      <ReferenceLine x={0} stroke="#475569" />
                      <Bar dataKey="contribution" radius={[0, 4, 4, 0]}>
                        {chartData.map((d, i) => (
                          <Cell key={i} fill={d.contribution > 0 ? '#f43f5e' : '#14b8a6'} />
                        ))}
                      </Bar>
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              </Card>

              {/* Feature contribution table */}
              <Card>
                <CardHeader title="Detailed Breakdown" subtitle="Every feature's contribution to this prediction" />
                <div className="overflow-x-auto max-h-[300px] overflow-auto">
                  <table className="w-full text-sm">
                    <thead className="bg-ink-800/50 text-ink-400 text-left sticky top-0">
                      <tr>
                        <th className="px-4 py-2">Feature</th>
                        <th className="px-4 py-2">Value</th>
                        <th className="px-4 py-2">Contribution</th>
                        <th className="px-4 py-2">Direction</th>
                      </tr>
                    </thead>
                    <tbody>
                      {explanation.explanations.map((e, i) => (
                        <tr key={i} className="border-t border-ink-700/50">
                          <td className="px-4 py-2 text-ink-200 text-xs">{e.label}</td>
                          <td className="px-4 py-2 text-ink-300 font-mono text-xs">{String(e.value)}</td>
                          <td className="px-4 py-2 font-mono text-xs">
                            <span className={e.contribution > 0 ? 'text-rose-400' : 'text-teal-400'}>
                              {e.contribution > 0 ? '+' : ''}{e.contribution.toFixed(4)}
                            </span>
                          </td>
                          <td className="px-4 py-2">
                            <Badge color={e.contribution > 0 ? 'rose' : 'teal'}>
                              {e.contribution > 0 ? 'Increases risk' : 'Decreases risk'}
                            </Badge>
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </Card>
            </>
          )}
        </div>
      </div>
    </div>
  );
}
