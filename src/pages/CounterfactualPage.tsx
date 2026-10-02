import { useState, useMemo, useCallback } from 'react';
import { useApp } from '@/context/AppContext';
import { Card, CardHeader, PageHeader, EmptyState, Button, Badge, Select, Slider } from '@/components/ui';
import { HelpCircle, ArrowRight, Search, TrendingDown, TrendingUp, Sparkles, Lightbulb } from 'lucide-react';
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell,
  ReferenceLine,
} from 'recharts';
import type { PageId } from '@/components/Sidebar';
import type { ModelType, Customer } from '@/ml/types';
import { computeCounterfactuals, findBestCounterfactual } from '@/ml/explanations';
import { encodeCustomer } from '@/ml/preprocessing';

const modelOptions: { value: ModelType; label: string }[] = [
  { value: 'logistic', label: 'Logistic Regression' },
  { value: 'randomForest', label: 'Random Forest' },
  { value: 'gradientBoosting', label: 'Gradient Boosting' },
];

export function CounterfactualPage({ onNavigate }: { onNavigate: (page: PageId) => void }) {
  const { trainResults, customers, featureMeta } = useApp();
  const [selectedModel, setSelectedModel] = useState<ModelType>('gradientBoosting');
  const [customerIdx, setCustomerIdx] = useState(() => {
    // Default to first churned customer
    const idx = customers.findIndex((c) => c.Churn === 'Yes');
    return idx >= 0 ? idx : 0;
  });
  const [search, setSearch] = useState('');
  const [modifiedValues, setModifiedValues] = useState<Record<string, string | number>>({});

  const trainResult = trainResults?.[selectedModel];

  const filteredIndices = useMemo(() => {
    if (!search) return customers.map((_, i) => i).slice(0, 200);
    const q = search.toLowerCase();
    return customers
      .map((c, i) => ({ c, i }))
      .filter(({ c }) =>
        c.customerID.toLowerCase().includes(q) ||
        String(c.Contract).toLowerCase().includes(q) ||
        c.Churn === q.toUpperCase() ||
        (q === 'churn' && c.Churn === 'Yes') ||
        (q === 'retain' && c.Churn === 'No')
      )
      .map(({ i }) => i)
      .slice(0, 200);
  }, [customers, search]);

  const originalCustomer = customers[customerIdx];

  // Build modified customer
  const modifiedCustomer = useMemo(() => {
    if (!originalCustomer) return null;
    const modified = { ...originalCustomer };
    for (const [key, value] of Object.entries(modifiedValues)) {
      (modified as Record<string, unknown>)[key] = value;
    }
    return modified;
  }, [originalCustomer, modifiedValues]);

  // Predictions
  const predictions = useMemo(() => {
    if (!trainResult || !modifiedCustomer) return null;
    const origFeatures = encodeCustomer(originalCustomer, trainResult.preprocessed.encoders, trainResult.preprocessed.featureMeta);
    const modFeatures = encodeCustomer(modifiedCustomer, trainResult.preprocessed.encoders, trainResult.preprocessed.featureMeta);
    const originalRisk = trainResult.model.predictProba(origFeatures);
    const modifiedRisk = trainResult.model.predictProba(modFeatures);
    return { originalRisk, modifiedRisk, delta: modifiedRisk - originalRisk };
  }, [trainResult, originalCustomer, modifiedCustomer]);

  // Counterfactuals
  const counterfactuals = useMemo(() => {
    if (!trainResult || !originalCustomer) return null;
    return computeCounterfactuals(originalCustomer, trainResult.model, trainResult.preprocessed, trainResult.preprocessed.featureMeta);
  }, [trainResult, originalCustomer]);

  const bestCF = useMemo(() => {
    if (!trainResult || !originalCustomer) return null;
    return findBestCounterfactual(originalCustomer, trainResult.model, trainResult.preprocessed, trainResult.preprocessed.featureMeta);
  }, [trainResult, originalCustomer]);

  // Top counterfactuals for chart (most risk reduction)
  const cfChartData = useMemo(() => {
    if (!counterfactuals) return [];
    return counterfactuals.results
      .filter((r) => r.riskDelta < -0.001)
      .slice(0, 10)
      .map((r) => ({
        feature: `${r.label}: ${r.originalValue} -> ${r.newValue}`,
        shortLabel: r.label.length > 15 ? r.label.slice(0, 13) + '...' : r.label,
        riskDelta: r.riskDelta,
        newRisk: r.newRisk,
      }));
  }, [counterfactuals]);

  const handleModifyFeature = useCallback((featureName: string, value: string | number) => {
    setModifiedValues((prev) => {
      const updated = { ...prev };
      const originalVal = originalCustomer[featureName as keyof Customer];
      if (String(value) === String(originalVal)) {
        delete updated[featureName];
      } else {
        updated[featureName] = value;
      }
      return updated;
    });
  }, [originalCustomer]);

  const handleReset = useCallback(() => {
    setModifiedValues({});
  }, []);

  if (!trainResults) {
    return (
      <div className="animate-fade-in">
        <PageHeader title="Why This Customer?" subtitle="Counterfactual analysis — what would change their churn risk?" icon={<HelpCircle size={28} />} />
        <Card className="p-8">
          <EmptyState
            icon={<HelpCircle size={48} />}
            title="No models trained yet"
            subtitle="Train models first to explore counterfactual explanations."
            action={<Button onClick={() => onNavigate('training')}>Go to Training <ArrowRight size={16} className="inline ml-2" /></Button>}
          />
        </Card>
      </div>
    );
  }

  return (
    <div className="animate-fade-in">
      <PageHeader
        title="Why This Customer?"
        subtitle="The signature feature — counterfactual analysis showing what would change a customer's churn risk"
        icon={<HelpCircle size={28} />}
      />

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Left column: customer search and selection */}
        <div className="space-y-4">
          <Card>
            <CardHeader title="Model" icon={<Sparkles size={18} />} />
            <div className="p-5">
              <Select label="Model" value={selectedModel} options={modelOptions} onChange={(v) => setSelectedModel(v as ModelType)} />
            </div>
          </Card>

          <Card>
            <CardHeader title="Find Customer" subtitle="Search for a customer to analyze" icon={<Search size={18} />} />
            <div className="p-5">
              <div className="relative mb-3">
                <Search size={14} className="absolute left-3 top-1/2 -translate-y-1/2 text-ink-500" />
                <input
                  type="text"
                  placeholder="Search by ID, contract, 'churn'..."
                  value={search}
                  onChange={(e) => setSearch(e.target.value)}
                  className="w-full bg-ink-800 border border-ink-600 rounded-lg pl-9 pr-3 py-2 text-sm text-ink-100 focus:border-teal-500 focus:outline-none"
                />
              </div>
              <div className="max-h-[260px] overflow-auto space-y-1">
                {filteredIndices.slice(0, 80).map((idx) => {
                  const c = customers[idx];
                  return (
                    <button
                      key={idx}
                      onClick={() => { setCustomerIdx(idx); setModifiedValues({}); }}
                      className={`w-full text-left px-3 py-2 rounded-lg transition-colors ${
                        idx === customerIdx ? 'bg-teal-500/10 border border-teal-500/20' : 'hover:bg-ink-800 border border-transparent'
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
        </div>

        {/* Center column: counterfactual explorer */}
        <div className="lg:col-span-2 space-y-4">
          {/* Risk comparison */}
          {predictions && (
            <Card>
              <CardHeader
                title="Churn Risk: Original vs. Modified"
                subtitle={Object.keys(modifiedValues).length > 0 ? `${Object.keys(modifiedValues).length} feature(s) modified` : 'No modifications yet — use the sliders below'}
              />
              <div className="p-5">
                <div className="grid grid-cols-3 gap-4 mb-4">
                  <div className="bg-ink-800/50 rounded-lg p-4 text-center">
                    <div className="text-xs text-ink-400 mb-1">Original Risk</div>
                    <div className={`text-2xl font-bold ${predictions.originalRisk >= 0.5 ? 'text-rose-400' : 'text-teal-400'}`}>
                      {(predictions.originalRisk * 100).toFixed(1)}%
                    </div>
                  </div>
                  <div className="bg-ink-800/50 rounded-lg p-4 text-center">
                    <div className="text-xs text-ink-400 mb-1">Modified Risk</div>
                    <div className={`text-2xl font-bold ${predictions.modifiedRisk >= 0.5 ? 'text-rose-400' : 'text-teal-400'}`}>
                      {(predictions.modifiedRisk * 100).toFixed(1)}%
                    </div>
                  </div>
                  <div className="bg-ink-800/50 rounded-lg p-4 text-center">
                    <div className="text-xs text-ink-400 mb-1">Change</div>
                    <div className={`text-2xl font-bold flex items-center justify-center gap-1 ${
                      predictions.delta < 0 ? 'text-emerald-400' : predictions.delta > 0 ? 'text-rose-400' : 'text-ink-300'
                    }`}>
                      {predictions.delta < 0 ? <TrendingDown size={20} /> : predictions.delta > 0 ? <TrendingUp size={20} /> : null}
                      {predictions.delta > 0 ? '+' : ''}{(predictions.delta * 100).toFixed(1)}%
                    </div>
                  </div>
                </div>

                {/* Risk bars */}
                <div className="space-y-2">
                  <div className="relative h-6 bg-ink-800 rounded-full overflow-hidden">
                    <div
                      className={`h-full rounded-full transition-all duration-500 ${predictions.originalRisk >= 0.5 ? 'bg-rose-500/70' : 'bg-teal-500/70'}`}
                      style={{ width: `${predictions.originalRisk * 100}%` }}
                    />
                    <span className="absolute inset-0 flex items-center px-3 text-xs text-ink-100">Original</span>
                  </div>
                  <div className="relative h-6 bg-ink-800 rounded-full overflow-hidden">
                    <div
                      className={`h-full rounded-full transition-all duration-500 ${predictions.modifiedRisk >= 0.5 ? 'bg-rose-500' : 'bg-teal-500'}`}
                      style={{ width: `${predictions.modifiedRisk * 100}%` }}
                    />
                    <span className="absolute inset-0 flex items-center px-3 text-xs text-ink-100">Modified</span>
                  </div>
                </div>

                {Object.keys(modifiedValues).length > 0 && (
                  <div className="mt-4 flex items-center justify-between">
                    <div className="text-xs text-ink-400">
                      Modified: {Object.entries(modifiedValues).map(([k, v]) => {
                        const meta = featureMeta.find((f) => f.name === k);
                        return `${meta?.label || k} = ${v}`;
                      }).join(', ')}
                    </div>
                    <Button variant="ghost" size="sm" onClick={handleReset}>Reset</Button>
                  </div>
                )}
              </div>
            </Card>
          )}

          {/* Best counterfactual callout */}
          {bestCF && bestCF.riskDelta < 0 && (
            <Card className="border-emerald-500/20 bg-emerald-500/5">
              <div className="p-5">
                <div className="flex items-start gap-3">
                  <div className="w-10 h-10 bg-emerald-500/10 rounded-lg flex items-center justify-center flex-shrink-0">
                    <Lightbulb size={20} className="text-emerald-400" />
                  </div>
                  <div>
                    <h3 className="text-sm font-semibold text-emerald-300 mb-1">Best Single Intervention</h3>
                    <p className="text-sm text-ink-200">
                      If <span className="font-semibold text-emerald-300">{bestCF.label}</span> changed from{' '}
                      <span className="font-mono text-ink-300">{String(bestCF.originalValue)}</span> to{' '}
                      <span className="font-mono text-emerald-300">{String(bestCF.newValue)}</span>,
                      churn risk drops by{' '}
                      <span className="font-bold text-emerald-300">{Math.abs(bestCF.riskDelta * 100).toFixed(1)} percentage points</span>{' '}
                      (from {(bestCF.originalRisk * 100).toFixed(1)}% to {(bestCF.newRisk * 100).toFixed(1)}%).
                    </p>
                  </div>
                </div>
              </div>
            </Card>
          )}

          {/* Feature modification controls */}
          <Card>
            <CardHeader
              title="Modify Features"
              subtitle="Adjust features to see how churn risk changes in real time"
              icon={<HelpCircle size={18} />}
            />
            <div className="p-5 grid grid-cols-1 md:grid-cols-2 gap-x-6 gap-y-4">
              {featureMeta.map((feat) => {
                const currentVal = modifiedValues[feat.name] ?? originalCustomer[feat.name as keyof Customer];
                const isModified = feat.name in modifiedValues;
                return (
                  <div key={feat.name} className={`p-3 rounded-lg border ${isModified ? 'border-teal-500/30 bg-teal-500/5' : 'border-ink-700'}`}>
                    {feat.type === 'categorical' && feat.categories ? (
                      <div>
                        <div className="flex items-center justify-between mb-2">
                          <label className="text-xs text-ink-300">{feat.label}</label>
                          {isModified && <Badge color="teal">modified</Badge>}
                        </div>
                        <div className="flex flex-wrap gap-1">
                          {feat.categories.map((cat) => (
                            <button
                              key={cat}
                              onClick={() => handleModifyFeature(feat.name, cat)}
                              className={`px-2 py-1 rounded-md text-xs transition-colors ${
                                String(currentVal) === cat
                                  ? 'bg-teal-500/20 text-teal-300 border border-teal-500/30'
                                  : 'bg-ink-700 text-ink-400 hover:text-ink-200 border border-transparent'
                              }`}
                            >
                              {cat}
                            </button>
                          ))}
                        </div>
                      </div>
                    ) : (
                      <div>
                        <div className="flex items-center justify-between mb-2">
                          <label className="text-xs text-ink-300">{feat.label}</label>
                          {isModified && <Badge color="teal">modified</Badge>}
                        </div>
                        {feat.name === 'tenure' && (
                          <Slider
                            label=""
                            value={Number(currentVal)}
                            min={0}
                            max={72}
                            step={1}
                            unit="mo"
                            onChange={(v) => handleModifyFeature(feat.name, v)}
                          />
                        )}
                        {feat.name === 'MonthlyCharges' && (
                          <Slider
                            label=""
                            value={Number(currentVal)}
                            min={18}
                            max={120}
                            step={1}
                            unit="$"
                            onChange={(v) => handleModifyFeature(feat.name, v)}
                          />
                        )}
                        {feat.name === 'TotalCharges' && (
                          <Slider
                            label=""
                            value={Number(currentVal)}
                            min={0}
                            max={8680}
                            step={100}
                            unit="$"
                            onChange={(v) => handleModifyFeature(feat.name, v)}
                          />
                        )}
                      </div>
                    )}
                  </div>
                );
              })}
            </div>
          </Card>

          {/* Top counterfactual interventions chart */}
          {cfChartData.length > 0 && (
            <Card>
              <CardHeader
                title="Top Risk-Reducing Interventions"
                subtitle="Single changes that would most reduce this customer's churn risk"
                icon={<TrendingDown size={18} />}
              />
              <div className="p-5">
                <ResponsiveContainer width="100%" height={Math.max(200, cfChartData.length * 35)}>
                  <BarChart data={cfChartData} layout="vertical" margin={{ left: 120 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                    <XAxis type="number" tick={{ fill: '#94a3b8', fontSize: 10 }} label={{ value: 'Risk Reduction', position: 'insideBottom', offset: -5, fill: '#64748b', fontSize: 10 }} />
                    <YAxis type="category" dataKey="feature" tick={{ fill: '#94a3b8', fontSize: 9 }} width={120} />
                    <Tooltip
                      contentStyle={{ background: '#0f172a', border: '1px solid #334155', borderRadius: '8px' }}
                      formatter={(value: number) => [`${(value * 100).toFixed(1)}%`, 'Risk Change']}
                    />
                    <ReferenceLine x={0} stroke="#475569" />
                    <Bar dataKey="riskDelta" radius={[0, 4, 4, 0]}>
                      {cfChartData.map((_, i) => (
                        <Cell key={i} fill="#10b981" />
                      ))}
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </Card>
          )}

          {/* All counterfactuals table */}
          {counterfactuals && (
            <Card>
              <CardHeader title="All Counterfactual Scenarios" subtitle="Every possible single-feature change ranked by risk reduction" />
              <div className="overflow-auto max-h-[400px]">
                <table className="w-full text-sm">
                  <thead className="bg-ink-800/50 text-ink-400 text-left sticky top-0">
                    <tr>
                      <th className="px-4 py-2">Feature</th>
                      <th className="px-4 py-2">From</th>
                      <th className="px-4 py-2">To</th>
                      <th className="px-4 py-2">New Risk</th>
                      <th className="px-4 py-2">Change</th>
                    </tr>
                  </thead>
                  <tbody>
                    {counterfactuals.results.slice(0, 50).map((r, i) => (
                      <tr key={i} className="border-t border-ink-700/50 hover:bg-ink-800/30">
                        <td className="px-4 py-2 text-ink-200 text-xs">{r.label}</td>
                        <td className="px-4 py-2 text-ink-400 font-mono text-xs">{String(r.originalValue)}</td>
                        <td className="px-4 py-2 text-ink-300 font-mono text-xs">{String(r.newValue)}</td>
                        <td className="px-4 py-2 font-mono text-xs">
                          <span className={r.newRisk >= 0.5 ? 'text-rose-400' : 'text-teal-400'}>
                            {(r.newRisk * 100).toFixed(1)}%
                          </span>
                        </td>
                        <td className="px-4 py-2 font-mono text-xs">
                          <span className={r.riskDelta < 0 ? 'text-emerald-400' : r.riskDelta > 0 ? 'text-rose-400' : 'text-ink-400'}>
                            {r.riskDelta > 0 ? '+' : ''}{(r.riskDelta * 100).toFixed(1)}%
                          </span>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </Card>
          )}
        </div>
      </div>
    </div>
  );
}
