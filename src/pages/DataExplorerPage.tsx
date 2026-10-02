import { useState, useMemo } from 'react';
import { useApp } from '@/context/AppContext';
import { Card, CardHeader, PageHeader, Badge } from '@/components/ui';
import { Database, Search } from 'lucide-react';
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  PieChart, Pie, Cell, Legend,
} from 'recharts';

export function DataExplorerPage() {
  const { customers, featureMeta } = useApp();
  const [search, setSearch] = useState('');
  const [selectedFeature, setSelectedFeature] = useState(featureMeta[0]?.name || 'tenure');

  const churned = customers.filter((c) => c.Churn === 'Yes').length;
  const retained = customers.length - churned;

  const filteredCustomers = useMemo(() => {
    const q = search.toLowerCase();
    return customers.filter((c) =>
      c.customerID.toLowerCase().includes(q) ||
      String(c.Contract).toLowerCase().includes(q) ||
      String(c.InternetService).toLowerCase().includes(q)
    ).slice(0, 50);
  }, [customers, search]);

  // Feature distribution
  const selectedMeta = featureMeta.find((f) => f.name === selectedFeature)!;
  const distributionData = useMemo(() => {
    if (selectedMeta.type === 'categorical' && selectedMeta.categories) {
      return selectedMeta.categories.map((cat) => {
        const count = customers.filter((c) => String(c[selectedMeta.name as keyof typeof c]) === cat).length;
        const churnCount = customers.filter((c) => String(c[selectedMeta.name as keyof typeof c]) === cat && c.Churn === 'Yes').length;
        return { name: cat, count, churnRate: count > 0 ? (churnCount / count) * 100 : 0 };
      });
    } else {
      // Numeric: histogram
      const values = customers.map((c) => Number(c[selectedMeta.name as keyof typeof c]));
      const min = Math.min(...values);
      const max = Math.max(...values);
      const nBins = 10;
      const binSize = (max - min) / nBins;
      return Array.from({ length: nBins }, (_, i) => {
        const start = min + i * binSize;
        const end = start + binSize;
        const inBin = customers.filter((c) => {
          const v = Number(c[selectedMeta.name as keyof typeof c]);
          return v >= start && (v < end || (i === nBins - 1 && v <= end));
        });
        const churnCount = inBin.filter((c) => c.Churn === 'Yes').length;
        return { name: `${Math.round(start)}-${Math.round(end)}`, count: inBin.length, churnRate: inBin.length > 0 ? (churnCount / inBin.length) * 100 : 0 };
      });
    }
  }, [customers, selectedMeta]);

  // Correlation with churn for each feature
  const correlations = useMemo(() => {
    return featureMeta.map((f) => {
      const values = customers.map((c) => {
        if (f.type === 'categorical' && f.categories) {
          // Use first category as binary proxy
          return String(c[f.name as keyof typeof c]) === f.categories[0] ? 1 : 0;
        }
        return Number(c[f.name as keyof typeof c]);
      });
      const churnBinary: number[] = customers.map((c) => (c.Churn === 'Yes' ? 1 : 0));
      const meanV = values.reduce((a, b) => a + b, 0) / values.length;
      const meanC = churnBinary.reduce((a, b) => a + b, 0) / churnBinary.length;
      let num = 0, denV = 0, denC = 0;
      for (let i = 0; i < values.length; i++) {
        num += (values[i] - meanV) * (churnBinary[i] - meanC);
        denV += (values[i] - meanV) ** 2;
        denC += (churnBinary[i] - meanC) ** 2;
      }
      const corr = num / (Math.sqrt(denV * denC) || 1);
      return { feature: f.label, correlation: corr, group: f.group };
    }).sort((a, b) => Math.abs(b.correlation) - Math.abs(a.correlation));
  }, [customers, featureMeta]);

  const pieData = [
    { name: 'Retained', value: retained, color: '#14b8a6' },
    { name: 'Churned', value: churned, color: '#f43f5e' },
  ];

  return (
    <div className="animate-fade-in">
      <PageHeader
        title="Data Explorer"
        subtitle="Explore the telecom customer churn dataset — distributions, churn rates, and correlations"
        icon={<Database size={28} />}
      />

      {/* Class balance */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 mb-6">
        <Card>
          <CardHeader title="Class Balance" subtitle="Overall churn distribution" />
          <div className="p-5">
            <ResponsiveContainer width="100%" height={200}>
              <PieChart>
                <Pie data={pieData} dataKey="value" nameKey="name" cx="50%" cy="50%" innerRadius={45} outerRadius={75} paddingAngle={3}>
                  {pieData.map((e) => <Cell key={e.name} fill={e.color} />)}
                </Pie>
                <Tooltip contentStyle={{ background: '#0f172a', border: '1px solid #334155', borderRadius: '8px' }} />
                <Legend wrapperStyle={{ fontSize: '12px' }} />
              </PieChart>
            </ResponsiveContainer>
          </div>
        </Card>

        {/* Feature correlation with churn */}
        <Card className="lg:col-span-2">
          <CardHeader title="Feature Correlation with Churn" subtitle="Pearson correlation — sorted by absolute value" />
          <div className="p-5">
            <ResponsiveContainer width="100%" height={200}>
              <BarChart data={correlations.slice(0, 10)} layout="vertical" margin={{ left: 80 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                <XAxis type="number" domain={[-0.5, 0.5]} tick={{ fill: '#94a3b8', fontSize: 10 }} />
                <YAxis type="category" dataKey="feature" tick={{ fill: '#94a3b8', fontSize: 10 }} width={80} />
                <Tooltip contentStyle={{ background: '#0f172a', border: '1px solid #334155', borderRadius: '8px' }} />
                <Bar dataKey="correlation" radius={[0, 4, 4, 0]}>
                  {correlations.slice(0, 10).map((d, i) => (
                    <Cell key={i} fill={d.correlation > 0 ? '#f43f5e' : '#14b8a6'} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        </Card>
      </div>

      {/* Feature distribution */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-6">
        <Card>
          <CardHeader title="Feature Distribution" subtitle="Select a feature to explore" />
          <div className="p-5">
            <div className="flex flex-wrap gap-2 mb-4">
              {featureMeta.map((f) => (
                <button
                  key={f.name}
                  onClick={() => setSelectedFeature(f.name)}
                  className={`px-2.5 py-1 rounded-md text-xs transition-colors ${
                    selectedFeature === f.name
                      ? 'bg-teal-500/20 text-teal-300 border border-teal-500/30'
                      : 'bg-ink-700 text-ink-400 hover:text-ink-200 border border-transparent'
                  }`}
                >
                  {f.label}
                </button>
              ))}
            </div>
            <ResponsiveContainer width="100%" height={220}>
              <BarChart data={distributionData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                <XAxis dataKey="name" tick={{ fill: '#94a3b8', fontSize: 10 }} angle={-20} textAnchor="end" height={60} interval={0} />
                <YAxis tick={{ fill: '#94a3b8', fontSize: 11 }} />
                <Tooltip contentStyle={{ background: '#0f172a', border: '1px solid #334155', borderRadius: '8px' }} />
                <Legend wrapperStyle={{ fontSize: '12px' }} />
                <Bar dataKey="count" name="Count" fill="#475569" radius={[4, 4, 0, 0]} />
                <Bar dataKey="churnRate" name="Churn Rate %" fill="#f43f5e" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </Card>

        {/* Raw data table */}
        <Card>
          <CardHeader title="Customer Records" subtitle={`${customers.length.toLocaleString()} total records`} />
          <div className="p-5">
            <div className="relative mb-3">
              <Search size={14} className="absolute left-3 top-1/2 -translate-y-1/2 text-ink-500" />
              <input
                type="text"
                placeholder="Search by ID, contract, or service..."
                value={search}
                onChange={(e) => setSearch(e.target.value)}
                className="w-full bg-ink-800 border border-ink-600 rounded-lg pl-9 pr-3 py-2 text-sm text-ink-100 focus:border-teal-500 focus:outline-none"
              />
            </div>
            <div className="overflow-auto max-h-[260px]">
              <table className="w-full text-xs">
                <thead className="sticky top-0 bg-ink-850">
                  <tr className="text-ink-400 text-left">
                    <th className="px-2 py-2">ID</th>
                    <th className="px-2 py-2">Tenure</th>
                    <th className="px-2 py-2">Contract</th>
                    <th className="px-2 py-2">Monthly</th>
                    <th className="px-2 py-2">Churn</th>
                  </tr>
                </thead>
                <tbody>
                  {filteredCustomers.map((c) => (
                    <tr key={c.customerID} className="border-t border-ink-700/50 hover:bg-ink-800/50">
                      <td className="px-2 py-1.5 text-ink-300 font-mono">{c.customerID}</td>
                      <td className="px-2 py-1.5 text-ink-300">{c.tenure}mo</td>
                      <td className="px-2 py-1.5 text-ink-300">{c.Contract}</td>
                      <td className="px-2 py-1.5 text-ink-300">${c.MonthlyCharges}</td>
                      <td className="px-2 py-1.5">
                        <Badge color={c.Churn === 'Yes' ? 'rose' : 'emerald'}>{c.Churn}</Badge>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </Card>
      </div>

      {/* Feature metadata table */}
      <Card>
        <CardHeader title="Feature Metadata" subtitle="All features used in model training" />
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead className="bg-ink-800/50 text-ink-400 text-left">
              <tr>
                <th className="px-4 py-3">Feature</th>
                <th className="px-4 py-3">Type</th>
                <th className="px-4 py-3">Group</th>
                <th className="px-4 py-3">Categories / Default</th>
              </tr>
            </thead>
            <tbody>
              {featureMeta.map((f) => (
                <tr key={f.name} className="border-t border-ink-700/50 hover:bg-ink-800/30">
                  <td className="px-4 py-2.5 text-ink-200">{f.label}</td>
                  <td className="px-4 py-2.5"><Badge color={f.type === 'numeric' ? 'blue' : 'teal'}>{f.type}</Badge></td>
                  <td className="px-4 py-2.5 text-ink-400">{f.group}</td>
                  <td className="px-4 py-2.5 text-ink-400 text-xs">
                    {f.categories ? f.categories.join(', ') : String(f.defaultValue)}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>
    </div>
  );
}
