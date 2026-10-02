import { useState, useMemo } from 'react';
import { useApp } from '@/context/AppContext';
import { Card, CardHeader, PageHeader, EmptyState, Button, Badge } from '@/components/ui';
import { ScrollText, ArrowRight, Trash2, Download, Search } from 'lucide-react';
import type { PageId } from '@/components/Sidebar';
import type { ExperimentLogEntry } from '@/ml/types';

const modelNames: Record<string, string> = {
  naive: 'Naive Baseline', logistic: 'Logistic Regression', decisionTree: 'Decision Tree',
  randomForest: 'Random Forest', gradientBoosting: 'Gradient Boosting',
};

export function ExperimentLogPage({ onNavigate }: { onNavigate: (page: PageId) => void }) {
  const { experimentLog, clearExperimentLog, trainResults } = useApp();
  const [search, setSearch] = useState('');
  const [sortBy, setSortBy] = useState<'timestamp' | 'rocAuc' | 'prAuc'>('timestamp');

  const filtered = useMemo(() => {
    let entries = [...experimentLog];
    if (search) {
      const q = search.toLowerCase();
      entries = entries.filter((e) =>
        e.modelName.toLowerCase().includes(q) ||
        e.modelType.toLowerCase().includes(q)
      );
    }
    entries.sort((a, b) => {
      if (sortBy === 'timestamp') return b.timestamp.localeCompare(a.timestamp);
      if (sortBy === 'rocAuc') return b.metrics.rocAuc - a.metrics.rocAuc;
      return b.metrics.prAuc - a.metrics.prAuc;
    });
    return entries;
  }, [experimentLog, search, sortBy]);

  const handleExport = () => {
    const blob = new Blob([JSON.stringify(experimentLog, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `experiment_log_${Date.now()}.json`;
    a.click();
    URL.revokeObjectURL(url);
  };

  if (experimentLog.length === 0) {
    return (
      <div className="animate-fade-in">
        <PageHeader title="Experiment Log" subtitle="Training history persisted across sessions" icon={<ScrollText size={28} />} />
        <Card className="p-8">
          <EmptyState
            icon={<ScrollText size={48} />}
            title="No experiments logged yet"
            subtitle="Train models to start building your experiment log. Each training run logs model type, hyperparameters, and all metrics."
            action={trainResults ? undefined : <Button onClick={() => onNavigate('training')}>Go to Training <ArrowRight size={16} className="inline ml-2" /></Button>}
          />
        </Card>
      </div>
    );
  }

  return (
    <div className="animate-fade-in">
      <PageHeader
        title="Experiment Log"
        subtitle={`${experimentLog.length} experiments logged — persisted in browser storage as JSON`}
        icon={<ScrollText size={28} />}
      />

      <div className="flex items-center gap-3 mb-4">
        <div className="relative flex-1">
          <Search size={14} className="absolute left-3 top-1/2 -translate-y-1/2 text-ink-500" />
          <input
            type="text"
            placeholder="Search by model name..."
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            className="w-full bg-ink-800 border border-ink-600 rounded-lg pl-9 pr-3 py-2 text-sm text-ink-100 focus:border-teal-500 focus:outline-none"
          />
        </div>
        <select
          value={sortBy}
          onChange={(e) => setSortBy(e.target.value as 'timestamp' | 'rocAuc' | 'prAuc')}
          className="bg-ink-800 border border-ink-600 rounded-lg px-3 py-2 text-sm text-ink-100 focus:border-teal-500 focus:outline-none"
        >
          <option value="timestamp">Sort by Date</option>
          <option value="rocAuc">Sort by ROC-AUC</option>
          <option value="prAuc">Sort by PR-AUC</option>
        </select>
        <Button variant="secondary" size="md" onClick={handleExport}>
          <Download size={16} className="inline mr-1" /> Export
        </Button>
        <Button variant="ghost" size="md" onClick={clearExperimentLog}>
          <Trash2 size={16} className="inline mr-1" /> Clear
        </Button>
      </div>

      <Card className="overflow-x-auto">
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead className="bg-ink-800/50 text-ink-400 text-left">
              <tr>
                <th className="px-4 py-3">Timestamp</th>
                <th className="px-4 py-3">Model</th>
                <th className="px-4 py-3">ROC-AUC</th>
                <th className="px-4 py-3">PR-AUC</th>
                <th className="px-4 py-3">MAE</th>
                <th className="px-4 py-3">F1</th>
                <th className="px-4 py-3">CV Mean</th>
                <th className="px-4 py-3">Dataset</th>
                <th className="px-4 py-3">Hyperparameters</th>
              </tr>
            </thead>
            <tbody>
              {filtered.map((e: ExperimentLogEntry) => {
                const isBest = e.metrics.rocAuc === Math.max(...experimentLog.map((x) => x.metrics.rocAuc));
                return (
                  <tr key={e.id} className="border-t border-ink-700/50 hover:bg-ink-800/30">
                    <td className="px-4 py-2.5 text-xs text-ink-400">
                      {new Date(e.timestamp).toLocaleString()}
                    </td>
                    <td className="px-4 py-2.5">
                      <div className="flex items-center gap-2">
                        <span className="text-ink-200">{e.modelName}</span>
                        {isBest && <Badge color="emerald">best</Badge>}
                      </div>
                    </td>
                    <td className="px-4 py-2.5 font-mono text-xs">
                      <span className={isBest ? 'text-teal-300 font-semibold' : 'text-ink-300'}>
                        {e.metrics.rocAuc.toFixed(4)}
                      </span>
                    </td>
                    <td className="px-4 py-2.5 font-mono text-xs text-ink-300">{e.metrics.prAuc.toFixed(4)}</td>
                    <td className="px-4 py-2.5 font-mono text-xs text-ink-300">{e.metrics.mae.toFixed(4)}</td>
                    <td className="px-4 py-2.5 font-mono text-xs text-ink-300">{e.metrics.f1.toFixed(4)}</td>
                    <td className="px-4 py-2.5 font-mono text-xs text-ink-300">
                      {e.metrics.cvMean.toFixed(4)} +/- {e.metrics.cvStd.toFixed(4)}
                    </td>
                    <td className="px-4 py-2.5 text-xs text-ink-400">{e.datasetSize.toLocaleString()}</td>
                    <td className="px-4 py-2.5 text-xs text-ink-400">
                      {Object.entries(e.hyperparameters).map(([k, v]) => `${k}=${v}`).join(', ')}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </Card>

      {/* Summary stats */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mt-6">
        <Card className="p-4">
          <div className="text-xs text-ink-400">Total Experiments</div>
          <div className="text-xl font-bold text-ink-100">{experimentLog.length}</div>
        </Card>
        <Card className="p-4">
          <div className="text-xs text-ink-400">Best ROC-AUC</div>
          <div className="text-xl font-bold text-teal-300">
            {Math.max(...experimentLog.map((e) => e.metrics.rocAuc)).toFixed(4)}
          </div>
        </Card>
        <Card className="p-4">
          <div className="text-xs text-ink-400">Models Used</div>
          <div className="text-xl font-bold text-ink-100">
            {new Set(experimentLog.map((e) => e.modelType)).size}
          </div>
        </Card>
        <Card className="p-4">
          <div className="text-xs text-ink-400">Best Model</div>
          <div className="text-xl font-bold text-ink-100">
            {(() => {
              const best = experimentLog.reduce((a, b) => (a.metrics.rocAuc > b.metrics.rocAuc ? a : b));
              return modelNames[best.modelType] || best.modelName;
            })()}
          </div>
        </Card>
      </div>
    </div>
  );
}
