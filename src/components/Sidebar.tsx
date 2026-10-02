import { type ReactNode } from 'react';
import {
  LayoutDashboard, Database, Cpu, BarChart3, Gauge,
  Sparkles, HelpCircle, ScrollText, FlaskConical, Activity,
} from 'lucide-react';

export type PageId =
  | 'dashboard'
  | 'data'
  | 'training'
  | 'comparison'
  | 'calibration'
  | 'explanations'
  | 'counterfactual'
  | 'experiments'
  | 'ablation';

interface NavItem {
  id: PageId;
  label: string;
  icon: ReactNode;
  description: string;
}

const navItems: NavItem[] = [
  { id: 'dashboard', label: 'Dashboard', icon: <LayoutDashboard size={18} />, description: 'Overview & status' },
  { id: 'data', label: 'Data Explorer', icon: <Database size={18} />, description: 'Dataset & distributions' },
  { id: 'training', label: 'Train Models', icon: <Cpu size={18} />, description: 'Train & cross-validate' },
  { id: 'comparison', label: 'Model Comparison', icon: <BarChart3 size={18} />, description: 'Metrics & ROC/PR curves' },
  { id: 'calibration', label: 'Calibration', icon: <Gauge size={18} />, description: 'Probability reliability' },
  { id: 'explanations', label: 'SHAP Explanations', icon: <Sparkles size={18} />, description: 'Feature contributions' },
  { id: 'counterfactual', label: 'Why This Customer?', icon: <HelpCircle size={18} />, description: 'Counterfactual analysis' },
  { id: 'experiments', label: 'Experiment Log', icon: <ScrollText size={18} />, description: 'Training history' },
  { id: 'ablation', label: 'Ablation Report', icon: <FlaskConical size={18} />, description: 'Feature group impact' },
];

export function Sidebar({
  currentPage,
  onNavigate,
  experimentCount,
  bestModelName,
  datasetSize,
}: {
  currentPage: PageId;
  onNavigate: (page: PageId) => void;
  experimentCount: number;
  bestModelName: string | null;
  datasetSize: number;
}) {
  return (
    <div className="w-64 bg-ink-900 border-r border-ink-700 flex flex-col h-screen sticky top-0">
      {/* Logo */}
      <div className="px-5 py-5 border-b border-ink-700">
        <div className="flex items-center gap-2.5">
          <div className="w-9 h-9 bg-teal-500 rounded-lg flex items-center justify-center">
            <Activity size={20} className="text-ink-950" />
          </div>
          <div>
            <h1 className="text-sm font-bold text-ink-100">Churn Lab</h1>
            <p className="text-xs text-ink-400">ML Modeling Studio</p>
          </div>
        </div>
      </div>

      {/* Navigation */}
      <nav className="flex-1 overflow-y-auto px-3 py-4 space-y-0.5">
        {navItems.map((item) => {
          const active = currentPage === item.id;
          return (
            <button
              key={item.id}
              onClick={() => onNavigate(item.id)}
              className={`w-full flex items-center gap-3 px-3 py-2.5 rounded-lg transition-all duration-200 group ${
                active
                  ? 'bg-teal-500/10 text-teal-300 border border-teal-500/20'
                  : 'text-ink-400 hover:text-ink-200 hover:bg-ink-800 border border-transparent'
              }`}
            >
              <span className={active ? 'text-teal-400' : 'text-ink-500 group-hover:text-ink-300'}>
                {item.icon}
              </span>
              <div className="text-left flex-1">
                <div className="text-sm font-medium">{item.label}</div>
                <div className={`text-xs ${active ? 'text-teal-400/70' : 'text-ink-500'}`}>{item.description}</div>
              </div>
            </button>
          );
        })}
      </nav>

      {/* Status bar */}
      <div className="px-4 py-4 border-t border-ink-700 space-y-2">
        <div className="flex items-center justify-between text-xs">
          <span className="text-ink-500">Dataset</span>
          <span className="text-ink-300 font-mono">{datasetSize.toLocaleString()} rows</span>
        </div>
        <div className="flex items-center justify-between text-xs">
          <span className="text-ink-500">Best Model</span>
          <span className="text-teal-300 font-medium">{bestModelName || 'Not trained'}</span>
        </div>
        <div className="flex items-center justify-between text-xs">
          <span className="text-ink-500">Experiments</span>
          <span className="text-ink-300 font-mono">{experimentCount}</span>
        </div>
      </div>
    </div>
  );
}
