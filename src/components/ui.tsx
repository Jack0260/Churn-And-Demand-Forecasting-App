import type { ReactNode } from 'react';

export function Card({ children, className = '' }: { children: ReactNode; className?: string }) {
  return (
    <div className={`bg-ink-850 border border-ink-700 rounded-xl ${className}`}>
      {children}
    </div>
  );
}

export function CardHeader({ title, subtitle, icon }: { title: string; subtitle?: string; icon?: ReactNode }) {
  return (
    <div className="px-5 py-4 border-b border-ink-700 flex items-center gap-3">
      {icon && <div className="text-teal-400">{icon}</div>}
      <div>
        <h3 className="text-sm font-semibold text-ink-100">{title}</h3>
        {subtitle && <p className="text-xs text-ink-400 mt-0.5">{subtitle}</p>}
      </div>
    </div>
  );
}

export function MetricBadge({ label, value, color = 'teal' }: { label: string; value: string | number; color?: string }) {
  const colors: Record<string, string> = {
    teal: 'bg-teal-500/10 text-teal-300 border-teal-500/20',
    emerald: 'bg-emerald-500/10 text-emerald-300 border-emerald-500/20',
    amber: 'bg-amber-500/10 text-amber-300 border-amber-500/20',
    rose: 'bg-rose-500/10 text-rose-300 border-rose-500/20',
    blue: 'bg-blue-500/10 text-blue-300 border-blue-500/20',
  };
  return (
    <div className={`px-3 py-1.5 rounded-lg border ${colors[color] || colors.teal}`}>
      <div className="text-xs text-ink-400">{label}</div>
      <div className="text-lg font-semibold">{typeof value === 'number' ? value.toFixed(4) : value}</div>
    </div>
  );
}

export function Badge({ children, color = 'teal' }: { children: ReactNode; color?: string }) {
  const colors: Record<string, string> = {
    teal: 'bg-teal-500/10 text-teal-300 border-teal-500/20',
    emerald: 'bg-emerald-500/10 text-emerald-300 border-emerald-500/20',
    amber: 'bg-amber-500/10 text-amber-300 border-amber-500/20',
    rose: 'bg-rose-500/10 text-rose-300 border-rose-500/20',
    gray: 'bg-ink-700 text-ink-300 border-ink-600',
  };
  return (
    <span className={`inline-flex items-center gap-1 px-2 py-0.5 rounded-md text-xs font-medium border ${colors[color] || colors.teal}`}>
      {children}
    </span>
  );
}

export function Button({
  children,
  onClick,
  variant = 'primary',
  size = 'md',
  disabled,
  className = '',
}: {
  children: ReactNode;
  onClick?: () => void;
  variant?: 'primary' | 'secondary' | 'ghost';
  size?: 'sm' | 'md' | 'lg';
  disabled?: boolean;
  className?: string;
}) {
  const variants: Record<string, string> = {
    primary: 'bg-teal-500 hover:bg-teal-400 text-ink-950 font-semibold disabled:opacity-40 disabled:hover:bg-teal-500',
    secondary: 'bg-ink-700 hover:bg-ink-600 text-ink-100 border border-ink-600 disabled:opacity-40',
    ghost: 'hover:bg-ink-700 text-ink-300 disabled:opacity-40',
  };
  const sizes: Record<string, string> = {
    sm: 'px-3 py-1.5 text-xs',
    md: 'px-4 py-2 text-sm',
    lg: 'px-6 py-3 text-base',
  };
  return (
    <button
      onClick={onClick}
      disabled={disabled}
      className={`rounded-lg transition-all duration-200 ${variants[variant]} ${sizes[size]} ${className}`}
    >
      {children}
    </button>
  );
}

export function Slider({
  label,
  value,
  min,
  max,
  step,
  onChange,
  unit = '',
}: {
  label: string;
  value: number;
  min: number;
  max: number;
  step: number;
  onChange: (v: number) => void;
  unit?: string;
}) {
  return (
    <div className="space-y-1.5">
      <div className="flex justify-between items-center">
        <label className="text-xs text-ink-300">{label}</label>
        <span className="text-xs font-mono text-teal-300">{value}{unit}</span>
      </div>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(e) => onChange(Number(e.target.value))}
        className="w-full h-1.5 bg-ink-700 rounded-full appearance-none cursor-pointer accent-teal-500"
      />
    </div>
  );
}

export function Select({
  label,
  value,
  options,
  onChange,
}: {
  label: string;
  value: string;
  options: { value: string; label: string }[];
  onChange: (v: string) => void;
}) {
  return (
    <div className="space-y-1.5">
      <label className="text-xs text-ink-300">{label}</label>
      <select
        value={value}
        onChange={(e) => onChange(e.target.value)}
        className="w-full bg-ink-800 border border-ink-600 rounded-lg px-3 py-2 text-sm text-ink-100 focus:border-teal-500 focus:outline-none transition-colors"
      >
        {options.map((opt) => (
          <option key={opt.value} value={opt.value}>{opt.label}</option>
        ))}
      </select>
    </div>
  );
}

export function PageHeader({ title, subtitle, icon }: { title: string; subtitle?: string; icon?: ReactNode }) {
  return (
    <div className="mb-6 flex items-center gap-3">
      {icon && <div className="text-teal-400">{icon}</div>}
      <div>
        <h1 className="text-2xl font-bold text-ink-100">{title}</h1>
        {subtitle && <p className="text-sm text-ink-400 mt-1">{subtitle}</p>}
      </div>
    </div>
  );
}

export function EmptyState({ icon, title, subtitle, action }: { icon: ReactNode; title: string; subtitle: string; action?: ReactNode }) {
  return (
    <div className="flex flex-col items-center justify-center py-20 text-center">
      <div className="text-ink-500 mb-4">{icon}</div>
      <h3 className="text-lg font-semibold text-ink-200 mb-2">{title}</h3>
      <p className="text-sm text-ink-400 max-w-md mb-6">{subtitle}</p>
      {action}
    </div>
  );
}

export function ProgressBar({ value, label }: { value: number; label?: string }) {
  return (
    <div className="w-full">
      {label && <div className="text-xs text-ink-400 mb-2">{label}</div>}
      <div className="h-2 bg-ink-700 rounded-full overflow-hidden">
        <div
          className="h-full bg-teal-500 rounded-full transition-all duration-300"
          style={{ width: `${Math.min(100, value * 100)}%` }}
        />
      </div>
    </div>
  );
}

export function Spinner({ size = 'md' }: { size?: 'sm' | 'md' | 'lg' }) {
  const sizes: Record<string, string> = { sm: 'w-4 h-4', md: 'w-6 h-6', lg: 'w-10 h-10' };
  return (
    <div className={`${sizes[size]} border-2 border-ink-600 border-t-teal-500 rounded-full animate-spin`} />
  );
}
