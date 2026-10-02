import { useState } from 'react';
import { AppProvider, useApp } from '@/context/AppContext';
import { Sidebar, type PageId } from '@/components/Sidebar';
import { DashboardPage } from '@/pages/DashboardPage';
import { DataExplorerPage } from '@/pages/DataExplorerPage';
import { TrainingPage } from '@/pages/TrainingPage';
import { ComparisonPage } from '@/pages/ComparisonPage';
import { CalibrationPage } from '@/pages/CalibrationPage';
import { ExplanationsPage } from '@/pages/ExplanationsPage';
import { CounterfactualPage } from '@/pages/CounterfactualPage';
import { ExperimentLogPage } from '@/pages/ExperimentLogPage';
import { AblationReportPage } from '@/pages/AblationReportPage';

const modelNames: Record<string, string> = {
  naive: 'Naive Baseline', logistic: 'Logistic Regression', randomForest: 'Random Forest', gradientBoosting: 'Gradient Boosting',
};

function AppContent() {
  const [currentPage, setCurrentPage] = useState<PageId>('dashboard');
  const { experimentLog, bestModelType, customers } = useApp();

  const renderPage = () => {
    switch (currentPage) {
      case 'dashboard': return <DashboardPage onNavigate={setCurrentPage} />;
      case 'data': return <DataExplorerPage />;
      case 'training': return <TrainingPage onNavigate={setCurrentPage} />;
      case 'comparison': return <ComparisonPage onNavigate={setCurrentPage} />;
      case 'calibration': return <CalibrationPage onNavigate={setCurrentPage} />;
      case 'explanations': return <ExplanationsPage onNavigate={setCurrentPage} />;
      case 'counterfactual': return <CounterfactualPage onNavigate={setCurrentPage} />;
      case 'experiments': return <ExperimentLogPage onNavigate={setCurrentPage} />;
      case 'ablation': return <AblationReportPage onNavigate={setCurrentPage} />;
      default: return <DashboardPage onNavigate={setCurrentPage} />;
    }
  };

  return (
    <div className="flex min-h-screen bg-ink-900">
      <Sidebar
        currentPage={currentPage}
        onNavigate={setCurrentPage}
        experimentCount={experimentLog.length}
        bestModelName={bestModelType ? modelNames[bestModelType] : null}
        datasetSize={customers.length}
      />
      <main className="flex-1 overflow-x-auto p-6 lg:p-8">
        {renderPage()}
      </main>
    </div>
  );
}

export default function App() {
  return (
    <AppProvider>
      <AppContent />
    </AppProvider>
  );
}
