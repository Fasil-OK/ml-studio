import { Routes, Route } from 'react-router-dom'
import AppShell from './components/layout/AppShell'
import HomePage from './pages/HomePage'
import ProjectPage from './pages/ProjectPage'
import DatasetPage from './pages/DatasetPage'
import ModelSelectPage from './pages/ModelSelectPage'
import TrainingPage from './pages/TrainingPage'
import EvaluationPage from './pages/EvaluationPage'
import ExplanationPage from './pages/ExplanationPage'
import TestingPage from './pages/TestingPage'
import SettingsPage from './pages/SettingsPage'

export default function App() {
  return (
    <Routes>
      <Route path="/" element={<HomePage />} />
      <Route element={<AppShell />}>
        <Route path="/project/:id" element={<ProjectPage />} />
        <Route path="/project/:id/data" element={<DatasetPage />} />
        <Route path="/project/:id/model" element={<ModelSelectPage />} />
        <Route path="/project/:id/train" element={<TrainingPage />} />
        <Route path="/project/:id/evaluate" element={<EvaluationPage />} />
        <Route path="/project/:id/explain" element={<ExplanationPage />} />
        <Route path="/project/:id/test" element={<TestingPage />} />
        <Route path="/project/:id/settings" element={<SettingsPage />} />
      </Route>
    </Routes>
  )
}
