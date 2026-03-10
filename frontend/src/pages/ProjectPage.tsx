import { useEffect, useState } from 'react'
import { useParams, Link } from 'react-router-dom'
import { Upload, Brain, Play, BarChart3, Eye, FlaskConical, ArrowRight } from 'lucide-react'
import { useProjectStore } from '../stores/projectStore'
import { getDataset, type DatasetInfo } from '../api/datasets'
import { listExperiments, type Experiment } from '../api/training'

export default function ProjectPage() {
  const { id } = useParams()
  const project = useProjectStore((s) => s.currentProject)
  const [dataset, setDataset] = useState<DatasetInfo | null>(null)
  const [experiments, setExperiments] = useState<Experiment[]>([])

  useEffect(() => {
    if (id) {
      getDataset(id).then(setDataset)
      listExperiments(id).then(setExperiments)
    }
  }, [id])

  const steps = [
    { label: 'Upload Data', icon: Upload, path: 'data', done: !!dataset, desc: dataset ? `${dataset.total_images} images, ${dataset.num_classes} classes` : 'Upload your image dataset' },
    { label: 'Select Model', icon: Brain, path: 'model', done: experiments.length > 0, desc: experiments.length > 0 ? `${experiments[0].architecture}` : 'Choose model architecture' },
    { label: 'Train', icon: Play, path: 'train', done: experiments.some(e => e.status === 'completed'), desc: 'Train with real-time monitoring' },
    { label: 'Evaluate', icon: BarChart3, path: 'evaluate', done: false, desc: 'Evaluate model performance' },
    { label: 'Explain', icon: Eye, path: 'explain', done: false, desc: 'Causal AI & explainability' },
    { label: 'Test', icon: FlaskConical, path: 'test', done: false, desc: 'Live test your model' },
  ]

  return (
    <div>
      <h2 className="text-2xl font-bold text-gray-900 mb-1">{project?.name || 'Project'}</h2>
      <p className="text-gray-500 mb-8">{project?.description || 'AutoML workflow dashboard'}</p>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
        {steps.map((step) => {
          const Icon = step.icon
          return (
            <Link
              key={step.path}
              to={`/project/${id}/${step.path}`}
              className={`bg-white rounded-xl border p-5 hover:shadow-md transition-all group ${
                step.done ? 'border-green-200' : 'border-gray-200'
              }`}
            >
              <div className="flex items-start justify-between">
                <div className={`p-2 rounded-lg ${step.done ? 'bg-green-50 text-green-600' : 'bg-blue-50 text-blue-600'}`}>
                  <Icon className="w-5 h-5" />
                </div>
                <ArrowRight className="w-4 h-4 text-gray-400 group-hover:text-blue-600 transition-colors" />
              </div>
              <h3 className="font-semibold text-gray-900 mt-3">{step.label}</h3>
              <p className="text-sm text-gray-500 mt-1">{step.desc}</p>
            </Link>
          )
        })}
      </div>
    </div>
  )
}
