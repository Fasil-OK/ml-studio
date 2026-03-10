import { useState, useEffect } from 'react'
import { useParams, useNavigate } from 'react-router-dom'
import { Check, Cpu, Zap } from 'lucide-react'
import { getModels, createExperiment, getResources } from '../api/training'
import { getDataset } from '../api/datasets'
import { useProjectStore } from '../stores/projectStore'
import toast from 'react-hot-toast'

export default function ModelSelectPage() {
  const { id } = useParams()
  const navigate = useNavigate()
  const project = useProjectStore((s) => s.currentProject)
  const [models, setModels] = useState<any[]>([])
  const [selected, setSelected] = useState<string>('')
  const [resources, setResources] = useState<any>(null)
  const [datasetId, setDatasetId] = useState<string>('')

  // Hyperparameters
  const [lr, setLr] = useState('0.001')
  const [batchSize, setBatchSize] = useState('32')
  const [epochs, setEpochs] = useState('50')
  const [optimizer, setOptimizer] = useState('adam')
  const [augmentation, setAugmentation] = useState('light')
  const [scheduler, setScheduler] = useState('cosine')
  const [mixedPrecision, setMixedPrecision] = useState(false)

  useEffect(() => {
    if (project) {
      getModels(project.task_type).then((m) => {
        setModels(m)
        if (m.length > 0) setSelected(m[0].name)
      })
    }
    if (id) {
      getDataset(id).then((d) => { if (d) setDatasetId(d.id) })
    }
    getResources().then(setResources)
  }, [project, id])

  const handleCreate = async () => {
    if (!id || !selected || !datasetId) return
    try {
      const experiment = await createExperiment(id, {
        dataset_id: datasetId,
        architecture: selected,
        pretrained: true,
        hyperparameters: {
          lr: parseFloat(lr),
          batch_size: parseInt(batchSize),
          epochs: parseInt(epochs),
          optimizer,
          augmentation,
          scheduler,
        },
        resource_config: {
          device: resources?.gpu ? 'cuda' : 'cpu',
          mixed_precision: mixedPrecision,
          num_workers: 2,
        },
      })
      toast.success('Experiment created!')
      navigate(`/project/${id}/train`)
    } catch {
      toast.error('Failed to create experiment')
    }
  }

  const selectedModel = models.find((m) => m.name === selected)

  return (
    <div>
      <h2 className="text-xl font-bold text-gray-900 mb-6">Select Model & Configure</h2>

      {/* Resource Info */}
      {resources && (
        <div className="bg-white rounded-xl border border-gray-200 p-4 mb-6 flex gap-6 text-sm">
          <div className="flex items-center gap-2">
            <Cpu className="w-4 h-4 text-gray-400" />
            <span className="text-gray-600">CPU: {resources.cpu?.count} cores</span>
          </div>
          <div className="flex items-center gap-2">
            <span className="text-gray-600">RAM: {resources.ram?.available_gb}GB free</span>
          </div>
          {resources.gpu && (
            <div className="flex items-center gap-2">
              <Zap className="w-4 h-4 text-green-500" />
              <span className="text-gray-600">{resources.gpu.name} ({resources.gpu.vram_total_mb}MB)</span>
            </div>
          )}
        </div>
      )}

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Architecture Grid */}
        <div className="lg:col-span-2">
          <h3 className="text-sm font-semibold text-gray-700 mb-3">Architectures</h3>
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
            {models.map((m) => (
              <div
                key={m.name}
                onClick={() => setSelected(m.name)}
                className={`bg-white rounded-xl border p-4 cursor-pointer transition-all ${
                  selected === m.name ? 'border-blue-500 ring-2 ring-blue-100' : 'border-gray-200 hover:border-blue-300'
                }`}
              >
                <div className="flex items-start justify-between">
                  <h4 className="font-semibold text-gray-900 text-sm">{m.display_name}</h4>
                  {selected === m.name && <Check className="w-4 h-4 text-blue-600" />}
                </div>
                <p className="text-xs text-gray-500 mt-1">{m.description}</p>
                <div className="flex gap-3 mt-3 text-xs text-gray-400">
                  <span>{m.params} params</span>
                  <span>{m.vram_mb}MB VRAM</span>
                  <span>{m.speed}</span>
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Hyperparameters */}
        <div className="bg-white rounded-xl border border-gray-200 p-5 h-fit">
          <h3 className="text-sm font-semibold text-gray-900 mb-4">Hyperparameters</h3>
          <div className="space-y-3">
            <div>
              <label className="text-xs text-gray-500">Learning Rate</label>
              <input value={lr} onChange={(e) => setLr(e.target.value)} className="w-full px-3 py-1.5 border border-gray-300 rounded-lg text-sm mt-1 outline-none focus:ring-2 focus:ring-blue-500" />
            </div>
            <div>
              <label className="text-xs text-gray-500">Batch Size</label>
              <select value={batchSize} onChange={(e) => setBatchSize(e.target.value)} className="w-full px-3 py-1.5 border border-gray-300 rounded-lg text-sm mt-1 outline-none">
                <option value="8">8</option><option value="16">16</option><option value="32">32</option><option value="64">64</option>
              </select>
            </div>
            <div>
              <label className="text-xs text-gray-500">Epochs</label>
              <input type="number" value={epochs} onChange={(e) => setEpochs(e.target.value)} className="w-full px-3 py-1.5 border border-gray-300 rounded-lg text-sm mt-1 outline-none focus:ring-2 focus:ring-blue-500" />
            </div>
            <div>
              <label className="text-xs text-gray-500">Optimizer</label>
              <select value={optimizer} onChange={(e) => setOptimizer(e.target.value)} className="w-full px-3 py-1.5 border border-gray-300 rounded-lg text-sm mt-1 outline-none">
                <option value="adam">Adam</option><option value="adamw">AdamW</option><option value="sgd">SGD</option>
              </select>
            </div>
            <div>
              <label className="text-xs text-gray-500">Augmentation</label>
              <select value={augmentation} onChange={(e) => setAugmentation(e.target.value)} className="w-full px-3 py-1.5 border border-gray-300 rounded-lg text-sm mt-1 outline-none">
                <option value="none">None</option><option value="light">Light</option><option value="heavy">Heavy</option>
              </select>
            </div>
            <div>
              <label className="text-xs text-gray-500">Scheduler</label>
              <select value={scheduler} onChange={(e) => setScheduler(e.target.value)} className="w-full px-3 py-1.5 border border-gray-300 rounded-lg text-sm mt-1 outline-none">
                <option value="cosine">Cosine</option><option value="step">Step</option><option value="plateau">Plateau</option>
              </select>
            </div>
            <label className="flex items-center gap-2 text-sm text-gray-700 pt-2">
              <input type="checkbox" checked={mixedPrecision} onChange={(e) => setMixedPrecision(e.target.checked)} className="rounded" />
              Mixed Precision (FP16)
            </label>
          </div>
          <button
            onClick={handleCreate}
            className="w-full mt-5 bg-blue-600 text-white py-2.5 rounded-lg text-sm font-medium hover:bg-blue-700 transition-colors"
          >
            Create Experiment & Train
          </button>
        </div>
      </div>
    </div>
  )
}
