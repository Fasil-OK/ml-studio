import { useState, useEffect } from 'react'
import { useParams } from 'react-router-dom'
import { Play, Square, Loader2 } from 'lucide-react'
import { listExperiments, startTraining, stopTraining, getTrainingStatus, type Experiment } from '../api/training'
import { useTrainingStore } from '../stores/trainingStore'
import { useTrainingSocket } from '../hooks/useTrainingSocket'
import { LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer, Legend } from 'recharts'
import toast from 'react-hot-toast'

export default function TrainingPage() {
  const { id } = useParams()
  const [experiment, setExperiment] = useState<Experiment | null>(null)
  const { metrics, status, currentEpoch, totalEpochs, setMetrics, setStatus, reset } = useTrainingStore()

  useEffect(() => {
    if (id) {
      listExperiments(id).then((exps) => {
        if (exps.length > 0) {
          const exp = exps[exps.length - 1]
          setExperiment(exp)
          setStatus(exp.status)
          // Load existing metrics
          getTrainingStatus(exp.id).then((s) => {
            if (s.metrics.length > 0) setMetrics(s.metrics)
          })
        }
      })
    }
    return () => reset()
  }, [id])

  useTrainingSocket(experiment?.status === 'running' ? experiment.id : null)

  const handleStart = async () => {
    if (!experiment) return
    try {
      await startTraining(experiment.id)
      setStatus('running')
      setExperiment({ ...experiment, status: 'running' })
      toast.success('Training started!')
    } catch {
      toast.error('Failed to start training')
    }
  }

  const handleStop = async () => {
    if (!experiment) return
    try {
      await stopTraining(experiment.id)
      setStatus('stopped')
      toast.success('Training stopped')
    } catch {
      toast.error('Failed to stop')
    }
  }

  const chartData = metrics.map((m) => ({
    epoch: m.epoch,
    train_loss: m.train_loss,
    val_loss: m.val_loss,
    train_acc: m.train_accuracy ? +(m.train_accuracy * 100).toFixed(1) : null,
    val_acc: m.val_accuracy ? +(m.val_accuracy * 100).toFixed(1) : null,
  }))

  const lastMetric = metrics.length > 0 ? metrics[metrics.length - 1] : null

  return (
    <div>
      <div className="flex items-center justify-between mb-6">
        <div>
          <h2 className="text-xl font-bold text-gray-900">Training</h2>
          {experiment && (
            <p className="text-sm text-gray-500 mt-1">{experiment.architecture} - {status}</p>
          )}
        </div>
        <div className="flex gap-3">
          {status !== 'running' && (
            <button onClick={handleStart} className="flex items-center gap-2 bg-green-600 text-white px-4 py-2 rounded-lg text-sm hover:bg-green-700">
              <Play className="w-4 h-4" /> Start Training
            </button>
          )}
          {status === 'running' && (
            <button onClick={handleStop} className="flex items-center gap-2 bg-red-600 text-white px-4 py-2 rounded-lg text-sm hover:bg-red-700">
              <Square className="w-4 h-4" /> Stop
            </button>
          )}
        </div>
      </div>

      {/* Progress Bar */}
      {currentEpoch !== null && totalEpochs !== null && (
        <div className="bg-white rounded-xl border border-gray-200 p-4 mb-6">
          <div className="flex justify-between text-sm text-gray-600 mb-2">
            <span>Epoch {currentEpoch} / {totalEpochs}</span>
            <span>{Math.round((currentEpoch / totalEpochs) * 100)}%</span>
          </div>
          <div className="w-full bg-gray-200 rounded-full h-2">
            <div className="bg-blue-600 h-2 rounded-full transition-all" style={{ width: `${(currentEpoch / totalEpochs) * 100}%` }} />
          </div>
        </div>
      )}

      {/* Metric Cards */}
      {lastMetric && (
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
          <div className="bg-white rounded-xl border border-gray-200 p-4">
            <p className="text-xs text-gray-500">Train Loss</p>
            <p className="text-xl font-bold text-gray-900">{lastMetric.train_loss?.toFixed(4)}</p>
          </div>
          <div className="bg-white rounded-xl border border-gray-200 p-4">
            <p className="text-xs text-gray-500">Val Loss</p>
            <p className="text-xl font-bold text-gray-900">{lastMetric.val_loss?.toFixed(4)}</p>
          </div>
          <div className="bg-white rounded-xl border border-gray-200 p-4">
            <p className="text-xs text-gray-500">Train Accuracy</p>
            <p className="text-xl font-bold text-gray-900">{lastMetric.train_accuracy ? (lastMetric.train_accuracy * 100).toFixed(1) + '%' : '-'}</p>
          </div>
          <div className="bg-white rounded-xl border border-gray-200 p-4">
            <p className="text-xs text-gray-500">Val Accuracy</p>
            <p className="text-xl font-bold text-green-600">{lastMetric.val_accuracy ? (lastMetric.val_accuracy * 100).toFixed(1) + '%' : '-'}</p>
          </div>
        </div>
      )}

      {/* Charts */}
      {chartData.length > 0 && (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          <div className="bg-white rounded-xl border border-gray-200 p-5">
            <h3 className="text-sm font-semibold text-gray-900 mb-3">Loss</h3>
            <div className="h-64">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={chartData}>
                  <XAxis dataKey="epoch" fontSize={11} />
                  <YAxis fontSize={11} />
                  <Tooltip />
                  <Legend />
                  <Line type="monotone" dataKey="train_loss" stroke="#3B82F6" name="Train" dot={false} strokeWidth={2} />
                  <Line type="monotone" dataKey="val_loss" stroke="#EF4444" name="Val" dot={false} strokeWidth={2} />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </div>
          <div className="bg-white rounded-xl border border-gray-200 p-5">
            <h3 className="text-sm font-semibold text-gray-900 mb-3">Accuracy</h3>
            <div className="h-64">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={chartData}>
                  <XAxis dataKey="epoch" fontSize={11} />
                  <YAxis fontSize={11} domain={[0, 100]} />
                  <Tooltip />
                  <Legend />
                  <Line type="monotone" dataKey="train_acc" stroke="#3B82F6" name="Train %" dot={false} strokeWidth={2} />
                  <Line type="monotone" dataKey="val_acc" stroke="#10B981" name="Val %" dot={false} strokeWidth={2} />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </div>
        </div>
      )}

      {/* Waiting state */}
      {status === 'running' && metrics.length === 0 && (
        <div className="text-center py-16 bg-white rounded-xl border border-gray-200">
          <Loader2 className="w-8 h-8 text-blue-600 animate-spin mx-auto mb-3" />
          <p className="text-gray-600">Training in progress, waiting for first epoch...</p>
        </div>
      )}
    </div>
  )
}
