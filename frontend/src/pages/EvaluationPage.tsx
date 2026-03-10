import { useState, useEffect } from 'react'
import { useParams } from 'react-router-dom'
import { Play, BarChart3 } from 'lucide-react'
import { listExperiments } from '../api/training'
import { runEvaluation, getEvaluation, type EvaluationResult } from '../api/evaluation'
import toast from 'react-hot-toast'

export default function EvaluationPage() {
  const { id } = useParams()
  const [evaluation, setEvaluation] = useState<EvaluationResult | null>(null)
  const [evaluating, setEvaluating] = useState(false)
  const [experimentId, setExperimentId] = useState<string>('')

  useEffect(() => {
    if (id) {
      listExperiments(id).then((exps) => {
        const completed = exps.filter(e => e.status === 'completed' || e.status === 'stopped')
        if (completed.length > 0) {
          const expId = completed[completed.length - 1].id
          setExperimentId(expId)
          getEvaluation(expId).then((e) => { if (e) setEvaluation(e) })
        }
      })
    }
  }, [id])

  const handleEvaluate = async () => {
    if (!experimentId) return
    setEvaluating(true)
    try {
      const result = await runEvaluation(experimentId)
      setEvaluation(result)
      toast.success('Evaluation complete!')
    } catch {
      toast.error('Evaluation failed')
    }
    setEvaluating(false)
  }

  return (
    <div>
      <div className="flex items-center justify-between mb-6">
        <h2 className="text-xl font-bold text-gray-900">Evaluation</h2>
        {!evaluation && (
          <button
            onClick={handleEvaluate}
            disabled={evaluating || !experimentId}
            className="flex items-center gap-2 bg-blue-600 text-white px-4 py-2 rounded-lg text-sm hover:bg-blue-700 disabled:opacity-50"
          >
            <Play className="w-4 h-4" /> {evaluating ? 'Evaluating...' : 'Run Evaluation'}
          </button>
        )}
      </div>

      {evaluation && (
        <div className="space-y-6">
          {/* Metric Cards */}
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            {Object.entries(evaluation.metrics).map(([key, value]) => (
              <div key={key} className="bg-white rounded-xl border border-gray-200 p-5">
                <p className="text-sm text-gray-500 capitalize">{key.replace(/_/g, ' ')}</p>
                <p className="text-3xl font-bold text-gray-900 mt-1">
                  {typeof value === 'number' ? (value * 100).toFixed(1) + '%' : value}
                </p>
              </div>
            ))}
          </div>

          {/* Confusion Matrix */}
          {evaluation.confusion_matrix && (
            <div className="bg-white rounded-xl border border-gray-200 p-5">
              <h3 className="text-sm font-semibold text-gray-900 mb-4 flex items-center gap-2">
                <BarChart3 className="w-4 h-4" /> Confusion Matrix
              </h3>
              <div className="overflow-auto">
                <table className="text-xs">
                  <tbody>
                    {evaluation.confusion_matrix.map((row, i) => (
                      <tr key={i}>
                        {row.map((val, j) => {
                          const maxVal = Math.max(...evaluation.confusion_matrix!.flat())
                          const intensity = val / Math.max(maxVal, 1)
                          return (
                            <td
                              key={j}
                              className="w-10 h-10 text-center border border-gray-100"
                              style={{
                                backgroundColor: i === j
                                  ? `rgba(59, 130, 246, ${0.1 + intensity * 0.6})`
                                  : val > 0
                                  ? `rgba(239, 68, 68, ${intensity * 0.5})`
                                  : 'transparent',
                              }}
                            >
                              {val}
                            </td>
                          )
                        })}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}

          {/* Per-class metrics */}
          {evaluation.per_class_metrics && evaluation.per_class_metrics.length > 0 && (
            <div className="bg-white rounded-xl border border-gray-200 p-5">
              <h3 className="text-sm font-semibold text-gray-900 mb-4">Per-Class Metrics</h3>
              <div className="overflow-auto">
                <table className="w-full text-sm">
                  <thead>
                    <tr className="border-b border-gray-200">
                      <th className="text-left py-2 px-3 text-gray-500 font-medium">Class</th>
                      <th className="text-right py-2 px-3 text-gray-500 font-medium">Precision</th>
                      <th className="text-right py-2 px-3 text-gray-500 font-medium">Recall</th>
                      <th className="text-right py-2 px-3 text-gray-500 font-medium">F1</th>
                      <th className="text-right py-2 px-3 text-gray-500 font-medium">Support</th>
                    </tr>
                  </thead>
                  <tbody>
                    {evaluation.per_class_metrics.map((c: any) => (
                      <tr key={c.class || c.iou} className="border-b border-gray-100">
                        <td className="py-2 px-3 font-medium text-gray-900">{c.class}</td>
                        <td className="py-2 px-3 text-right">{c.precision ? (c.precision * 100).toFixed(1) + '%' : (c.iou ? (c.iou * 100).toFixed(1) + '%' : '-')}</td>
                        <td className="py-2 px-3 text-right">{c.recall ? (c.recall * 100).toFixed(1) + '%' : '-'}</td>
                        <td className="py-2 px-3 text-right">{c.f1 ? (c.f1 * 100).toFixed(1) + '%' : '-'}</td>
                        <td className="py-2 px-3 text-right text-gray-500">{c.support ?? '-'}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}
        </div>
      )}

      {!evaluation && !evaluating && (
        <div className="text-center py-20 bg-white rounded-xl border border-gray-200">
          <BarChart3 className="w-10 h-10 text-gray-300 mx-auto mb-3" />
          <p className="text-gray-500">Complete training first, then evaluate your model</p>
        </div>
      )}
    </div>
  )
}
