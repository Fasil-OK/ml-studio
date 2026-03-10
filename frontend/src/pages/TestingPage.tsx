import { useState, useEffect } from 'react'
import { useParams } from 'react-router-dom'
import { Upload, FlaskConical, Clock } from 'lucide-react'
import { useDropzone } from 'react-dropzone'
import { listExperiments } from '../api/training'
import { predict, type PredictionResult } from '../api/inference'
import toast from 'react-hot-toast'

export default function TestingPage() {
  const { id } = useParams()
  const [experimentId, setExperimentId] = useState('')
  const [result, setResult] = useState<PredictionResult | null>(null)
  const [loading, setLoading] = useState(false)
  const [imagePreview, setImagePreview] = useState('')
  const [history, setHistory] = useState<Array<{ image: string; result: PredictionResult }>>([])

  useEffect(() => {
    if (id) {
      listExperiments(id).then((exps) => {
        const completed = exps.filter(e => e.status === 'completed' || e.status === 'stopped')
        if (completed.length > 0) setExperimentId(completed[completed.length - 1].id)
      })
    }
  }, [id])

  const onDrop = async (files: File[]) => {
    if (!experimentId || files.length === 0) return
    const file = files[0]
    const preview = URL.createObjectURL(file)
    setImagePreview(preview)
    setLoading(true)
    try {
      const res = await predict(experimentId, file)
      setResult(res)
      setHistory((prev) => [{ image: preview, result: res }, ...prev.slice(0, 9)])
    } catch {
      toast.error('Prediction failed')
    }
    setLoading(false)
  }

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: { 'image/*': [] },
    multiple: false,
  })

  return (
    <div>
      <h2 className="text-xl font-bold text-gray-900 mb-6">Live Testing</h2>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div>
          {/* Drop zone */}
          <div
            {...getRootProps()}
            className={`border-2 border-dashed rounded-xl p-10 text-center cursor-pointer bg-white transition-colors ${
              isDragActive ? 'border-blue-500 bg-blue-50' : 'border-gray-300 hover:border-blue-400'
            }`}
          >
            <input {...getInputProps()} />
            {imagePreview ? (
              <img src={imagePreview} alt="Test" className="max-h-64 mx-auto rounded-lg" />
            ) : (
              <>
                <FlaskConical className="w-10 h-10 text-gray-400 mx-auto mb-3" />
                <p className="text-sm text-gray-600">Drop an image to test your model</p>
              </>
            )}
          </div>
        </div>

        {/* Results */}
        <div>
          {loading && (
            <div className="bg-white rounded-xl border border-gray-200 p-8 text-center">
              <p className="text-gray-500">Running prediction...</p>
            </div>
          )}
          {result && !loading && (
            <div className="bg-white rounded-xl border border-gray-200 p-5">
              <div className="flex items-center justify-between mb-4">
                <h3 className="text-sm font-semibold text-gray-900">Predictions</h3>
                <span className="flex items-center gap-1 text-xs text-gray-400">
                  <Clock className="w-3 h-3" /> {result.processing_time_ms}ms
                </span>
              </div>
              <div className="space-y-3">
                {result.predictions.map((p, i) => (
                  <div key={i}>
                    <div className="flex justify-between text-sm mb-1">
                      <span className={`font-medium ${i === 0 ? 'text-blue-700' : 'text-gray-700'}`}>{p.class}</span>
                      <span className="text-gray-500">{p.confidence.toFixed(1)}%</span>
                    </div>
                    <div className="w-full bg-gray-100 rounded-full h-2">
                      <div
                        className={`h-2 rounded-full ${i === 0 ? 'bg-blue-500' : 'bg-gray-300'}`}
                        style={{ width: `${p.confidence}%` }}
                      />
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      </div>

      {/* History */}
      {history.length > 1 && (
        <div className="mt-8">
          <h3 className="text-sm font-semibold text-gray-900 mb-3">Test History</h3>
          <div className="grid grid-cols-2 md:grid-cols-5 gap-3">
            {history.slice(1).map((h, i) => (
              <div key={i} className="bg-white rounded-lg border border-gray-200 p-2">
                <img src={h.image} alt="History" className="w-full h-20 object-cover rounded" />
                <p className="text-xs font-medium text-gray-700 mt-1 truncate">{h.result.predictions[0]?.class}</p>
                <p className="text-xs text-gray-400">{h.result.predictions[0]?.confidence.toFixed(1)}%</p>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  )
}
