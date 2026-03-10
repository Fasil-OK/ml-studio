import { useState, useEffect } from 'react'
import { useParams } from 'react-router-dom'
import { Eye, Upload } from 'lucide-react'
import { useDropzone } from 'react-dropzone'
import { listExperiments } from '../api/training'
import { generateExplanation } from '../api/inference'
import toast from 'react-hot-toast'

export default function ExplanationPage() {
  const { id } = useParams()
  const [experimentId, setExperimentId] = useState('')
  const [method, setMethod] = useState('gradcam')
  const [result, setResult] = useState<any>(null)
  const [loading, setLoading] = useState(false)
  const [imagePreview, setImagePreview] = useState<string>('')

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
    setImagePreview(URL.createObjectURL(file))
    setLoading(true)
    try {
      const res = await generateExplanation(experimentId, file, method)
      setResult(res)
    } catch {
      toast.error('Explanation generation failed')
    }
    setLoading(false)
  }

  const { getRootProps, getInputProps } = useDropzone({ onDrop, accept: { 'image/*': [] }, multiple: false })

  return (
    <div>
      <h2 className="text-xl font-bold text-gray-900 mb-6">Explainability (Causal AI)</h2>

      <div className="flex gap-4 mb-6">
        <select
          value={method}
          onChange={(e) => setMethod(e.target.value)}
          className="px-3 py-2 border border-gray-300 rounded-lg text-sm outline-none"
        >
          <option value="gradcam">GradCAM</option>
          <option value="integrated_gradients">Integrated Gradients</option>
          <option value="shap">GradientSHAP</option>
        </select>
      </div>

      <div
        {...getRootProps()}
        className="border-2 border-dashed rounded-xl p-8 text-center cursor-pointer bg-white border-gray-300 hover:border-blue-400 transition-colors mb-6"
      >
        <input {...getInputProps()} />
        <Upload className="w-8 h-8 text-gray-400 mx-auto mb-2" />
        <p className="text-sm text-gray-600">Drop an image to explain the model's prediction</p>
      </div>

      {loading && (
        <div className="text-center py-8">
          <p className="text-gray-500">Generating explanation...</p>
        </div>
      )}

      {result && (
        <div className="space-y-6">
          <div className="bg-white rounded-xl border border-gray-200 p-5">
            <div className="flex items-center gap-3 mb-4">
              <Eye className="w-5 h-5 text-blue-600" />
              <div>
                <p className="font-semibold text-gray-900">Prediction: {result.prediction}</p>
                <p className="text-sm text-gray-500">Confidence: {result.confidence}%</p>
              </div>
            </div>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            {imagePreview && (
              <div className="bg-white rounded-xl border border-gray-200 p-4">
                <p className="text-xs font-medium text-gray-500 mb-2">Original</p>
                <img src={imagePreview} alt="Input" className="w-full rounded-lg" />
              </div>
            )}
            {result.heatmap_path && (
              <div className="bg-white rounded-xl border border-gray-200 p-4">
                <p className="text-xs font-medium text-gray-500 mb-2">Heatmap</p>
                <img src={result.heatmap_path} alt="Heatmap" className="w-full rounded-lg" />
              </div>
            )}
            {result.overlay_path && (
              <div className="bg-white rounded-xl border border-gray-200 p-4">
                <p className="text-xs font-medium text-gray-500 mb-2">Overlay</p>
                <img src={result.overlay_path} alt="Overlay" className="w-full rounded-lg" />
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  )
}
