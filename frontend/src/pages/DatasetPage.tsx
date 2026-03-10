import { useState, useEffect } from 'react'
import { useParams } from 'react-router-dom'
import { Upload, AlertTriangle, Image, BarChart3 } from 'lucide-react'
import { uploadDataset, getDataset, type DatasetInfo } from '../api/datasets'
import { useDropzone } from 'react-dropzone'
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer } from 'recharts'
import toast from 'react-hot-toast'

export default function DatasetPage() {
  const { id } = useParams()
  const [dataset, setDataset] = useState<DatasetInfo | null>(null)
  const [uploading, setUploading] = useState(false)

  useEffect(() => {
    if (id) getDataset(id).then(setDataset)
  }, [id])

  const onDrop = async (files: File[]) => {
    if (!id || files.length === 0) return
    setUploading(true)
    try {
      const result = await uploadDataset(id, files[0])
      setDataset(result)
      toast.success('Dataset uploaded and analyzed!')
    } catch (err: any) {
      const detail = err?.response?.data?.detail
      const status = err?.response?.status
      if (status === 404) {
        toast.error('Project not found. Please go back and create a new project.')
      } else {
        toast.error(`Upload failed: ${detail || err?.message || 'Unknown error'}`)
      }
    }
    setUploading(false)
  }

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: { 'application/zip': ['.zip'] },
    multiple: false,
  })

  const chartData = dataset?.class_counts
    ? Object.entries(dataset.class_counts).map(([name, count]) => ({ name, count }))
    : []

  return (
    <div>
      <h2 className="text-xl font-bold text-gray-900 mb-6">Dataset</h2>

      {/* Upload Zone */}
      <div
        {...getRootProps()}
        className={`border-2 border-dashed rounded-xl p-10 text-center cursor-pointer transition-colors mb-6 ${
          isDragActive ? 'border-blue-500 bg-blue-50' : 'border-gray-300 hover:border-blue-400 bg-white'
        }`}
      >
        <input {...getInputProps()} />
        <Upload className="w-10 h-10 text-gray-400 mx-auto mb-3" />
        {uploading ? (
          <p className="text-sm text-blue-600 font-medium">Uploading & analyzing...</p>
        ) : (
          <>
            <p className="text-sm text-gray-600 font-medium">
              {isDragActive ? 'Drop your dataset ZIP here' : 'Drag & drop a ZIP file, or click to browse'}
            </p>
            <p className="text-xs text-gray-400 mt-1">
              Classification: class_name/images | Detection: COCO/YOLO format | Segmentation: images/ + masks/
            </p>
          </>
        )}
      </div>

      {/* Analysis Results */}
      {dataset && (
        <div className="space-y-6">
          {/* Stats Cards */}
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <div className="bg-white rounded-xl border border-gray-200 p-4">
              <p className="text-sm text-gray-500">Total Images</p>
              <p className="text-2xl font-bold text-gray-900">{dataset.total_images?.toLocaleString()}</p>
            </div>
            <div className="bg-white rounded-xl border border-gray-200 p-4">
              <p className="text-sm text-gray-500">Classes</p>
              <p className="text-2xl font-bold text-gray-900">{dataset.num_classes}</p>
            </div>
            <div className="bg-white rounded-xl border border-gray-200 p-4">
              <p className="text-sm text-gray-500">Format</p>
              <p className="text-2xl font-bold text-gray-900">{dataset.annotation_format}</p>
            </div>
            <div className="bg-white rounded-xl border border-gray-200 p-4">
              <p className="text-sm text-gray-500">Avg Size</p>
              <p className="text-2xl font-bold text-gray-900">
                {dataset.image_stats?.avg_width}x{dataset.image_stats?.avg_height}
              </p>
            </div>
          </div>

          {/* Class Distribution */}
          {chartData.length > 0 && (
            <div className="bg-white rounded-xl border border-gray-200 p-5">
              <h3 className="text-sm font-semibold text-gray-900 mb-4 flex items-center gap-2">
                <BarChart3 className="w-4 h-4" /> Class Distribution
              </h3>
              <div className="h-64">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={chartData}>
                    <XAxis dataKey="name" fontSize={11} angle={-45} textAnchor="end" height={80} />
                    <YAxis fontSize={11} />
                    <Tooltip />
                    <Bar dataKey="count" fill="#3B82F6" radius={[4, 4, 0, 0]} />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </div>
          )}

          {/* Quality Issues */}
          {dataset.quality_issues && dataset.quality_issues.length > 0 && (
            <div className="bg-amber-50 rounded-xl border border-amber-200 p-5">
              <h3 className="text-sm font-semibold text-amber-800 mb-3 flex items-center gap-2">
                <AlertTriangle className="w-4 h-4" /> Quality Issues
              </h3>
              <ul className="space-y-2">
                {dataset.quality_issues.map((issue, i) => (
                  <li key={i} className="text-sm text-amber-700">
                    <strong>{issue.type}:</strong> {issue.message || issue.count || `${issue.class} (${issue.count} samples)`}
                  </li>
                ))}
              </ul>
            </div>
          )}
        </div>
      )}
    </div>
  )
}
