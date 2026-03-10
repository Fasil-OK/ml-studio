export const TASK_TYPES = [
  { value: 'classification', label: 'Image Classification' },
  { value: 'detection', label: 'Object Detection' },
  { value: 'segmentation', label: 'Image Segmentation' },
] as const

export const SPEED_LABELS: Record<string, string> = {
  very_fast: 'Very Fast',
  fast: 'Fast',
  medium: 'Medium',
  slow: 'Slow',
}

export const ACCURACY_LABELS: Record<string, string> = {
  fair: 'Fair',
  good: 'Good',
  very_good: 'Very Good',
  excellent: 'Excellent',
}
