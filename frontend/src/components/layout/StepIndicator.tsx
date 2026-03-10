import { Check } from 'lucide-react'

interface Step {
  label: string
  completed: boolean
  active: boolean
}

export default function StepIndicator({ steps }: { steps: Step[] }) {
  return (
    <div className="flex items-center gap-2 mb-6">
      {steps.map((step, i) => (
        <div key={step.label} className="flex items-center">
          <div className={`flex items-center gap-2 px-3 py-1.5 rounded-full text-sm ${
            step.active
              ? 'bg-blue-100 text-blue-700 font-medium'
              : step.completed
              ? 'bg-green-50 text-green-700'
              : 'bg-gray-100 text-gray-400'
          }`}>
            {step.completed && <Check className="w-3.5 h-3.5" />}
            {step.label}
          </div>
          {i < steps.length - 1 && <div className="w-8 h-px bg-gray-200 mx-1" />}
        </div>
      ))}
    </div>
  )
}
