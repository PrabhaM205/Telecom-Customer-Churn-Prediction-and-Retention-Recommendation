interface PriorityBadgeProps {
  priorityScore: number; // 0–1
}

function getTier(score: number): { label: string; className: string } {
  if (score >= 0.7) return { label: "High Priority", className: "bg-red-50 text-red-700 border-red-200" };
  if (score >= 0.4) return { label: "Medium Priority", className: "bg-amber-50 text-amber-700 border-amber-200" };
  return { label: "Low Priority", className: "bg-green-50 text-green-700 border-green-200" };
}

export default function PriorityBadge({ priorityScore }: PriorityBadgeProps) {
  const { label, className } = getTier(priorityScore);
  return (
    <span className={`inline-flex items-center rounded-full border px-2.5 py-0.5 text-xs font-medium ${className}`}>
      {label}
    </span>
  );
}