// frontend/src/components/kpi/KpiGrid.tsx

import type { ReactNode } from "react";

interface KpiGridProps {
  children: ReactNode;
}

export default function KpiGrid({ children }: KpiGridProps) {
  return <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-4">{children}</div>;
}
