// frontend/src/components/charts/RiskDistributionChart.tsx

import { Bar, BarChart, CartesianGrid, Cell, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import type { RiskTierBreakdown } from "@/api/types";

interface RiskDistributionChartProps {
  data: RiskTierBreakdown[];
}

const TIER_COLORS: Record<string, string> = {
  high: "#ef4444",
  medium: "#f59e0b",
  low: "#22c55e",
};

const TIER_ORDER = ["low", "medium", "high"];

export default function RiskDistributionChart({ data }: RiskDistributionChartProps) {
  const chartData = [...data]
    .sort((a, b) => TIER_ORDER.indexOf(a.tier) - TIER_ORDER.indexOf(b.tier))
    .map((d) => ({ tier: d.tier, count: d.count }));

  return (
    <div className="rounded-lg border border-gray-200 bg-white p-5">
      <div className="text-sm font-medium text-white mb-4">Customers by Risk Tier</div>
      <ResponsiveContainer width="100%" height={260}>
        <BarChart data={chartData} margin={{ top: 8, right: 8, left: 0, bottom: 0 }}>
          <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#f0f0f0" />
          <XAxis dataKey="tier" tickFormatter={(v) => v.charAt(0).toUpperCase() + v.slice(1)} />
          <YAxis allowDecimals={false} />
          <Tooltip
            formatter={(value: number) => [value.toLocaleString(), "Customers"]}
            labelFormatter={(label) => `${String(label).charAt(0).toUpperCase()}${String(label).slice(1)} risk`}
          />
          <Bar dataKey="count" radius={[4, 4, 0, 0]}>
            {chartData.map((entry) => (
              <Cell key={entry.tier} fill={TIER_COLORS[entry.tier] ?? "#94a3b8"} />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}