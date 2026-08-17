import {
  Bar,
  BarChart,
  CartesianGrid,
  LabelList,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

interface ShapGlobalImportanceItem {
  feature: string;
  mean_abs_shap: number;
}

interface ShapSummaryChartProps {
  data: ShapGlobalImportanceItem[];
  topN?: number;
}

export default function ShapSummaryChart({
  data,
  topN = 15,
}: ShapSummaryChartProps) {
  const chartData = [...data]
    .map((item) => ({
      feature: String(item.feature),
      importance: Number(item.mean_abs_shap),
    }))
    .filter((item) => Number.isFinite(item.importance))
    .sort((a, b) => b.importance - a.importance)
    .slice(0, topN);

  const maxImportance = Math.max(
    ...chartData.map((item) => item.importance),
    0
  );

  const xAxisMax =
    maxImportance > 0 ? maxImportance * 1.15 : 1;

  return (
    <div className="rounded-lg border border-gray-700 bg-gray-900 p-5">
      {/* Header */}
      <div className="mb-4">
        <div className="text-sm font-medium text-gray-200">
          Top {chartData.length} Features by SHAP Importance
        </div>

        <div className="mt-1 text-xs text-gray-400">
          Mean absolute SHAP value — higher values indicate greater
          influence on churn prediction
        </div>
      </div>

      {/* Empty state */}
      {chartData.length === 0 ? (
        <div className="flex h-[300px] items-center justify-center text-gray-400">
          No SHAP data available
        </div>
      ) : (
        <ResponsiveContainer
          width="100%"
          height={Math.max(400, chartData.length * 35)}
        >
          <BarChart
            data={chartData}
            layout="vertical"
            margin={{
              top: 10,
              right: 80,
              left: 10,
              bottom: 10,
            }}
          >
            {/* Grid */}
            <CartesianGrid
              strokeDasharray="3 3"
              horizontal={false}
              stroke="#374151"
            />

            {/* X Axis */}
            <XAxis
              type="number"
              domain={[0, xAxisMax]}
              tick={{
                fill: "#d1d5db",
                fontSize: 12,
              }}
              tickFormatter={(value) =>
                Number(value).toFixed(2)
              }
              axisLine={{
                stroke: "#6b7280",
              }}
              tickLine={{
                stroke: "#6b7280",
              }}
            />

            {/* Y Axis */}
            <YAxis
              type="category"
              dataKey="feature"
              width={180}
              tick={{
                fill: "#d1d5db",
                fontSize: 12,
              }}
              axisLine={{
                stroke: "#6b7280",
              }}
              tickLine={{
                stroke: "#6b7280",
              }}
            />

            {/* Tooltip */}
            <Tooltip
              cursor={{
                fill: "rgba(59, 130, 246, 0.08)",
              }}
              contentStyle={{
                backgroundColor: "#111827",
                border: "1px solid #4b5563",
                borderRadius: "8px",
                color: "#ffffff",
              }}
              labelStyle={{
                color: "#ffffff",
                fontWeight: 600,
              }}
              formatter={(value) => [
                Number(value).toFixed(6),
                "Mean |SHAP|",
              ]}
            />

            {/* SHAP Bars */}
            <Bar
              dataKey="importance"
              fill="#3b82f6"
              stroke="#60a5fa"
              strokeWidth={1}
              barSize={20}
              minPointSize={5}
              radius={[0, 5, 5, 0]}
            >
              <LabelList
                dataKey="importance"
                position="right"
                fill="#d1d5db"
                fontSize={11}
                formatter={(value) =>
                  Number(value).toFixed(4)
                }
              />
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      )}
    </div>
  );
}