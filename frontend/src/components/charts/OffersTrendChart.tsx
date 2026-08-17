// frontend/src/components/charts/OffersTrendChart.tsx

import {
  Bar,
  CartesianGrid,
  ComposedChart,
  Legend,
  Line,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

import type { OfferTrendPoint } from "@/api/types";

interface OffersTrendChartProps {
  data: OfferTrendPoint[];
}

interface ChartDataPoint extends OfferTrendPoint {
  acceptanceRate: number;
  totalProcessed: number;
}

function formatDate(date: string): string {
  const parsedDate = new Date(date);

  if (Number.isNaN(parsedDate.getTime())) {
    return date;
  }

  return parsedDate.toLocaleDateString(
    "en-US",
    {
      month: "short",
      day: "numeric",
    }
  );
}

function CustomTooltip({
  active,
  payload,
  label,
}: any) {
  if (!active || !payload || !payload.length) {
    return null;
  }

  const data = payload[0]?.payload as
    | ChartDataPoint
    | undefined;

  if (!data) {
    return null;
  }

  return (
    <div className="min-w-[190px] rounded-lg border border-border bg-panel p-4 shadow-xl">

      <p className="mb-3 text-xs font-semibold text-ink">
        {formatDate(label)}
      </p>

      <div className="space-y-2">

        {/* Generated */}
        <div className="flex items-center justify-between gap-6">
          <div className="flex items-center gap-2">
            <span className="h-2 w-2 rounded-full bg-cyan-400" />

            <span className="text-xs text-ink-muted">
              Generated
            </span>
          </div>

          <span className="text-xs font-semibold text-ink">
            {data.generated}
          </span>
        </div>

        {/* Accepted */}
        <div className="flex items-center justify-between gap-6">
          <div className="flex items-center gap-2">
            <span className="h-2 w-2 rounded-full bg-emerald-400" />

            <span className="text-xs text-ink-muted">
              Accepted
            </span>
          </div>

          <span className="text-xs font-semibold text-ink">
            {data.accepted}
          </span>
        </div>

        {/* Dismissed */}
        <div className="flex items-center justify-between gap-6">
          <div className="flex items-center gap-2">
            <span className="h-2 w-2 rounded-full bg-rose-400" />

            <span className="text-xs text-ink-muted">
              Dismissed
            </span>
          </div>

          <span className="text-xs font-semibold text-ink">
            {data.dismissed}
          </span>
        </div>

        {/* Divider */}
        <div className="my-2 border-t border-border" />

        {/* Acceptance Rate */}
        <div className="flex items-center justify-between gap-6">
          <span className="text-xs text-ink-muted">
            Acceptance Rate
          </span>

          <span className="text-xs font-bold text-ink">
            {data.acceptanceRate.toFixed(1)}%
          </span>
        </div>

      </div>
    </div>
  );
}

export default function OffersTrendChart({
  data,
}: OffersTrendChartProps) {

  // ============================================================
  // PREPARE CHART DATA
  // ============================================================

  const chartData: ChartDataPoint[] =
    data.map((item) => {

      const totalProcessed =
        item.accepted +
        item.dismissed;

      const acceptanceRate =
        totalProcessed > 0
          ? (item.accepted / totalProcessed) * 100
          : 0;

      return {
        ...item,
        acceptanceRate,
        totalProcessed,
      };
    });

  // ============================================================
  // EMPTY STATE
  // ============================================================

  if (chartData.length === 0) {
    return (
      <div className="w-full">

        <div className="mb-5">
          <h2 className="text-sm font-semibold text-ink">
            Retention Offer Performance
          </h2>

          <p className="mt-1 text-xs text-ink-muted">
            Track generated, accepted, and dismissed
            retention offers.
          </p>
        </div>

        <div className="flex h-[320px] items-center justify-center rounded-lg border border-dashed border-border">
          <div className="text-center">

            <p className="text-sm font-medium text-ink">
              No offer activity yet
            </p>

            <p className="mt-1 text-xs text-ink-muted">
              Generated retention offers will appear here.
            </p>

          </div>
        </div>

      </div>
    );
  }

  // ============================================================
  // TOTALS
  // ============================================================

  const totalGenerated =
    chartData.reduce(
      (sum, item) =>
        sum + item.generated,
      0
    );

  const totalAccepted =
    chartData.reduce(
      (sum, item) =>
        sum + item.accepted,
      0
    );

  const totalDismissed =
    chartData.reduce(
      (sum, item) =>
        sum + item.dismissed,
      0
    );

  const totalProcessed =
    totalAccepted +
    totalDismissed;

  const overallAcceptanceRate =
    totalProcessed > 0
      ? (totalAccepted / totalProcessed) * 100
      : 0;

  // ============================================================
  // RENDER
  // ============================================================

  return (
    <div className="w-full">

      {/* ========================================================
          HEADER
          ======================================================== */}

      <div className="mb-5 flex flex-col gap-4 lg:flex-row lg:items-start lg:justify-between">

        <div>
          <h2 className="text-sm font-semibold text-ink">
            Retention Offer Performance
          </h2>

          <p className="mt-1 text-xs text-ink-muted">
            Generated, accepted, and dismissed offers
            with acceptance rate.
          </p>
        </div>

        {/* ======================================================
            SUMMARY METRICS
            ====================================================== */}

        <div className="flex flex-wrap gap-2">

          {/* Generated */}
          <div className="rounded-md border border-border bg-panel px-3 py-2">
            <div className="text-[10px] uppercase tracking-wide text-ink-muted">
              Generated
            </div>

            <div className="mt-0.5 text-sm font-semibold text-ink">
              {totalGenerated.toLocaleString()}
            </div>
          </div>

          {/* Accepted */}
          <div className="rounded-md border border-border bg-panel px-3 py-2">
            <div className="text-[10px] uppercase tracking-wide text-ink-muted">
              Accepted
            </div>

            <div className="mt-0.5 text-sm font-semibold text-ink">
              {totalAccepted.toLocaleString()}
            </div>
          </div>

          {/* Dismissed */}
          <div className="rounded-md border border-border bg-panel px-3 py-2">
            <div className="text-[10px] uppercase tracking-wide text-ink-muted">
              Dismissed
            </div>

            <div className="mt-0.5 text-sm font-semibold text-ink">
              {totalDismissed.toLocaleString()}
            </div>
          </div>

          {/* Acceptance Rate */}
          <div className="rounded-md border border-border bg-panel px-3 py-2">
            <div className="text-[10px] uppercase tracking-wide text-ink-muted">
              Acceptance Rate
            </div>

            <div className="mt-0.5 text-sm font-semibold text-ink">
              {overallAcceptanceRate.toFixed(1)}%
            </div>
          </div>

        </div>
      </div>

      {/* ========================================================
          CHART
          ======================================================== */}

      <div className="h-[340px] w-full">

        <ResponsiveContainer
          width="100%"
          height="100%"
        >

          <ComposedChart
            data={chartData}
            margin={{
              top: 10,
              right: 15,
              left: 0,
              bottom: 10,
            }}
          >

            {/* Grid */}
            <CartesianGrid
              strokeDasharray="3 3"
              stroke="currentColor"
              opacity={0.06}
              vertical={false}
            />

            {/* X Axis */}
            <XAxis
              dataKey="date"
              tickFormatter={formatDate}
              tick={{
                fontSize: 11,
              }}
              stroke="currentColor"
              opacity={0.45}
              tickLine={false}
              axisLine={false}
            />

            {/* Left Y Axis */}
            <YAxis
              yAxisId="count"
              allowDecimals={false}
              tick={{
                fontSize: 11,
              }}
              stroke="currentColor"
              opacity={0.45}
              tickLine={false}
              axisLine={false}
            />

            {/* Right Y Axis */}
            <YAxis
              yAxisId="rate"
              orientation="right"
              domain={[0, 100]}
              tickFormatter={(value) =>
                `${value}%`
              }
              tick={{
                fontSize: 11,
              }}
              stroke="currentColor"
              opacity={0.45}
              tickLine={false}
              axisLine={false}
            />

            {/* Tooltip */}
            <Tooltip
              content={<CustomTooltip />}
              cursor={{
                opacity: 0.08,
              }}
            />

            {/* Legend */}
            <Legend
              verticalAlign="top"
              align="left"
              height={35}
              iconType="circle"
              wrapperStyle={{
                fontSize: "11px",
              }}
            />

            {/* ==================================================
                ACCEPTED
                ================================================== */}

            <Bar
              yAxisId="count"
              dataKey="accepted"
              name="Accepted"
              stackId="offers"
              fill="#34d399"
              radius={[
                0,
                0,
                0,
                0,
              ]}
              maxBarSize={42}
            />

            {/* ==================================================
                DISMISSED
                ================================================== */}

            <Bar
              yAxisId="count"
              dataKey="dismissed"
              name="Dismissed"
              stackId="offers"
              fill="#fb7185"
              radius={[
                4,
                4,
                0,
                0,
              ]}
              maxBarSize={42}
            />

            {/* ==================================================
                GENERATED
                ================================================== */}

            <Line
              yAxisId="count"
              type="monotone"
              dataKey="generated"
              name="Generated"
              stroke="#22d3ee"
              strokeWidth={2.5}
              dot={{
                r: 3,
                strokeWidth: 2,
                fill: "#22d3ee",
              }}
              activeDot={{
                r: 5,
              }}
            />

            {/* ==================================================
                ACCEPTANCE RATE
                ================================================== */}

            <Line
              yAxisId="rate"
              type="monotone"
              dataKey="acceptanceRate"
              name="Acceptance Rate"
              stroke="#fbbf24"
              strokeWidth={2}
              strokeDasharray="5 4"
              dot={{
                r: 2.5,
                strokeWidth: 2,
                fill: "#fbbf24",
              }}
              activeDot={{
                r: 5,
              }}
            />

          </ComposedChart>

        </ResponsiveContainer>

      </div>

      {/* ========================================================
          FOOTNOTE
          ======================================================== */}

      <div className="mt-2 flex items-center justify-between text-[10px] text-ink-muted">

        <span>
          Acceptance rate is calculated from accepted
          and dismissed offers.
        </span>

        <span>
          {chartData.length}{" "}
          {chartData.length === 1
            ? "day"
            : "days"}
        </span>

      </div>

    </div>
  );
}