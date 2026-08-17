// frontend/src/components/kpi/KpiCard.tsx

import type { ReactNode } from "react";

type Tone = "cyan" | "amber" | "emerald" | "rose" | "violet" | "neutral";

interface KpiCardProps {
  label: string;
  value: string;
  sublabel?: string;
  accentClassName?: string; // legacy override, e.g. "text-risk-high"
  tone?: Tone;
  icon?: ReactNode;
}

const TONE_TEXT: Record<Tone, string> = {
  cyan: "text-accent-cyan",
  amber: "text-accent-amber",
  emerald: "text-accent-emerald",
  rose: "text-accent-rose",
  violet: "text-accent-violet",
  neutral: "text-ink",
};

export default function KpiCard({ label, value, sublabel, accentClassName, tone = "neutral", icon }: KpiCardProps) {
  return (
    <div className="rounded-lg border border-border bg-panel p-5 transition-colors hover:border-border-subtle">
      <div className="flex items-center justify-between">
        <div className="eyebrow text-[10px]">{label}</div>
        {icon && <div className={`${TONE_TEXT[tone]} opacity-80`}>{icon}</div>}
      </div>
      <div className={`mt-2 text-3xl font-semibold ${accentClassName ?? TONE_TEXT[tone]}`}>{value}</div>
      {sublabel && <div className="mt-1 text-xs text-ink-faint">{sublabel}</div>}
    </div>
  );
}
