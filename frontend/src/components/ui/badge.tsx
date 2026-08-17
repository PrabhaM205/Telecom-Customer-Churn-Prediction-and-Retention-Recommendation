import type { HTMLAttributes } from "react";

type BadgeVariant = "default" | "success" | "warning" | "danger" | "info" | "outline";

interface BadgeProps extends HTMLAttributes<HTMLSpanElement> {
  variant?: BadgeVariant;
}

const VARIANT_CLASSES: Record<BadgeVariant, string> = {
  default: "bg-panel-raised text-ink-muted border-border-subtle",
  success: "bg-accent-emerald/10 text-accent-emerald border-accent-emerald/30",
  warning: "bg-accent-amber/10 text-accent-amber border-accent-amber/30",
  danger: "bg-accent-rose/10 text-accent-rose border-accent-rose/30",
  info: "bg-accent-cyan/10 text-accent-cyan border-accent-cyan/30",
  outline: "bg-transparent text-ink-muted border-border",
};

export default function Badge({ variant = "default", className = "", ...props }: BadgeProps) {
  return (
    <span
      className={`inline-flex items-center gap-1 rounded-full border px-2 py-0.5 text-xs font-medium capitalize ${VARIANT_CLASSES[variant]} ${className}`}
      {...props}
    />
  );
}
