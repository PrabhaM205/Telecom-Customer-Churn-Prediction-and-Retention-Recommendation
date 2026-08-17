import { NavLink } from "react-router-dom";
import { useEffect, useState } from "react";

interface IconProps {
  className?: string;
}

function IconGrid({ className }: IconProps) {
  return (
    <svg viewBox="0 0 20 20" fill="none" className={className}>
      <rect x="2.5" y="2.5" width="6" height="6" rx="1" stroke="currentColor" strokeWidth="1.5" />
      <rect x="11.5" y="2.5" width="6" height="6" rx="1" stroke="currentColor" strokeWidth="1.5" />
      <rect x="2.5" y="11.5" width="6" height="6" rx="1" stroke="currentColor" strokeWidth="1.5" />
      <rect x="11.5" y="11.5" width="6" height="6" rx="1" stroke="currentColor" strokeWidth="1.5" />
    </svg>
  );
}

function IconUsers({ className }: IconProps) {
  return (
    <svg viewBox="0 0 20 20" fill="none" className={className}>
      <circle cx="7" cy="6.5" r="2.5" stroke="currentColor" strokeWidth="1.5" />
      <path d="M2.5 16c0-2.76 2.01-4.5 4.5-4.5s4.5 1.74 4.5 4.5" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" />
      <circle cx="14" cy="7" r="2" stroke="currentColor" strokeWidth="1.5" />
      <path d="M12.5 11.75c1.9.28 3.5 1.77 3.5 4.25" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" />
    </svg>
  );
}

function IconSpark({ className }: IconProps) {
  return (
    <svg viewBox="0 0 20 20" fill="none" className={className}>
      <path d="M10 2.5l1.4 4.6 4.6 1.4-4.6 1.4L10 14.5l-1.4-4.6-4.6-1.4 4.6-1.4L10 2.5z" stroke="currentColor" strokeWidth="1.3" strokeLinejoin="round" />
    </svg>
  );
}

function IconGauge({ className }: IconProps) {
  return (
    <svg viewBox="0 0 20 20" fill="none" className={className}>
      <path d="M3 13a7 7 0 1114 0" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" />
      <path d="M10 13l3.2-4.2" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" />
      <circle cx="10" cy="13" r="1" fill="currentColor" />
    </svg>
  );
}

function IconChat({ className }: IconProps) {
  return (
    <svg viewBox="0 0 20 20" fill="none" className={className}>
      <path
        d="M3 4.5h14a1 1 0 011 1v7a1 1 0 01-1 1H8l-3.5 3v-3H3a1 1 0 01-1-1v-7a1 1 0 011-1z"
        stroke="currentColor"
        strokeWidth="1.4"
        strokeLinejoin="round"
      />
    </svg>
  );
}

interface NavItem {
  to: string;
  label: string;
  icon: (props: IconProps) => JSX.Element;
}

const NAV_ITEMS: NavItem[] = [
  { to: "/", label: "Executive Dashboard", icon: IconGrid },
  { to: "/customer-risk", label: "Customer Risk", icon: IconUsers },
  { to: "/recommendations", label: "Retention Recommendations", icon: IconSpark },
  { to: "/model-performance", label: "Model Performance", icon: IconGauge },
  { to: "/assistant", label: "AI Retention Assistant", icon: IconChat },
];

type ApiState = "checking" | "online" | "offline";

function ApiStatus() {
  const [status, setStatus] = useState<ApiState>("checking");
  const base = import.meta.env.VITE_API_BASE_URL as string | undefined;

  useEffect(() => {
    if (!base) {
      setStatus("offline");
      return;
    }
    let cancelled = false;
    const check = () => {
      fetch(base)
        .then(() => !cancelled && setStatus("online"))
        .catch(() => !cancelled && setStatus("offline"));
    };
    check();
    const id = setInterval(check, 15000);
    return () => {
      cancelled = true;
      clearInterval(id);
    };
  }, [base]);

  const dotClass =
    status === "online"
      ? "bg-accent-emerald"
      : status === "offline"
        ? "bg-accent-rose"
        : "bg-accent-amber animate-pulse";

  const label =
    status === "online" ? "API connected" : status === "offline" ? "API unreachable" : "Checking API…";

  return (
    <div className="rounded-lg border border-border-subtle bg-panel-raised/40 px-3 py-2.5">
      <div className="flex items-center gap-2">
        <span className={`h-1.5 w-1.5 rounded-full ${dotClass}`} />
        <span className="eyebrow text-[10px]">{label}</span>
      </div>
      {base && <div className="mt-1 truncate font-mono text-[10px] text-ink-faint">{base}</div>}
    </div>
  );
}

export default function Sidebar() {
  return (
    <nav className="flex w-64 shrink-0 flex-col border-r border-border bg-panel">
      <div className="flex items-center gap-3 border-b border-border px-5 py-5">
        <div className="flex h-9 w-9 items-center justify-center rounded-lg bg-gradient-to-br from-accent-cyan to-accent-violet shadow-glow">
          <span className="font-mono text-sm font-bold text-canvas">C</span>
        </div>
        <div>
          <div className="font-mono text-sm font-semibold tracking-wide text-ink">Churn Console</div>
          <div className="eyebrow text-[10px]">Retention Intelligence</div>
        </div>
      </div>

      <ul className="flex-1 space-y-1 px-3 py-4">
        {NAV_ITEMS.map((item) => {
          const Icon = item.icon;
          return (
            <li key={item.to}>
              <NavLink
                to={item.to}
                end={item.to === "/"}
                className={({ isActive }) =>
                  `group relative flex items-center gap-3 rounded-md px-3 py-2.5 text-sm font-medium transition-colors ${
                    isActive ? "bg-panel-raised text-ink" : "text-ink-muted hover:bg-panel-raised/60 hover:text-ink"
                  }`
                }
              >
                {({ isActive }) => (
                  <>
                    <span
                      className={`absolute left-0 top-1/2 h-5 w-0.5 -translate-y-1/2 rounded-full ${
                        isActive ? "bg-accent-cyan" : "bg-transparent"
                      }`}
                    />
                    <Icon
                      className={`h-4 w-4 shrink-0 ${
                        isActive ? "text-accent-cyan" : "text-ink-faint group-hover:text-ink-muted"
                      }`}
                    />
                    {item.label}
                  </>
                )}
              </NavLink>
            </li>
          );
        })}
      </ul>

      <div className="space-y-3 border-t border-border p-3">
        <ApiStatus />
        <div className="px-1 font-mono text-[10px] text-ink-faint">v1.0.0 · churn-ml</div>
      </div>
    </nav>
  );
}
