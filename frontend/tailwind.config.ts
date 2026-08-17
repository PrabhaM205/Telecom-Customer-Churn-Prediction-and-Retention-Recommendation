import type { Config } from "tailwindcss";

export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        canvas: "#0A0E1A",
        panel: "#0F1524",
        "panel-raised": "#141B2E",
        border: {
          DEFAULT: "#1E293B",
          subtle: "#182234",
        },
        ink: {
          DEFAULT: "#E2E8F0",
          muted: "#64748B",
          faint: "#3E4C63",
        },
        accent: {
          cyan: "#22D3EE",
          amber: "#F5B942",
          emerald: "#34D399",
          rose: "#FB7185",
          violet: "#A78BFA",
        },
        risk: {
          low: "#34D399",
          medium: "#F5B942",
          high: "#FB7185",
        },
      },
      fontFamily: {
        mono: ['"IBM Plex Mono"', "ui-monospace", "SFMono-Regular", "monospace"],
        sans: ['"Inter"', "ui-sans-serif", "system-ui", "sans-serif"],
      },
      boxShadow: {
        glow: "0 0 0 1px rgba(34,211,238,0.15), 0 0 24px rgba(34,211,238,0.08)",
      },
    },
  },
  plugins: [],
} satisfies Config;