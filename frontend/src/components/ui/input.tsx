import { forwardRef, type InputHTMLAttributes } from "react";

const Input = forwardRef<HTMLInputElement, InputHTMLAttributes<HTMLInputElement>>(
  ({ className = "", ...props }, ref) => (
    <input
      ref={ref}
      className={`w-full rounded-md border border-border bg-panel-raised px-3 py-2 text-sm text-ink placeholder:text-ink-faint outline-none transition-colors focus:border-accent-cyan/60 focus:ring-1 focus:ring-accent-cyan/30 ${className}`}
      {...props}
    />
  )
);
Input.displayName = "Input";

export default Input;
