import { useState, type FormEvent } from "react";

interface ChatInputProps {
  onSubmit: (customerId: string) => void;
  disabled?: boolean;
}

export default function ChatInput({
  onSubmit,
  disabled,
}: ChatInputProps) {
  const [value, setValue] = useState("");

  const handleSubmit = (e: FormEvent) => {
    e.preventDefault();

    const trimmed = value.trim();

    if (!trimmed || disabled) return;

    onSubmit(trimmed);
    setValue("");
  };

  return (
    <form
      onSubmit={handleSubmit}
      className="flex gap-2 border-t border-gray-200 p-4"
    >
      <input
        type="text"
        value={value}
        onChange={(e) => setValue(e.target.value)}
        placeholder="Enter a customer ID (e.g. 1771-OADNZ)..."
        disabled={disabled}
        className="flex-1 rounded-md border border-gray-200 px-3 py-2 text-sm text-black placeholder:text-black focus:outline-none focus:ring-2 focus:ring-gray-300 disabled:opacity-50"
      />

      <button
        type="submit"
        disabled={disabled}
        className="rounded-md bg-gray-900 px-4 py-2 text-sm font-medium text-white hover:bg-gray-700 disabled:opacity-50"
      >
        Analyze
      </button>
    </form>
  );
}