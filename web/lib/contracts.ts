export type Direction = "to_genz" | "to_standard";
export type Translation = {
  translation: string;
  direction: Direction;
  truncated: boolean;
  model_id: string;
};
export type Health = {
  status: "ready" | "unavailable";
  model_id: string | null;
  limits: {
    max_input_chars: number;
    max_source_tokens: number;
    max_new_tokens: number;
  };
};
export const characterCount = (text: string) => Array.from(text).length;
export function validateInput(text: string, limit = 500) {
  if (!text.trim()) return "Écris une phrase avant de la traduire.";
  if (characterCount(text.trim()) > limit)
    return `Ta phrase dépasse la limite de ${limit} caractères.`;
  return null;
}
export function isTranslation(value: unknown): value is Translation {
  if (!value || typeof value !== "object") return false;
  const v = value as Record<string, unknown>;
  return (
    typeof v.translation === "string" &&
    (v.direction === "to_genz" || v.direction === "to_standard") &&
    typeof v.truncated === "boolean" &&
    typeof v.model_id === "string"
  );
}
