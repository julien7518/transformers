import { apiError } from "./proxy";
import { characterCount } from "./contracts";
export const fixtures = [
  {
    standard: "Ce film est vraiment excellent.",
    genz: "Ce film est une dinguerie.",
  },
  { standard: "Je suis vraiment déçu.", genz: "J’ai trop le seum." },
  {
    standard: "Je suis très motivé pour ce projet.",
    genz: "Je suis grave chaud pour ce projet.",
  },
];
export async function mockTranslate(request: Request) {
  let value;
  try {
    value = await request.json();
  } catch {
    return apiError(422, "INVALID_REQUEST", "La requête JSON est invalide.");
  }
  if (
    !value ||
    typeof value.text !== "string" ||
    !["to_genz", "to_standard"].includes(value.direction)
  )
    return apiError(
      422,
      "INVALID_REQUEST",
      "Le texte ou la direction est invalide.",
    );
  if (!value.text.trim())
    return apiError(
      400,
      "EMPTY_TEXT",
      "Écris une phrase avant de la traduire.",
    );
  if (characterCount(value.text.trim()) > 500)
    return apiError(400, "INPUT_TOO_LONG", "La limite est de 500 caractères.");
  const pair = fixtures.find(
    (p) =>
      p[value.direction === "to_genz" ? "standard" : "genz"] ===
      value.text.trim(),
  );
  if (!pair)
    return apiError(
      400,
      "MOCK_EXAMPLE_ONLY",
      "Mode démonstration : choisis un des exemples proposés.",
    );
  return Response.json({
    translation: pair[value.direction === "to_genz" ? "genz" : "standard"],
    direction: value.direction,
    truncated: false,
    model_id: "fixture-demo",
  });
}
