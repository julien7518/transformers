import { afterEach, expect, it, vi } from "vitest";
import { proxy } from "@/lib/proxy";
import { mockTranslate } from "@/lib/mock";
afterEach(() => {
  vi.unstubAllGlobals();
  vi.unstubAllEnvs();
  vi.useRealTimers();
});
it("relaie le corps, la direction et le statut backend sans les modifier", async () => {
  vi.stubEnv("ATTENTION_API_URL", "http://localhost:9876/");
  const body = {
    error: { code: "INPUT_TOO_LONG", message: "Limite tokens dépassée." },
  };
  const mock = vi.fn().mockResolvedValue(Response.json(body, { status: 400 }));
  vi.stubGlobal("fetch", mock);
  const response = await proxy(
    "translate",
    new Request("http://local/api/translate", {
      method: "POST",
      body: JSON.stringify({ text: "Bonjour", direction: "to_standard" }),
    }),
  );
  expect(response.status).toBe(400);
  expect(await response.json()).toEqual(body);
  expect(mock.mock.calls[0][0]).toBe("http://localhost:9876/api/translate");
  expect(JSON.parse(mock.mock.calls[0][1].body).direction).toBe("to_standard");
});
it("retourne 503 si le backend est arrêté", async () => {
  vi.stubGlobal("fetch", vi.fn().mockRejectedValue(new Error("offline")));
  const response = await proxy("health");
  expect(response.status).toBe(503);
  expect((await response.json()).error.code).toBe("BACKEND_UNAVAILABLE");
});
it("retourne 504 après 60 secondes", async () => {
  vi.useFakeTimers();
  vi.stubGlobal(
    "fetch",
    vi.fn(
      (_url, { signal }) =>
        new Promise((_resolve, reject) =>
          signal.addEventListener("abort", () => reject(new Error("abort"))),
        ),
    ),
  );
  const pending = proxy("health");
  await vi.advanceTimersByTimeAsync(60_000);
  const response = await pending;
  expect(response.status).toBe(504);
  expect((await response.json()).error.code).toBe("BACKEND_TIMEOUT");
});
it("les fixtures ne traduisent que leurs exemples et valident les requêtes", async () => {
  const req = (value: unknown) =>
    new Request("http://local", {
      method: "POST",
      body: JSON.stringify(value),
    });
  expect(
    (await mockTranslate(req({ text: "", direction: "to_genz" }))).status,
  ).toBe(400);
  expect(
    (await mockTranslate(req({ text: "ok", direction: "unknown" }))).status,
  ).toBe(422);
  const response = await mockTranslate(
    req({ text: "Ce film est vraiment excellent.", direction: "to_genz" }),
  );
  expect((await response.json()).model_id).toBe("fixture-demo");
});
