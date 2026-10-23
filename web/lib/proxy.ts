const TIMEOUT = 60_000;
export function apiError(status: number, code: string, message: string) {
  return Response.json({error: {code, message}}, {status});
}
export async function proxy(path: 'health' | 'translate', request?: Request) {
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), TIMEOUT);
  try {
    const base = (process.env.ATTENTION_API_URL || 'http://127.0.0.1:8000').replace(/\/$/, '');
    const response = await fetch(`${base}/api/${path}`, {method: request ? 'POST' : 'GET', headers: {'Content-Type': 'application/json'}, body: request ? await request.text() : undefined, signal: controller.signal, cache: 'no-store'});
    return new Response(await response.text(), {status: response.status, headers: {'Content-Type': 'application/json', 'Cache-Control': 'no-store'}});
  } catch {
    return controller.signal.aborted
      ? apiError(504, 'BACKEND_TIMEOUT', 'Le modèle met trop de temps à répondre. Réessaie dans un instant.')
      : apiError(503, 'BACKEND_UNAVAILABLE', 'Le service de traduction est injoignable. Vérifie que l’API Python est démarrée.');
  } finally { clearTimeout(timeout); }
}
