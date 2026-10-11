/**
 * The dashboard's only route to the backend.
 *
 * Kept out of the component so it can be tested without rendering React, and so
 * there is exactly one place that knows the URL shape.
 */

// Never hardcode the backend origin: the dashboard is served from a different
// host in every real deployment. Falls back to the Vite dev proxy.
export const API_URL = import.meta.env.VITE_API_URL ?? '/api';

/**
 * A failed request, carrying the HTTP status so callers can distinguish
 * "you sent me garbage" (400) from "too big" (413) from "server is down".
 */
export class ApiError extends Error {
  constructor(message, status) {
    super(message);
    this.name = 'ApiError';
    this.status = status;
  }
}

/**
 * POST one frame and return the parsed multi-task result.
 *
 * @param {File|Blob} file the image to analyse
 * @param {object} [opts]
 * @param {AbortSignal} [opts.signal] aborts the request; streaming callers pass
 *   a fresh controller per frame so a stale frame cannot land after a newer one.
 * @param {string} [opts.apiUrl] override, mainly for tests
 * @returns {Promise<object>} the `/predict` response body
 * @throws {ApiError} on any non-2xx response
 */
export async function predict(file, { signal, apiUrl = API_URL } = {}) {
  const formData = new FormData();
  formData.append('file', file);

  const res = await fetch(`${apiUrl}/predict`, {
    method: 'POST',
    body: formData,
    signal,
  });

  if (!res.ok) {
    // The API sends {"detail": "..."} on 400/413/422. Fall back to the status
    // line when the body is not JSON (a proxy error page, for instance).
    let detail = '';
    try {
      detail = (await res.json()).detail ?? '';
    } catch {
      /* non-JSON error body */
    }
    throw new ApiError(detail || `Backend returned ${res.status}`, res.status);
  }

  return res.json();
}

/**
 * Readiness probe. Surfaces the checkpoint the server actually loaded, which is
 * how you tell "no checkpoint" from "stale checkpoint" without reading logs.
 */
export async function health({ apiUrl = API_URL, signal } = {}) {
  const res = await fetch(`${apiUrl}/health`, { signal });
  if (!res.ok) {
    throw new ApiError(`Health check returned ${res.status}`, res.status);
  }
  return res.json();
}

/** Mirrors SEG_PALETTE in per_camera/classes.py, indexed by class_id. */
export const CLASS_COLORS = ['#ef4444', '#22c55e', '#3b82f6'];

export function classColor(classId) {
  return CLASS_COLORS[classId] ?? '#ef4444';
}
