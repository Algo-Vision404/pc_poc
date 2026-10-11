import { describe, it, expect, vi, beforeEach } from 'vitest';
import { predict, health, classColor, ApiError, API_URL } from './api.js';

function jsonResponse(body, status = 200) {
  return {
    ok: status >= 200 && status < 300,
    status,
    json: async () => body,
  };
}

function pngFile() {
  return new File(['not-really-a-png'], 'f.png', { type: 'image/png' });
}

describe('predict', () => {
  beforeEach(() => {
    vi.stubGlobal('fetch', vi.fn());
  });

  it('POSTs the frame as multipart to /predict', async () => {
    fetch.mockResolvedValue(jsonResponse({ detections: [] }));

    await predict(pngFile());

    const [url, init] = fetch.mock.calls[0];
    expect(url).toBe(`${API_URL}/predict`);
    expect(init.method).toBe('POST');
    expect(init.body).toBeInstanceOf(FormData);
    expect(init.body.get('file')).toBeInstanceOf(File);
  });

  it('honours an apiUrl override', async () => {
    fetch.mockResolvedValue(jsonResponse({ detections: [] }));

    await predict(pngFile(), { apiUrl: 'http://example.test/api' });

    expect(fetch.mock.calls[0][0]).toBe('http://example.test/api/predict');
  });

  it('returns the parsed body', async () => {
    const body = { detections: [{ class: 'block', score: 0.9 }], score: 0.9 };
    fetch.mockResolvedValue(jsonResponse(body));

    await expect(predict(pngFile())).resolves.toEqual(body);
  });

  it('passes the abort signal through to fetch', async () => {
    fetch.mockResolvedValue(jsonResponse({ detections: [] }));
    const controller = new AbortController();

    await predict(pngFile(), { signal: controller.signal });

    expect(fetch.mock.calls[0][1].signal).toBe(controller.signal);
  });

  it('surfaces the API detail on a 400', async () => {
    fetch.mockResolvedValue(jsonResponse({ detail: 'Unsupported image format' }, 400));

    const err = await predict(pngFile()).catch((e) => e);

    expect(err).toBeInstanceOf(ApiError);
    expect(err.status).toBe(400);
    expect(err.message).toBe('Unsupported image format');
  });

  it('surfaces 413 for an oversized upload', async () => {
    fetch.mockResolvedValue(jsonResponse({ detail: 'File exceeds 10 MB' }, 413));

    const err = await predict(pngFile()).catch((e) => e);

    expect(err.status).toBe(413);
  });

  it('falls back to the status line when the error body is not JSON', async () => {
    // A proxy error page, not the API's {"detail": ...}.
    fetch.mockResolvedValue({
      ok: false,
      status: 502,
      json: async () => {
        throw new SyntaxError('Unexpected token <');
      },
    });

    const err = await predict(pngFile()).catch((e) => e);

    expect(err.status).toBe(502);
    expect(err.message).toBe('Backend returned 502');
  });

  it('falls back to the status line when detail is absent', async () => {
    fetch.mockResolvedValue(jsonResponse({}, 500));

    const err = await predict(pngFile()).catch((e) => e);

    expect(err.message).toBe('Backend returned 500');
  });

  it('lets a network failure propagate untouched', async () => {
    fetch.mockRejectedValue(new TypeError('Failed to fetch'));

    await expect(predict(pngFile())).rejects.toThrow('Failed to fetch');
  });

  it('propagates an abort rather than wrapping it', async () => {
    // App.jsx checks err.name === 'AbortError' to stay quiet on superseded
    // frames; wrapping it would turn every stream tick into a visible error.
    const abort = new DOMException('The operation was aborted.', 'AbortError');
    fetch.mockRejectedValue(abort);

    const err = await predict(pngFile()).catch((e) => e);

    expect(err.name).toBe('AbortError');
    expect(err).not.toBeInstanceOf(ApiError);
  });
});

describe('health', () => {
  beforeEach(() => {
    vi.stubGlobal('fetch', vi.fn());
  });

  it('reports the loaded checkpoint', async () => {
    fetch.mockResolvedValue(
      jsonResponse({ arch_version: 'big-per-camera-net-v3', checkpoint_epoch: 19 })
    );

    const info = await health();

    expect(fetch.mock.calls[0][0]).toBe(`${API_URL}/health`);
    expect(info.checkpoint_epoch).toBe(19);
  });

  it('throws an ApiError when the probe fails', async () => {
    fetch.mockResolvedValue(jsonResponse({}, 503));

    const err = await health().catch((e) => e);

    expect(err).toBeInstanceOf(ApiError);
    expect(err.status).toBe(503);
  });
});

describe('classColor', () => {
  it('maps each class to its palette colour', () => {
    expect(classColor(0)).toBe('#ef4444');
    expect(classColor(1)).toBe('#22c55e');
    expect(classColor(2)).toBe('#3b82f6');
  });

  it('falls back instead of returning undefined for an unknown class', () => {
    // A checkpoint trained with more classes must still render, not draw
    // `border: 2px solid undefined`.
    expect(classColor(7)).toBe('#ef4444');
    expect(classColor(undefined)).toBe('#ef4444');
  });
});
