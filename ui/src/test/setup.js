import '@testing-library/jest-dom/vitest';
import { vi } from 'vitest';

/**
 * jsdom ships no canvas backend, and `getContext('2d')` returns null while
 * logging "Not implemented". Rather than pull in the native `canvas` package
 * (a C dependency that would make `npm ci` need a compiler), stub the 2D
 * context with a recorder.
 *
 * The tests assert on *which drawing calls were made*, which is exactly what
 * matters here: that a sphere draws an ellipse, a triangle draws three lines,
 * and a block draws a rect. Pixel output would need a real rasteriser and
 * would test the browser, not this code.
 */
class RecordingContext2D {
  constructor(canvas) {
    this.canvas = canvas;
    this.calls = [];
    this.fillStyle = '';
    this.strokeStyle = '';
    this.lineWidth = 1;
  }

  _record(name, args) {
    this.calls.push({ name, args });
  }

  fillRect(...args) {
    this._record('fillRect', args);
  }
  strokeRect(...args) {
    this._record('strokeRect', args);
  }
  rect(...args) {
    this._record('rect', args);
  }
  ellipse(...args) {
    this._record('ellipse', args);
  }
  moveTo(...args) {
    this._record('moveTo', args);
  }
  lineTo(...args) {
    this._record('lineTo', args);
  }
  closePath(...args) {
    this._record('closePath', args);
  }
  beginPath(...args) {
    this._record('beginPath', args);
  }
  fill(...args) {
    this._record('fill', args);
  }
  stroke(...args) {
    this._record('stroke', args);
  }
  drawImage(...args) {
    this._record('drawImage', args);
  }
  createLinearGradient() {
    this._record('createLinearGradient', []);
    return { addColorStop: () => {} };
  }

  callsOf(name) {
    return this.calls.filter((c) => c.name === name);
  }
}

const contexts = new WeakMap();

HTMLCanvasElement.prototype.getContext = function getContext(kind) {
  if (kind !== '2d') return null;
  let ctx = contexts.get(this);
  if (!ctx) {
    ctx = new RecordingContext2D(this);
    contexts.set(this, ctx);
  }
  return ctx;
};

// jsdom implements neither. The dashboard uses both on the streaming path.
if (!HTMLCanvasElement.prototype.toBlob) {
  HTMLCanvasElement.prototype.toBlob = function toBlob(cb) {
    cb(new Blob(['fake-png'], { type: 'image/png' }));
  };
}
if (!HTMLCanvasElement.prototype.toDataURL) {
  HTMLCanvasElement.prototype.toDataURL = () => 'data:image/png;base64,AAAA';
}

// jsdom has no URL.createObjectURL, and App.jsx revokes the previous preview
// URL on every upload. Without these the upload path throws in tests.
if (!URL.createObjectURL) {
  let n = 0;
  const live = new Set();
  URL.createObjectURL = vi.fn(() => {
    const url = `blob:mock/${++n}`;
    live.add(url);
    return url;
  });
  URL.revokeObjectURL = vi.fn((url) => live.delete(url));
  URL.__liveBlobUrls = live;
}
