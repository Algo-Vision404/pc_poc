import { describe, it, expect } from 'vitest';
import { createSyntheticFrame, CLASS_NAMES, MAX_OBJECTS } from './synthetic.js';

/** A deterministic PRNG so "random" frames are assertable. */
function makeRng(values) {
  let i = 0;
  return () => values[i++ % values.length];
}

/** Sequence-based rng: consumes values in order, cycling at the end. */
function seq(...values) {
  return makeRng(values);
}

function newCanvas() {
  return document.createElement('canvas');
}

describe('createSyntheticFrame', () => {
  it('sizes the canvas to the requested square', () => {
    const canvas = newCanvas();
    createSyntheticFrame(canvas, { rng: seq(0.9), size: 128 });

    expect(canvas.width).toBe(128);
    expect(canvas.height).toBe(128);
  });

  it('draws a background before any object', () => {
    const canvas = newCanvas();
    createSyntheticFrame(canvas, { rng: seq(0.99), size: 128 });
    const ctx = canvas.getContext('2d');

    const first = ctx.calls[0];
    expect(['createLinearGradient', 'fillRect']).toContain(first.name);
  });

  it('emits an empty frame when the rng lands under P_EMPTY_FRAME', () => {
    const canvas = newCanvas();
    // First draw decides emptiness: 0.1 < 0.2 means no objects.
    const { objects } = createSyntheticFrame(canvas, { rng: seq(0.1) });

    expect(objects).toHaveLength(0);
    const ctx = canvas.getContext('2d');
    expect(ctx.callsOf('fill')).toHaveLength(0);
  });

  it('draws a block as a rect covering the whole box', () => {
    const canvas = newCanvas();
    // 8 draws make exactly one object: 0.9 skips the empty branch, 0.1 gives
    // count 1, the next 0.1 selects class 0.
    const { objects } = createSyntheticFrame(canvas, {
      rng: seq(0.9, 0.1, 0.1, 0.5, 0.5, 0.5, 0.5, 0.5),
      size: 128,
    });
    const ctx = canvas.getContext('2d');

    expect(objects).toHaveLength(1);
    expect(objects[0].name).toBe('block');

    const rects = ctx.callsOf('rect');
    expect(rects).toHaveLength(1);
    const [x, y, w, h] = rects[0].args;
    expect({ x, y, w, h }).toMatchObject({
      x: objects[0].x,
      y: objects[0].y,
      w: objects[0].w,
      h: objects[0].h,
    });
    expect(ctx.callsOf('ellipse')).toHaveLength(0);
  });

  it('draws a sphere as an ellipse inscribed in its box', () => {
    const canvas = newCanvas();
    // 0.1 gives count 1, then 0.5 -> floor(0.5*3) = 1 -> sphere
    const { objects } = createSyntheticFrame(canvas, {
      rng: seq(0.9, 0.1, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5),
      size: 128,
    });
    const ctx = canvas.getContext('2d');

    expect(objects[0].name).toBe('sphere');
    const ellipses = ctx.callsOf('ellipse');
    expect(ellipses).toHaveLength(1);

    const [cx, cy, rx, ry] = ellipses[0].args;
    const o = objects[0];
    expect(cx).toBeCloseTo(o.x + o.w / 2, 5);
    expect(cy).toBeCloseTo(o.y + o.h / 2, 5);
    expect(rx).toBeCloseTo(o.w / 2, 5);
    expect(ry).toBeCloseTo(o.h / 2, 5);
    expect(ctx.callsOf('rect')).toHaveLength(0);
  });

  it('draws a triangle with the apex at top-centre and a flat base', () => {
    const canvas = newCanvas();
    // 0.1 gives count 1, then 0.9 -> floor(0.9*3) = 2 -> triangle
    const { objects } = createSyntheticFrame(canvas, {
      rng: seq(0.9, 0.1, 0.9, 0.5, 0.5, 0.5, 0.5, 0.5),
      size: 128,
    });
    const ctx = canvas.getContext('2d');

    expect(objects[0].name).toBe('triangle');
    expect(ctx.callsOf('closePath')).toHaveLength(1);

    const apex = ctx.callsOf('moveTo')[0].args;
    const o = objects[0];
    expect(apex[0]).toBeCloseTo(o.x + o.w / 2, 5);
    expect(apex[1]).toBe(o.y);

    const base = ctx.callsOf('lineTo').map((c) => c.args);
    expect(base).toHaveLength(2);
    // Both base corners sit on the bottom edge.
    expect(base[0][1]).toBe(o.y + o.h);
    expect(base[1][1]).toBe(o.y + o.h);
    expect(Math.min(base[0][0], base[1][0])).toBe(o.x);
    expect(Math.max(base[0][0], base[1][0])).toBe(o.x + o.w);
  });

  it('never draws outside the canvas', () => {
    for (let seed = 0; seed < 200; seed++) {
      const canvas = newCanvas();
      // A cheap deterministic generator over [0,1).
      let s = seed + 1;
      const rng = () => {
        s = (s * 1103515245 + 12345) % 2147483648;
        return s / 2147483648;
      };
      const { objects } = createSyntheticFrame(canvas, { rng, size: 128 });

      for (const o of objects) {
        expect(o.x).toBeGreaterThanOrEqual(0);
        expect(o.y).toBeGreaterThanOrEqual(0);
        expect(o.x + o.w).toBeLessThanOrEqual(128);
        expect(o.y + o.h).toBeLessThanOrEqual(128);
        expect(o.w).toBeGreaterThanOrEqual(8);
      }
    }
  });

  it('emits between 0 and MAX_OBJECTS and returns matching ground truth', () => {
    const counts = new Set();
    for (let seed = 0; seed < 300; seed++) {
      let s = seed + 7;
      const rng = () => {
        s = (s * 1103515245 + 12345) % 2147483648;
        return s / 2147483648;
      };
      const { objects } = createSyntheticFrame(newCanvas(), { rng });
      expect(objects.length).toBeLessThanOrEqual(MAX_OBJECTS);
      counts.add(objects.length);
    }

    // Empty frames must actually occur, or the detector never sees a negative.
    expect(counts.has(0)).toBe(true);
  });

  it('exercises all three classes over many frames', () => {
    const seen = new Set();
    for (let seed = 0; seed < 300; seed++) {
      let s = seed + 13;
      const rng = () => {
        s = (s * 1103515245 + 12345) % 2147483648;
        return s / 2147483648;
      };
      for (const o of createSyntheticFrame(newCanvas(), { rng }).objects) {
        seen.add(o.name);
      }
    }

    expect(seen).toEqual(new Set(CLASS_NAMES));
  });

  it('paints farthest-first so nearer objects occlude them', () => {
    const canvas = newCanvas();
    let s = 42;
    const rng = () => {
      s = (s * 1103515245 + 12345) % 2147483648;
      return s / 2147483648;
    };

    const { objects } = createSyntheticFrame(canvas, { rng });
    for (let i = 1; i < objects.length; i++) {
      expect(objects[i].depth).toBeLessThanOrEqual(objects[i - 1].depth);
    }
  });
});
