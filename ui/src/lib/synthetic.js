/**
 * Synthetic frames for the demo stream.
 *
 * These deliberately mirror `per_camera.data.generate_synthetic_dataset`: the
 * same three silhouettes, the same per-class hue families, 0-3 objects per
 * frame, nearer objects drawn larger and brighter. The previous version drew a
 * single random rectangle, which meant the demo stream fed the model frames
 * from a distribution it was never trained on — spheres and triangles never
 * appeared, so two thirds of the model's classes were unexercised by the demo.
 */

/** Must stay in sync with CLASS_NAMES in per_camera/classes.py. */
export const CLASS_NAMES = ['block', 'sphere', 'triangle'];

export const MAX_OBJECTS = 3;
export const P_EMPTY_FRAME = 0.2;

/** Base hue per class: red-ish, green-ish, blue-ish. Matches the backend. */
const CLASS_HUES = [8, 145, 217];

/**
 * Draw one frame onto `canvas` and return what was drawn.
 *
 * @param {HTMLCanvasElement} canvas
 * @param {object} [opts]
 * @param {() => number} [opts.rng] injectable so tests are deterministic
 * @param {number} [opts.size] square edge length in pixels
 * @returns {{objects: Array<{cls:number, name:string, x:number, y:number,
 *   w:number, h:number, depth:number}>}} the ground truth of the frame
 */
export function createSyntheticFrame(canvas, { rng = Math.random, size = 128 } = {}) {
  canvas.width = size;
  canvas.height = size;
  const ctx = canvas.getContext('2d');

  drawBackground(ctx, size);

  const objects = [];
  const n = rng() < P_EMPTY_FRAME ? 0 : 1 + Math.floor(rng() * MAX_OBJECTS);

  for (let i = 0; i < n; i++) {
    const cls = Math.floor(rng() * CLASS_NAMES.length);
    // Nearer objects are larger and brighter, so depth has a real cue.
    const depth = 0.05 + rng() * 0.45;
    const scale = 1.1 - 0.7 * depth;
    const w = Math.max(8, Math.round(size * 0.19 * scale * (0.85 + rng() * 0.3)));
    const h = Math.max(8, Math.round(size * 0.19 * scale * (0.85 + rng() * 0.3)));
    const x = Math.floor(rng() * (size - w));
    const y = Math.floor(rng() * (size - h));

    objects.push({ cls, name: CLASS_NAMES[cls], x, y, w, h, depth });
  }

  // Farthest first so nearer objects occlude them, as in the real generator.
  objects.sort((a, b) => b.depth - a.depth);
  for (const obj of objects) {
    drawObject(ctx, obj, size);
  }

  return { objects };
}

function drawBackground(ctx, size) {
  // A receding ground plane: dark at the horizon, lighter toward the camera.
  const grad = ctx.createLinearGradient(0, 0, 0, size);
  grad.addColorStop(0, '#101820');
  grad.addColorStop(1, '#2b3a4a');
  ctx.fillStyle = grad;
  ctx.fillRect(0, 0, size, size);
}

function drawObject(ctx, { cls, x, y, w, h, depth }, size) {
  const haze = 1 - 0.45 * depth;
  const sat = 70;
  const light = Math.round(55 * haze + 12);
  ctx.fillStyle = `hsl(${CLASS_HUES[cls]}, ${sat}%, ${light}%)`;
  ctx.strokeStyle = `hsl(${CLASS_HUES[cls]}, ${sat}%, ${Math.max(6, light - 18)}%)`;
  ctx.lineWidth = 1.5;

  ctx.beginPath();
  if (cls === 0) {
    // block: the full box
    ctx.rect(x, y, w, h);
  } else if (cls === 1) {
    // sphere: ellipse inscribed in the box
    ctx.ellipse(x + w / 2, y + h / 2, w / 2, h / 2, 0, 0, Math.PI * 2);
  } else {
    // triangle: apex at top-centre, base across the bottom
    ctx.moveTo(x + w / 2, y);
    ctx.lineTo(x + w, y + h);
    ctx.lineTo(x, y + h);
    ctx.closePath();
  }
  ctx.fill();
  ctx.stroke();

  // Keep `size` referenced for parity with the backend's normalised coords.
  void size;
}

/**
 * Encode the canvas as a PNG Blob.
 *
 * @returns {Promise<Blob|null>} resolves to `null` if the canvas cannot be
 *   encoded — callers must handle that rather than posting a bogus frame.
 */
export function canvasToBlob(canvas) {
  return new Promise((resolve) => {
    canvas.toBlob((blob) => resolve(blob), 'image/png');
  });
}
