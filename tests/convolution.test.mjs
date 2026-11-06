import test from 'node:test';
import assert from 'node:assert/strict';
import { boxKernel, convolve, convolveSeparable, gaussianKernel } from '../dist/esm/index.js';

const lcg = (seed) => {
  let s = seed;
  return () => (s = (Math.imul(s, 1664525) + 1013904223) >>> 0) / 2 ** 32;
};
const randomImage = (width, height, seed, translucent = true) => {
  const rand = lcg(seed);
  const data = new Uint8ClampedArray(width * height * 4);
  for (let i = 0; i < data.length; i++)
    data[i] = i % 4 === 3 && !translucent ? 255 : Math.floor(rand() * 256);
  return { width, height, data };
};

// Naive reference: explicit neighbour offsets, per-channel pixel lookup function.
function naive(
  img,
  kw,
  kh,
  k,
  { border = 'clamp', alpha = 'preserve', divisor = 1, bias = 0 } = {},
) {
  const px = (x, y, c) => {
    if (border === 'clamp') {
      x = Math.min(img.width - 1, Math.max(0, x));
      y = Math.min(img.height - 1, Math.max(0, y));
    } else if (x < 0 || y < 0 || x >= img.width || y >= img.height) return 0;
    const o = (y * img.width + x) * 4;
    return c < 3 && alpha === 'premultiplied'
      ? (img.data[o + c] * img.data[o + 3]) / 255
      : img.data[o + c];
  };
  const out = [];
  for (let y = 0; y < img.height; y++) {
    for (let x = 0; x < img.width; x++) {
      const v = [0, 0, 0, 0];
      for (let c = 0; c < 4; c++) {
        for (let dy = -(kh - 1) / 2; dy <= (kh - 1) / 2; dy++) {
          for (let dx = -(kw - 1) / 2; dx <= (kw - 1) / 2; dx++) {
            v[c] += px(x + dx, y + dy, c) * k[((kh - 1) / 2 - dy) * kw + ((kw - 1) / 2 - dx)];
          }
        }
        v[c] = v[c] / divisor + bias;
      }
      const clamp = (n) => Math.min(255, Math.max(0, Math.round(n)));
      if (alpha === 'premultiplied') {
        const a = Math.min(255, Math.max(0, v[3]));
        out.push(...[0, 1, 2].map((c) => (a > 0 ? clamp((v[c] * 255) / a) : 0)), clamp(a));
      } else {
        out.push(
          clamp(v[0]),
          clamp(v[1]),
          clamp(v[2]),
          alpha === 'preserve' ? img.data[(y * img.width + x) * 4 + 3] : clamp(v[3]),
        );
      }
    }
  }
  return out;
}

test('rejects malformed images, kernels and options', () => {
  const img = randomImage(2, 2, 1);
  const k = { width: 1, height: 1, values: [1] };
  assert.throws(() => convolve(null, k), TypeError);
  assert.throws(() => convolve({ ...img, data: [0, 0, 0, 0] }, k), TypeError);
  assert.throws(() => convolve({ ...img, data: new Uint8Array(15) }, k), RangeError);
  for (const size of [{ width: 0 }, { width: -1 }, { height: 1.5 }])
    assert.throws(() => convolve({ ...img, ...size }, k), RangeError);
  assert.throws(() => convolve(img, null), TypeError);
  for (const bad of [{ width: 2 }, { width: 65 }, { height: 4 }, { width: Number.NaN }]) {
    assert.throws(() => convolve(img, { ...k, ...bad }), RangeError);
  }
  assert.throws(() => convolve(img, { ...k, values: [1, 2] }), RangeError);
  assert.throws(() => convolve(img, { ...k, values: [Number.NaN] }), RangeError);
  assert.throws(() => convolve(img, { ...k, values: new Int8Array(1) }), TypeError);
  assert.throws(() => convolve(img, { ...k, divisor: 0 }), RangeError);
  assert.throws(() => convolve(img, { ...k, bias: Infinity }), RangeError);
  assert.throws(() => convolve(img, k, { border: 'wrap' }), TypeError);
  assert.throws(() => convolve(img, k, { alpha: 'ignore' }), TypeError);
  assert.throws(() => convolve(img, k, { maxPixels: 3 }), RangeError);
  assert.throws(() => convolve(img, k, null), TypeError);
  assert.throws(() => convolveSeparable(img, [1, 1], [1]), RangeError);
  assert.throws(() => convolveSeparable(img, [1], [Infinity]), RangeError);
  assert.throws(() => gaussianKernel(0), RangeError);
  assert.throws(() => gaussianKernel(1, 40), RangeError);
  assert.throws(() => boxKernel(4), RangeError);
});

test('identity kernel and manual cases', () => {
  const img = randomImage(5, 4, 2);
  assert.deepEqual(
    Array.from(convolve(img, { width: 1, height: 1, values: [1] }).data),
    Array.from(img.data),
  );
  const row = {
    width: 3,
    height: 1,
    data: Uint8ClampedArray.from([0, 0, 0, 255, 90, 90, 90, 255, 0, 0, 0, 255]),
  };
  const blur = convolve(
    row,
    { width: 3, height: 1, values: [1, 1, 1], divisor: 3 },
    { border: 'zero' },
  );
  assert.deepEqual(Array.from(blur.data), [30, 30, 30, 255, 30, 30, 30, 255, 30, 30, 30, 255]);
  const clamped = convolve(
    row,
    { width: 3, height: 1, values: [1, 1, 1], divisor: 3 },
    { border: 'clamp' },
  );
  assert.deepEqual(
    Array.from(clamped.data).filter((_, i) => i % 4 === 0),
    [30, 30, 30],
  );
});

test('kernel is flipped: a weight at index 0 moves an impulse to a lower x', () => {
  const img = { width: 5, height: 1, data: new Uint8ClampedArray(20) };
  img.data.set([100, 100, 100, 255], 8);
  const out = convolve(img, { width: 3, height: 1, values: [1, 0, 0] }, { border: 'zero' });
  assert.equal(out.data[4], 100);
  assert.equal(out.data[8], 0);
});

test('matches naive reference for all border and alpha policies', () => {
  const img = randomImage(7, 6, 3);
  const kernels = [
    { kw: 3, kh: 3, k: [1, 2, 1, 0, 1, 0, -1, -2, -1], divisor: 2, bias: 3 },
    { kw: 5, kh: 3, k: Array.from({ length: 15 }, (_, i) => (i % 4) - 1), divisor: 4, bias: 0 },
    { kw: 1, kh: 5, k: [1, -2, 4, -2, 1], divisor: 1, bias: 128 },
  ];
  for (const border of ['clamp', 'zero']) {
    for (const alpha of ['preserve', 'straight', 'premultiplied']) {
      for (const { kw, kh, k, divisor, bias } of kernels) {
        const actual = convolve(
          img,
          { width: kw, height: kh, values: k, divisor, bias },
          { border, alpha },
        );
        assert.deepEqual(
          Array.from(actual.data),
          naive(img, kw, kh, k, { border, alpha, divisor, bias }),
          `${border}/${alpha}/${kw}x${kh}`,
        );
      }
    }
  }
});

test('alpha policies are observably different for translucent pixels', () => {
  const img = { width: 2, height: 1, data: Uint8ClampedArray.from([255, 0, 0, 255, 0, 255, 0, 0]) };
  const k = { width: 3, height: 1, values: [1, 1, 1], divisor: 3 };
  const preserve = convolve(img, k, { border: 'clamp', alpha: 'preserve' });
  const straight = convolve(img, k, { border: 'clamp', alpha: 'straight' });
  const premul = convolve(img, k, { border: 'clamp', alpha: 'premultiplied' });
  assert.equal(preserve.data[7], 0);
  assert.equal(straight.data[7], 85);
  assert.deepEqual(Array.from(preserve.data.slice(4, 7)), [85, 170, 0]);
  assert.deepEqual(Array.from(premul.data.slice(4, 8)), [255, 0, 0, 85]);
});

test('separable convolution equals the 2-D outer-product kernel', () => {
  const img = randomImage(9, 8, 4);
  const h = [1, 2, 1];
  const v = [1, 0, -1, 2, 1];
  const full = v.flatMap((b) => h.map((a) => a * b));
  for (const border of ['clamp', 'zero']) {
    for (const alpha of ['preserve', 'straight', 'premultiplied']) {
      const a = convolveSeparable(img, h, v, { border, alpha, divisor: 4, bias: 2 });
      const b = convolve(
        img,
        { width: 3, height: 5, values: full, divisor: 4, bias: 2 },
        { border, alpha },
      );
      assert.deepEqual(Array.from(a.data), Array.from(b.data));
    }
  }
});

test('helper kernels and ownership', () => {
  assert.deepEqual(boxKernel(3), [1, 1, 1]);
  const g = gaussianKernel(1.5);
  assert.equal(g.length, 2 * Math.ceil(4.5) + 1);
  assert.ok(Math.abs(g.reduce((a, b) => a + b, 0) - 1) < 1e-12);
  assert.ok(g[4] < g[5]);
  assert.deepEqual(gaussianKernel(1, 0), [1]);
  const img = randomImage(3, 3, 8);
  const before = Array.from(img.data);
  const out = convolve(img, {
    width: 3,
    height: 3,
    values: boxKernel(3).flatMap(() => boxKernel(3)),
    divisor: 9,
  });
  assert.ok(out.data instanceof Uint8ClampedArray);
  assert.notEqual(out.data.buffer, img.data.buffer);
  assert.deepEqual(Array.from(img.data), before);
});
