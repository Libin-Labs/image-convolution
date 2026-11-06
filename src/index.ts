/** Interleaved 8-bit RGBA pixels, row-major, `data.length === width * height * 4`. */
export interface PixelImage {
  width: number;
  height: number;
  data: Uint8Array | Uint8ClampedArray;
}

export type KernelValues = readonly number[] | Float32Array | Float64Array;
export type BorderMode = 'clamp' | 'zero';
/**
 * preserve: RGB convolved as stored, alpha copied unchanged.
 * straight: R, G, B and A each convolved independently (colour of transparent pixels bleeds).
 * premultiplied: RGB weighted by alpha before convolution and divided by the result alpha afterwards.
 */
export type AlphaMode = 'preserve' | 'straight' | 'premultiplied';

export interface ConvolutionKernel {
  /** Odd, 1..63. */
  width: number;
  /** Odd, 1..63. */
  height: number;
  /** Row-major weights, `width * height` finite numbers. */
  values: KernelValues;
  /** Each sum is divided by this finite non-zero value. Default 1 (no hidden normalisation). */
  divisor?: number;
  /** Added after division. Default 0. */
  bias?: number;
}

export interface ConvolveOptions {
  border?: BorderMode;
  alpha?: AlphaMode;
  maxPixels?: number;
}

export interface SeparableOptions extends ConvolveOptions {
  divisor?: number;
  bias?: number;
}

export const MAX_KERNEL_SIZE = 63;
export const DEFAULT_MAX_PIXELS = 4_194_304;

function checkImage(image: unknown, maxPixels: number): asserts image is PixelImage {
  const candidate = image as Partial<PixelImage> | null;
  if (typeof candidate !== 'object' || candidate === null)
    throw new TypeError('image must be an object');
  const { width, height, data } = candidate;
  if (!Number.isSafeInteger(width) || !Number.isSafeInteger(height) || width! < 1 || height! < 1) {
    throw new RangeError('width and height must be positive integers');
  }
  if (!(data instanceof Uint8Array || data instanceof Uint8ClampedArray)) {
    throw new TypeError('data must be a Uint8Array or Uint8ClampedArray');
  }
  if (width! * height! > maxPixels) throw new RangeError('image exceeds maxPixels');
  if (data.length !== width! * height! * 4)
    throw new RangeError('data length must equal width * height * 4');
}

function options(value: ConvolveOptions): {
  border: BorderMode;
  alpha: AlphaMode;
  maxPixels: number;
} {
  if (typeof value !== 'object' || value === null) throw new TypeError('options must be an object');
  const border = value.border ?? 'clamp';
  const alpha = value.alpha ?? 'preserve';
  if (border !== 'clamp' && border !== 'zero') throw new TypeError('unknown border mode');
  if (alpha !== 'preserve' && alpha !== 'straight' && alpha !== 'premultiplied')
    throw new TypeError('unknown alpha mode');
  const maxPixels = value.maxPixels ?? DEFAULT_MAX_PIXELS;
  if (!Number.isSafeInteger(maxPixels) || maxPixels < 1)
    throw new RangeError('maxPixels must be a positive integer');
  return { border, alpha, maxPixels };
}

function scalar(name: string, value: unknown, fallback: number, nonZero: boolean): number {
  const chosen = value === undefined ? fallback : value;
  if (typeof chosen !== 'number' || !Number.isFinite(chosen) || (nonZero && chosen === 0)) {
    throw new RangeError(`${name} must be a finite${nonZero ? ' non-zero' : ''} number`);
  }
  return chosen;
}

function weights(name: string, values: unknown, expected?: number): readonly number[] {
  if (!(
    Array.isArray(values) ||
    values instanceof Float32Array ||
    values instanceof Float64Array
  )) {
    throw new TypeError(`${name} must be a number array, Float32Array or Float64Array`);
  }
  const list = values as ArrayLike<unknown>;
  if (expected !== undefined && list.length !== expected)
    throw new RangeError(`${name} must contain ${expected} values`);
  const copy: number[] = [];
  for (let i = 0; i < list.length; i++) {
    const value = list[i];
    if (typeof value !== 'number' || !Number.isFinite(value))
      throw new RangeError(`${name}[${i}] must be a finite number`);
    copy.push(value);
  }
  return copy;
}

function oddSize(name: string, value: unknown): number {
  if (
    !Number.isSafeInteger(value) ||
    (value as number) < 1 ||
    (value as number) > MAX_KERNEL_SIZE ||
    (value as number) % 2 === 0
  ) {
    throw new RangeError(`${name} must be an odd integer from 1 to ${MAX_KERNEL_SIZE}`);
  }
  return value as number;
}

/** Working buffer: raw channel values, or RGB scaled by alpha/255 for premultiplied mode. */
function load(image: PixelImage, alpha: AlphaMode): Float64Array {
  const source = new Float64Array(image.data.length);
  for (let i = 0; i < source.length; i += 4) {
    const a = image.data[i + 3]!;
    const scale = alpha === 'premultiplied' ? a / 255 : 1;
    source[i] = image.data[i]! * scale;
    source[i + 1] = image.data[i + 1]! * scale;
    source[i + 2] = image.data[i + 2]! * scale;
    source[i + 3] = a;
  }
  return source;
}

function store(
  image: PixelImage,
  values: Float64Array,
  alpha: AlphaMode,
  divisor: number,
  bias: number,
): PixelImage {
  const out = new Uint8ClampedArray(values.length);
  const byte = (v: number): number => Math.min(255, Math.max(0, Math.round(v)));
  for (let i = 0; i < values.length; i += 4) {
    const v = (c: number): number => values[i + c]! / divisor + bias;
    if (alpha === 'premultiplied') {
      const a = Math.min(255, Math.max(0, v(3)));
      for (let c = 0; c < 3; c++) out[i + c] = a > 0 ? byte((v(c) * 255) / a) : 0;
      out[i + 3] = byte(a);
    } else {
      for (let c = 0; c < 3; c++) out[i + c] = byte(v(c));
      out[i + 3] = alpha === 'preserve' ? image.data[i + 3]! : byte(v(3));
    }
  }
  return { width: image.width, height: image.height, data: out };
}

function index(coordinate: number, size: number, border: BorderMode): number {
  if (coordinate >= 0 && coordinate < size) return coordinate;
  return border === 'clamp' ? Math.min(size - 1, Math.max(0, coordinate)) : -1;
}

/** True 2-D convolution (the kernel is flipped, unlike correlation). Returns a new image. */
export function convolve(
  image: PixelImage,
  kernel: ConvolutionKernel,
  convolveOptions: ConvolveOptions = {},
): PixelImage {
  const { border, alpha, maxPixels } = options(convolveOptions);
  checkImage(image, maxPixels);
  if (typeof kernel !== 'object' || kernel === null)
    throw new TypeError('kernel must be an object');
  const kw = oddSize('kernel.width', kernel.width);
  const kh = oddSize('kernel.height', kernel.height);
  const k = weights('kernel.values', kernel.values, kw * kh);
  const divisor = scalar('divisor', kernel.divisor, 1, true);
  const bias = scalar('bias', kernel.bias, 0, false);
  const { width, height } = image;
  const source = load(image, alpha);
  const sums = new Float64Array(source.length);
  const cx = (kw - 1) / 2;
  const cy = (kh - 1) / 2;
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const base = (y * width + x) * 4;
      for (let j = 0; j < kh; j++) {
        const sy = index(y + cy - j, height, border);
        if (sy < 0) continue;
        for (let i = 0; i < kw; i++) {
          const sx = index(x + cx - i, width, border);
          if (sx < 0) continue;
          const weight = k[j * kw + i]!;
          const at = (sy * width + sx) * 4;
          sums[base] = sums[base]! + source[at]! * weight;
          sums[base + 1] = sums[base + 1]! + source[at + 1]! * weight;
          sums[base + 2] = sums[base + 2]! + source[at + 2]! * weight;
          sums[base + 3] = sums[base + 3]! + source[at + 3]! * weight;
        }
      }
    }
  }
  return store(image, sums, alpha, divisor, bias);
}

/** Equivalent to `convolve` with the outer-product kernel `vertical[j] * horizontal[i]`, in O(k) per pixel. */
export function convolveSeparable(
  image: PixelImage,
  horizontal: KernelValues,
  vertical: KernelValues,
  separableOptions: SeparableOptions = {},
): PixelImage {
  const { border, alpha, maxPixels } = options(separableOptions);
  checkImage(image, maxPixels);
  const h = weights('horizontal', horizontal);
  const v = weights('vertical', vertical);
  oddSize('horizontal length', h.length);
  oddSize('vertical length', v.length);
  const divisor = scalar('divisor', separableOptions.divisor, 1, true);
  const bias = scalar('bias', separableOptions.bias, 0, false);
  const { width, height } = image;
  const source = load(image, alpha);
  const middle = new Float64Array(source.length);
  const cx = (h.length - 1) / 2;
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const base = (y * width + x) * 4;
      for (let i = 0; i < h.length; i++) {
        const sx = index(x + cx - i, width, border);
        if (sx < 0) continue;
        const at = (y * width + sx) * 4;
        for (let c = 0; c < 4; c++) middle[base + c] = middle[base + c]! + source[at + c]! * h[i]!;
      }
    }
  }
  const sums = new Float64Array(source.length);
  const cy = (v.length - 1) / 2;
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const base = (y * width + x) * 4;
      for (let j = 0; j < v.length; j++) {
        const sy = index(y + cy - j, height, border);
        if (sy < 0) continue;
        const at = (sy * width + x) * 4;
        for (let c = 0; c < 4; c++) sums[base + c] = sums[base + c]! + middle[at + c]! * v[j]!;
      }
    }
  }
  return store(image, sums, alpha, divisor, bias);
}

/** `size` equal weights; pair with divisor `size` (1-D) or use as both axes with divisor `size * size`. */
export function boxKernel(size: number): number[] {
  return Array.from({ length: oddSize('size', size) }, () => 1);
}

/** Normalised (sum 1) sampled Gaussian; default radius is ceil(3 * sigma). */
export function gaussianKernel(sigma: number, radius?: number): number[] {
  if (typeof sigma !== 'number' || !Number.isFinite(sigma) || sigma <= 0)
    throw new RangeError('sigma must be positive and finite');
  const r = radius === undefined ? Math.ceil(3 * sigma) : radius;
  if (!Number.isSafeInteger(r) || r < 0 || r * 2 + 1 > MAX_KERNEL_SIZE) {
    throw new RangeError(`radius must be an integer from 0 to ${(MAX_KERNEL_SIZE - 1) / 2}`);
  }
  const raw = Array.from({ length: 2 * r + 1 }, (_, i) =>
    Math.exp(-((i - r) ** 2) / (2 * sigma * sigma)),
  );
  const total = raw.reduce((a, b) => a + b, 0);
  return raw.map((value) => value / total);
}
