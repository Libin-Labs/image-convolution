# @lp-labs/image-convolution

Convolves 8-bit RGBA pixel buffers with odd kernels (direct or separable), explicit border handling and explicit alpha policies. Catalog identity 491; standalone package version 1.0.0.
No runtime, peer, or sibling dependencies. Node >=18, ESM and CommonJS with declarations.

## Install

```sh
npm install @lp-labs/image-convolution
```

## Quickstart (runnable JavaScript)

```js
import assert from 'node:assert/strict';
import { boxKernel, convolve, convolveSeparable } from '@lp-labs/image-convolution';
const image = {
  width: 3,
  height: 1,
  data: Uint8ClampedArray.from([0, 0, 0, 255, 90, 90, 90, 255, 0, 0, 0, 255]),
};
const blurred = convolve(
  image,
  { width: 3, height: 1, values: [1, 1, 1], divisor: 3 },
  { border: 'zero' },
);
assert.deepEqual(Array.from(blurred.data.slice(0, 4)), [30, 30, 30, 255]);
const same = convolveSeparable(image, boxKernel(3), [1], { border: 'zero', divisor: 3 });
assert.deepEqual(Array.from(same.data), Array.from(blurred.data));
```

## Configuration and errors

`PixelImage` is `{ width, height, data }` with interleaved RGBA bytes in a Uint8Array or Uint8ClampedArray (`data.length === width * height * 4`). Results are new images with a Uint8ClampedArray; inputs are never modified.

`convolve(image, kernel, options)`: `kernel` has odd `width`/`height` (1..63), row-major finite `values` (number array, Float32Array or Float64Array), optional non-zero `divisor` (default 1, **no automatic normalisation**) and `bias` (default 0, added after division). It is true convolution: the kernel is flipped relative to correlation. `convolveSeparable(image, horizontal, vertical, options)` equals `convolve` with the outer-product kernel `vertical[j] * horizontal[i]` (accepts `divisor` and `bias` in options). `boxKernel(size)` and `gaussianKernel(sigma, radius?)` (sums to 1, default radius `ceil(3 * sigma)`) are helpers.

Options: `border` is `clamp` (repeat edge pixels, default) or `zero` (outside pixels contribute 0, including alpha). `alpha` is `preserve` (default: RGB convolved from the stored values and the alpha channel copied unchanged), `straight` (R, G, B and A are convolved independently, so the colour of transparent pixels bleeds) or `premultiplied` (RGB is weighted by alpha/255 before convolution, then divided by the resulting alpha, and alpha is convolved; fully transparent output pixels become black). `maxPixels` defaults to 4194304. Sums are rounded half up and clamped to 0..255.

Errors: `TypeError` for wrong container types and unknown border/alpha names; `RangeError` for bad dimensions, even or oversized kernels, wrong value counts, non-finite numbers, a zero divisor and exceeded `maxPixels`.

## Complete public declarations

Every runtime export and public type is shown below. This is the complete emitted declaration surface.

```ts
export interface PixelImage {
  width: number;
  height: number;
  data: Uint8Array | Uint8ClampedArray;
}
export type KernelValues = readonly number[] | Float32Array | Float64Array;
export type BorderMode = 'clamp' | 'zero';
export type AlphaMode = 'preserve' | 'straight' | 'premultiplied';
export interface ConvolutionKernel {
  width: number;
  height: number;
  values: KernelValues;
  divisor?: number;
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
export declare const MAX_KERNEL_SIZE = 63;
export declare const DEFAULT_MAX_PIXELS = 4194304;
export declare function convolve(
  image: PixelImage,
  kernel: ConvolutionKernel,
  convolveOptions?: ConvolveOptions,
): PixelImage;
export declare function convolveSeparable(
  image: PixelImage,
  horizontal: KernelValues,
  vertical: KernelValues,
  separableOptions?: SeparableOptions,
): PixelImage;
export declare function boxKernel(size: number): number[];
export declare function gaussianKernel(sigma: number, radius?: number): number[];
```

## Performance, boundaries and limitations

Colour handling is deliberately simple: values are treated as raw 8-bit numbers, not converted from sRGB to linear light, so blurs and sharpening are performed in gamma-encoded space; no colour management, bit depths other than 8, planar or YUV layouts, or other border modes (wrap, mirror) are provided. Results depend on the chosen alpha policy as described above. Complexity is O(pixels * kernelWidth * kernelHeight) for `convolve` and O(pixels * (kernelWidth + kernelHeight)) for the separable form; both allocate Float64 working buffers of 4 values per pixel (about 32 bytes per pixel) for all pixels, bounded by `maxPixels`. No FFT, GPU, worker or SIMD acceleration.

## Development

```sh
npm ci --offline
npm run build
npm run typecheck
npm test
npm run format:check
```

The offline install requires the locked development dependencies already in the npm cache.
Build emits separate ESM/CJS and declarations; type checks include bare-package .mts/.cts consumers with negative cases.
Runtime tests import only the single-quoted standalone dist entry and also run against installed offline tarballs.
No build artifacts or development dependencies are required at runtime.

## Support

If this library is useful to you, consider supporting its maintenance with a [donation via PayPal](https://www.paypal.me/LibinPrasanth).

## Author

Created by [Libin Prasanth](https://www.linkedin.com/in/libinprasanth/). Connect on LinkedIn for updates and discussion.
