import {
  boxKernel,
  convolve,
  convolveSeparable,
  gaussianKernel,
  type AlphaMode,
  type BorderMode,
  type ConvolutionKernel,
  type PixelImage,
} from '@lp-labs/image-convolution';
const image: PixelImage = { width: 1, height: 1, data: new Uint8ClampedArray(4) };
const kernel: ConvolutionKernel = {
  width: 3,
  height: 3,
  values: boxKernel(3).flatMap((a) => boxKernel(3).map((b) => a * b)),
  divisor: 9,
};
const border: BorderMode = 'zero';
const alpha: AlphaMode = 'premultiplied';
const blurred: PixelImage = convolve(image, kernel, { border, alpha });
convolveSeparable(blurred, gaussianKernel(1), gaussianKernel(1), {
  divisor: 1,
}).data.length.toFixed();
// @ts-expect-error unknown alpha policy
convolve(image, kernel, { alpha: 'ignore' });
// @ts-expect-error unknown border policy
convolve(image, kernel, { border: 'wrap' });
// @ts-expect-error pixels must be a byte typed array
convolve({ width: 1, height: 1, data: [0, 0, 0, 0] }, kernel);
// @ts-expect-error kernel values are required
convolve(image, { width: 3, height: 3 });
