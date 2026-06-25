# 2 - FilterDownSample

Filters an IQ signal with a FIR decimation filter and downsamples it by an
integer factor. Two interchangeable implementations are provided: a direct
time-domain FIR convolution and an FFT-based (frequency-domain) convolution
using cuFFT.

## Files

- `kernel.cu` — CUDA kernels for both the time-domain (`filterAndDownSample`)
  and the FFT-based (`packSignalComplex`, `packFilterComplex`,
  `complexMultiplyScale`, `extractDownsample`) implementations
- `fds_main.cu` — host wrapper: loads the input signal and filter coefficients,
  runs the selected implementation, times the execution and writes the output signal
- `outputCheck.py` — plots the power spectral density of `output.bin` to verify the filtered/downsampled output
- `input.bin` — input IQ test signal
- `coef2.bin` — FIR Decimation Filter Coefficients for a Decimation Factor of 2 (float32)

## Build

```
make block FilterDownSample
```

produces `build/filter_downsample`.

Example:

```
make block FilterDownSample
build/filter_downsample 02FilterDownSample/input.bin 02FilterDownSample/output.bin 02FilterDownSample/coef2.bin 2
```

## Time domain vs. frequency domain

The implementation is chosen at compile time with the `USE_FFT_FILTER` macro
at the top of `fds_main.cu`:

- `USE_FFT_FILTER 0` (default) — direct time-domain FIR convolution, computed
  only at the decimated output positions
- `USE_FFT_FILTER 1` — FFT-based convolution: a single cuFFT transform spanning
  the whole signal, frequency-domain multiply, inverse transform, then decimate

Change the value and rebuild (`make block FilterDownSample`) to switch between
them. The FFT-based path scales with signal length regardless of filter
length, so it becomes favorable for long filters; the time-domain path is
faster for short filters such as `coef2.bin`.

## Processing rate

Measured with the default time-domain algorithm (`USE_FFT_FILTER 0`):

| Test Case| Throughput |
|---|---|
| Kernel only         |6290.25 MSamples/s|
| H2D + kernel + D2H  | 266.076 MSamples/s|

The implementation can also be extended to support multiple parallel streams with minimal effort, allowing the overall processing throughput to be increased significantly.
