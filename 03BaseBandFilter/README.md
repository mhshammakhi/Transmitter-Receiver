# 3 - BaseBandFilter

Filters a baseband IQ signal with a FIR filter to reduce noise effect, and
also outputs the per-sample magnitude-squared. Two interchangeable
implementations are provided: a direct time-domain FIR convolution and an
FFT-based (frequency-domain) convolution using cuFFT.

## Files

- `kernel.cu` — CUDA kernels for both the time-domain (`filterBaseband`) and
  the FFT-based (`packSignalComplex`, `packFilterComplex`,
  `complexMultiplyScale`, `extractFiltered`) implementations
- `bbf_main.cu` — host wrapper: loads the input signal and filter coefficients,
  runs the selected implementation, times the execution and writes the output
  signal (and optionally the magnitude-squared)
- `outputCheck.py` — plots the power spectral density of `output.bin` to verify the filtered output
- `input.bin` — input IQ test signal
- `coef.bin` / `coeflong.bin` — FIR filter coefficients (float32), short and long examples

## Build

```
make block BaseBandFilter
```

produces `build/baseband_filter`.

Example:

```
make block BaseBandFilter
build/baseband_filter 03BaseBandFilter/input.bin 03BaseBandFilter/output.bin 03BaseBandFilter/coef.bin
```

The optional fourth argument writes the magnitude-squared per sample:

```
build/baseband_filter 03BaseBandFilter/input.bin 03BaseBandFilter/output.bin 03BaseBandFilter/coef.bin 03BaseBandFilter/abs.bin
```

## Time domain vs. frequency domain

The implementation is chosen at compile time with the `USE_FFT_FILTER` macro
at the top of `bbf_main.cu`:

- `USE_FFT_FILTER 0` (default) — direct time-domain FIR convolution, computed
  at every output position
- `USE_FFT_FILTER 1` — FFT-based convolution: a single cuFFT transform spanning
  the whole signal, frequency-domain multiply, inverse transform, then extract

Change the value and rebuild (`make block BaseBandFilter`) to switch between
them. The FFT-based path scales with signal length regardless of filter
length, so it becomes favorable for long filters such as `coeflong.bin`; the
time-domain path is faster for short filters such as `coef.bin`.

## Processing rate

| Test Case| Throughput | Description
|---|---|---|
| Kernel only    |7787.07 MSamples/s|Time Domain - Short Coef |
| Kernel only    |354.1 MSamples/s|Time Domain - Long Coef |
| H2D + kernel + D2H | 160 MSamples/s|Time Domain - * |
| Kernel only    |1190.92 MSamples/s|Frequency Domain - Short/long Coef |
| H2D + kernel + D2H | 170 MSamples/s|Frequency Domain - * |

The implementation can also be extended to support multiple parallel streams with minimal effort, allowing the overall processing throughput to be increased significantly.
