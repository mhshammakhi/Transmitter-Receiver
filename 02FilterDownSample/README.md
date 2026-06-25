# 2 - FilterDownSample

Filters an IQ signal with a FIR decimation filter and downsamples it by an
integer factor.

## Files

- `kernel.cu` — filter + downsample CUDA kernel
- `fds_main.cu` — host wrapper: loads the input signal and filter coefficients,
  runs the kernel, times the execution and writes the output signal
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

## Processing rate

| Test Case| Throughput |
|---|---|
| Kernel only         |6290.25 MSamples/s|
| H2D + kernel + D2H  | 266.076 MSamples/s|

The implementation can also be extended to support multiple parallel streams with minimal effort, allowing the overall processing throughput to be increased significantly.
