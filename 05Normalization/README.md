# 5 - Normalization

Scales an IQ signal by a single factor computed from the block itself. Two
interchangeable modes are provided: UnitPowerNormalization (scale so average
power is 1) and AGC (scale so peak amplitude is 1).

## Files

- `kernel.cu` — CUDA kernels: `computePowerSamples` (pow(abs(sample),2) per
  sample), `parallelSum_arbitraryLen` / `parallelMax_arbitraryLen` (block
  reductions feeding each mode), `NF_LoopFilter_MF` / `NF_LoopFilter_AGC`
  (turn the reduction result into a normalize factor), and `Normalize_MF`
  (applies the factor to the IQ signal)
- `norm_main.cu` — host wrapper: loads the input signal, runs the selected
  mode, times the execution and writes the output signal
- `outputCheck.py` — plots the power spectral density of `output.bin`, and
  prints its max amplitude and average power to verify the normalization
- `input.bin` — input IQ test signal

## Build

```
make block Normalization
```

produces `build/normalization`.

Example:

```
make block Normalization
build/normalization 05Normalization/input.bin 05Normalization/output.bin
```

## AGC vs. unit power normalization

The mode is chosen at compile time with the `USE_AGC` macro at the top of
`norm_main.cu`:

- `USE_AGC 0` (default) — **UnitPowerNormalization**: scales the signal so its
  average power (RMS²) is 1
- `USE_AGC 1` — **AGC**: scales the signal so its peak amplitude (max |sample|)
  is 1

Change the value and rebuild (`make block Normalization`) to switch between
them.

## Processing rate

| Test Case| Throughput |
|---|---|
| Kernel only   UnitPower      | 3380.52 MSamples/s|
| H2D + kernel + D2H  UnitPower| 277.561 MSamples/s|
| Kernel only  AGC | 702.244 MSamples/s|
| H2D + kernel + D2H AGC |261.281 MSamples/s|
