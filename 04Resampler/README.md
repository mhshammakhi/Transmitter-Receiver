# 4 - Resampler

Converts an IQ signal between arbitrary (non-integer) sample rates using a
cubic (Farrow-structure) interpolator. Unlike the integer-factor
`FilterDownSample` decimator, the rate change `iFs/oFs` need not be an
integer, so each output sample falls at an arbitrary fractional position
between input samples.

## Files

- `kernel.cu` — `resampler` CUDA kernel (grid-stride loop over output
  samples, four-tap cubic interpolation per sample via
  `cubic_interpolation_comp`) and `resampler_cals_step_host`, a host-side
  helper that precomputes the per-launch position step from the launch
  configuration and `iFs`/`oFs`
- `resampler_main.cu` — host wrapper: loads the input signal, sizes the
  output buffer from the kernel's own `outlen` formula, runs the kernel,
  times the execution and writes the output signal
- `outputCheck.py` — plots the power spectral density of `output.bin` to
  verify the resampled output
- `input.bin` — input IQ test signal

## Build

```
make block Resampler
```

produces `build/resampler`.

Example (rate change of 4:3, i.e. `oFs/iFs = 0.75`):

```
make block Resampler
build/resampler 04Resampler/input.bin 04Resampler/output.bin 4 3
```

`iFs` and `oFs` only matter as a ratio — pass any pair of values whose
quotient is the desired input/output rate change (e.g. actual sample rates
in Hz, or just `4 3`).

## Persistent state across frames

The kernel keeps a small amount of `__managed__` state
(`d_resample_nFromPrev`, `d_last_startInterpIndex`, `d_resample_prev_re/im`)
so it can be called frame-by-frame on a continuous stream without losing
interpolation phase or the trailing input samples a frame's last output
needed. `resampler_main.cu` runs the whole input as a single frame, relying
on the cold-start defaults declared in `kernel.cu`.

## Processing rate

| Test Case| Throughput |
|---|---|
| Kernel only         |4054.51 MSamples/s|
| H2D + kernel + D2H  |225.701 MSamples/s|

The implementation can also be extended to support multiple parallel streams with minimal effort, allowing the overall processing throughput to be increased significantly.
