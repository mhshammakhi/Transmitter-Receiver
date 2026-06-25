# 1 - DDC (Digital Down Converter)

Mixes an IQ input signal with a complex local oscillator to shift it from an
intermediate/non-baseband frequency down to baseband.

## Files

- `kernel.cu` — DDC CUDA kernel
- `ddc_main.cu` — host wrapper: loads the input signal, runs the kernel, times
  the execution and writes the output signal
- `outputCheck.py` — plots the power spectral density of `output.bin` to verify the DDC output
- `signal.bin` — input IQ test signal with a 0.1 normalized frequency offset

## Build

```
make block DDC
```

produces `build/ddc`.

Example:

```
make block DDC
build/ddc 01DDC/signal.bin 01DDC/output.bin 0.1
```

## Processing rate 

This table was generated for a single data stream running on an RTX 4050. The achievable processing rate may vary depending on the GPU model, motherboard bus, CPU performance, and frame length.

| Test Case| Throughput |
|---|---|
| Kernel only         |9782.47 MSamples/s|
| H2D + kernel + D2H  |276.374 MSamples/s|




The implementation can also be extended to support multiple parallel streams with minimal effort, allowing the overall processing throughput to be increased significantly.
