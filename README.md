# Transmitter-Receiver

CUDA kernels and host wrappers for the signal-processing blocks used in a
digital demodulator — the kind of building blocks a software-defined radio
(SDR) receiver chain is made of. Each block lives in its own directory, is
buildable and runnable in isolation against a test signal, and ships with a
Python script to verify its output and a measured GPU throughput table.

## Receiver chain

The blocks are numbered by their typical position in a receiver chain, from
the raw IF/baseband samples down to soft bits for the channel decoder:

```
IF/baseband samples
       │
       ▼
 01 DDC  ──────────────►  shift to baseband (mix with NCO)
       │
       ▼
 02 FilterDownSample ───►  decimating FIR filter
       │
       ▼
 03 BaseBandFilter ─────►  matched/noise-reduction FIR filter
       │
       ▼
 04 Resampler ───────────►  arbitrary (non-integer) rate conversion
       │
       ▼
 05 Normalization ──────►  AGC / unit-power scaling
       │
       ▼
 06 TimingRecovery ─────►  symbol timing (Gardner) + carrier PLL
       │
       ▼
    PLL (Reserved)
       │
       ▼
 09 Soft-Demapper ──────►  symbols → soft bits (PSK / APSK)
       │
       ▼
 10 ChannelEncoderDecoder ► LDPC / TPC / Turbo decoding
```

Gaps in the numbering (07, 08) are reserved for a stage (PLL) that is not
published in this repository yet.

## Blocks

| # | Block | Status | Description |
|---|-------|--------|--------------|
| 01 | [DDC](01DDC/README.md) | done | Digital down-converter: mixes the IQ input with a complex NCO to shift it to baseband |
| 02 | [FilterDownSample](02FilterDownSample/README.md) | done | FIR decimation filter, with time-domain and cuFFT frequency-domain implementations |
| 03 | [BaseBandFilter](03BaseBandFilter/README.md) | done | Baseband FIR filtering to reduce noise, also outputs magnitude-squared |
| 04 | [Resampler](04Resampler/README.md) | done | Arbitrary (non-integer) sample-rate conversion via cubic (Farrow) interpolation |
| 05 | [Normalization](05Normalization/README.md) | done | Unit-power normalization or AGC (peak-amplitude) scaling |
| 06 | [TimingRecovery](06TimingRecovery/FastGardner) | kernel + host wrapper exist, no block README yet | Gardner symbol-timing recovery and carrier PLL, ported from the `demo/` prototypes |
| 09 | [SoftDemapper](09SoftDemapper) | kernel only, no host wrapper / `make` target | Maps QPSK / 8PSK / 16APSK / 32APSK symbols to soft bits (LLR) |
| 10 | [ChannelEncoderDecoder](10ChannelEncoderDecoder/README.md) | kernels for LDPC, TPC and Turbo decoders, no `make` target | Channel decoders (see its own roadmap for what's planned) |

Each block directory under 01–05 follows the same layout: a `kernel.cu`
with the device code, a `*_main.cu` host wrapper that loads a test signal,
runs and times the kernel, and writes the result, and an `outputCheck.py`
to plot/verify the output. Open a block's own README for its exact build
command, CLI arguments and measured throughput.

## Prerequisites

- CUDA Toolkit (developed against `nvcc` 12.2) and a CUDA-capable GPU
- GNU Make
- `cuFFT` for the frequency-domain paths in `FilterDownSample` and
  `BaseBandFilter` (ships with the CUDA Toolkit)
- Python 3 with `numpy`, `scipy`, and `matplotlib` to run the `outputCheck.py`
  verification scripts

## Building

Each block is built independently through the root `Makefile`:

```
make block <BlockName>
```

e.g.

```
make block DDC
build/ddc 01DDC/input.bin 01DDC/output.bin 0.1
```

Run `make block` with no arguments to list the available block names.
Binaries are written to `build/`; `make clean` removes that directory.

`TimingRecovery` additionally needs `demod_parallel.a`, built from
`06TimingRecovery/FastGardner/demod_parallel.cu` (see the comment above the
`TimingRecovery` target in the `Makefile`) — the repo only ships the Windows
`demod_parallel.lib`, which does not link on Linux.

## Repository layout

- `01DDC/`, `02FilterDownSample/`, `03BaseBandFilter/`, `04Resampler/`,
  `05Normalization/`, `06TimingRecovery/` — CUDA blocks described above
- `09SoftDemapper/`, `10ChannelEncoderDecoder/` — additional kernels not yet
  wired into the `Makefile`
- `utils/` — shared host-side helpers: pinned-memory buffers
  (`pinnedMemory.hpp`) and binary IQ file I/O (`utils.cpp`/`utils.h`)
- `demo/` — prebuilt Windows executables (`.exe` + `RunMe.bat`) combining
  timing recovery and carrier PLL for QPSK/OQPSK test signals, plus a
  MATLAB script (`showResult.m`) to plot the recovered constellation
- `Archive/` — earlier MATLAB-based signal generators/checkers for the DDC
  and FilterDownSample blocks, kept for reference

## Verifying output

Every block's `outputCheck.py` reads its `output.bin` (interleaved
float32 IQ) and plots the power spectral density (and, for `Normalization`,
prints max amplitude / average power) so the result can be sanity-checked
against what the block is expected to do, e.g.:

```
python3 01DDC/outputCheck.py 01DDC/output.bin
```

See each block's README for the exact invocation.
