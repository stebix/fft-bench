# fft-bench

A benchmarking tool for comparing FFT (Fast Fourier Transform) libraries available in Python. It measures runtime performance across configurable problem sizes, dimensions, data types, and thread counts, then produces comparative visualizations.

## Supported Backends

### Python backends

| Backend | Threading | Notes |
|---------|-----------|-------|
| NumPy   | No        | Uses `numpy.fft` |
| SciPy   | Yes       | Uses `scipy.fft` with `workers` parameter |
| pyFFTW  | Yes       | Pre-planned FFTW via `pyfftw.builders`; planning cost excluded from timing |
| CuPy    | No        | GPU-accelerated via cuFFT; timed from Python with device synchronization |

### Native backends

These backends invoke an external `cufft-bench` binary that performs its own timing using CUDA events, providing sub-microsecond GPU-side resolution without Python overhead.

| Backend   | Mode   | Notes |
|-----------|--------|-------|
| `cufft`     | kernel | Times only the `cufftExec*` call (pure FFT compute) |
| `cufft-e2e` | e2e    | Times the full cycle: memory allocation, host-to-device copy, FFT, device-to-host copy, deallocation |

Native backends are **self-timed** — they bypass fft-bench's `time.perf_counter()` loop and return per-iteration timings directly from CUDA events. They are automatically discovered and registered when the `cufft-bench` binary is available (see [Native Backend Setup](#native-backend-setup) below).

## Installation

Requires Python 3.12+. Managed with [uv](https://docs.astral.sh/uv/):

```bash
uv sync
```

### Native Backend Setup

The native cuFFT backends require the `cufft-bench` binary, which is built separately from the [cufft-bench](https://github.com/stebix/cufft-bench) companion project. fft-bench locates the binary in two ways:

1. **Environment variable** — set `CUFFT_BENCH_PATH` to the absolute path of the binary:
   ```bash
   export CUFFT_BENCH_PATH=/path/to/cufft-bench/build/cufft-bench
   ```
2. **PATH lookup** — if the variable is not set, fft-bench falls back to `shutil.which("cufft-bench")`, so placing the binary on your `PATH` also works.

If neither method finds the binary, the `cufft` and `cufft-e2e` backends are silently skipped and won't appear in the available backends list.

## Usage

The tool provides two subcommands: `run` (execute benchmarks) and `plot` (generate visualizations from results).

### Running Benchmarks

```bash
# Run all backends with default settings
fft-bench run

# Customize backends, sizes, and dimensions
fft-bench run --backends numpy scipy --sizes 64 128 256 512 1024 --ndims 1 2

# Full example with all options
fft-bench run --backends numpy scipy pyfftw \
              --sizes 64 128 256 512 1024 \
              --ndims 1 2 3 \
              --dtypes float64 complex128 \
              --threads 1 2 4 \
              --warmup 5 --repetitions 20 \
              -o results.json

# Run native cuFFT backends alongside Python backends
CUFFT_BENCH_PATH=/path/to/cufft-bench/build/cufft-bench \
  fft-bench run --backends numpy scipy cufft cufft-e2e \
                --sizes 256 512 1024 --ndims 1 2 --dtypes float32
```

**Key options:**

| Option | Default | Description |
|--------|---------|-------------|
| `--backends` | all available | Backends to benchmark (`numpy`, `scipy`, `pyfftw`, `cupy`, `cufft`, `cufft-e2e`) |
| `--sizes` | `64 128 256 512 1024` | FFT sizes along each dimension |
| `--ndims` | `1` | Number of dimensions (1, 2, or 3) |
| `--dtypes` | `float64 complex128` | Data types (`float32`, `float64`, `complex64`, `complex128`) |
| `--threads` | `1` | Thread counts to test (ignored for non-threaded backends) |
| `--warmup` | `3` | Warmup iterations before timing |
| `--repetitions` | `10` | Timed repetitions per configuration |
| `-o, --output` | `results.json` | Output file path |
| `--overwrite-policy` | `auto-rename` | `auto-rename`, `force`, or `error` |
| `-f, --force` | | Shorthand for `--overwrite-policy force` |
| `-v, --verbose` | | Enable debug logging |

Incompatible configurations (e.g. threading on NumPy, unsupported dtypes) are automatically filtered out with a logged warning.

### Generating Plots

```bash
# Generate plots from a results file
fft-bench plot results.json

# Customize output directory and format
fft-bench plot results.json -o plots/ --format pdf --dpi 300
```

**Key options:**

| Option | Default | Description |
|--------|---------|-------------|
| `-o, --output` | `plots` | Output directory |
| `--format` | `png` | Image format (`png`, `pdf`, `svg`) |
| `--dpi` | `150` | Resolution for raster formats |
| `--overwrite-policy` | `auto-rename` | `auto-rename`, `force`, or `error` |
| `-f, --force` | | Shorthand for `--overwrite-policy force` |

**Generated plot types:**

- **Runtime vs Size** -- log-log plot per dimension count, lines by backend and dtype
- **Backend Comparison** -- grouped bar chart per (ndim, dtype) combination
- **Threading Scaling** -- runtime vs thread count for threading-capable backends
- **Dtype Impact** -- runtime vs size broken down by data type per backend

### Python API

fft-bench can also be used programmatically, e.g. from Jupyter notebooks. Install the optional `notebook` dependency group for pandas support:

```bash
uv sync --extra notebook
```

#### Running benchmarks

```python
import fft_bench

suite = fft_bench.run(
    backends=["numpy", "scipy"],
    sizes=[64, 128, 256, 512, 1024],
    ndims=[1, 2],
    dtypes=["float64", "complex128"],
)

# Native backends work identically (requires CUFFT_BENCH_PATH or binary on PATH)
suite = fft_bench.run(
    backends=["numpy", "cufft", "cufft-e2e"],
    sizes=[256, 512, 1024],
    dtypes=["float32"],
)

suite.run_id   # "brave-calm-otter"
suite.hardware.cpu_model  # "AMD EPYC 7452 32-Core Processor"
```

#### Loading previous results

```python
suite = fft_bench.load("results.json")
```

#### Converting to DataFrame

```python
df = suite.to_dataframe()

# Filter and plot
df_1d = df[(df["ndim"] == 1) & (df["success"])]
df_1d.pivot_table(index="size", columns="backend", values="mean").plot()
```

#### Aggregating multiple runs

Each suite carries a unique `run_id`, making it straightforward to combine and distinguish runs:

```python
import pandas as pd

df_all = pd.concat([suite1.to_dataframe(), suite2.to_dataframe()])
df_all.groupby(["run_id", "backend", "size"]).mean(numeric_only=True)
```

## Output Format

Results are saved as JSON containing hardware info, library versions, and per-configuration timing data with summary statistics (mean, std, median, min, max). The `run` and `plot` stages are fully decoupled through this JSON contract.

## Adding a New Backend

Create a file in `src/fft_bench/backends/` implementing the `Backend` protocol (the `setup`/`execute`/`teardown` lifecycle) and register it. No other changes are required.

For backends that perform their own timing (e.g. wrapping an external binary), implement a `run_timed(shape, dtype, threads, warmup, repetitions) -> list[float]` method instead. The runner detects this via duck typing and calls it directly, bypassing the standard three-phase cycle. Timings must be returned in seconds.
