# fft-bench

A benchmarking tool for comparing FFT (Fast Fourier Transform) libraries available in Python. It measures runtime performance across configurable problem sizes, dimensions, data types, and thread counts, then produces comparative visualizations.

## Supported Backends

| Backend | Threading | Notes |
|---------|-----------|-------|
| NumPy   | No        | Uses `numpy.fft` |
| SciPy   | Yes       | Uses `scipy.fft` with `workers` parameter |
| pyFFTW  | Yes       | Pre-planned FFTW via `pyfftw.builders`; planning cost excluded from timing |

## Installation

Requires Python 3.12+. Managed with [uv](https://docs.astral.sh/uv/):

```bash
uv sync
```

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
```

**Key options:**

| Option | Default | Description |
|--------|---------|-------------|
| `--backends` | all available | Backends to benchmark |
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
