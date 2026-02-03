# FFT-Bench Architecture Plan

## Overview

FFT-Bench is a benchmarking tool for comparing FFT (Fast Fourier Transform) implementations available in Python. It measures runtime performance across variable problem configurations and produces comparative visualizations.

## Design Philosophy

- **Extensibility over complexity**: New backends are added by creating a single file with a class that satisfies the `Backend` protocol and registering it. No framework changes needed.
- **Separation of concerns**: Benchmarking is split into distinct stages (configure → run → serialize → plot) connected by a well-defined JSON data contract. The `run` and `plot` steps are fully independent CLI subcommands.
- **Fail gracefully per-config**: If one backend/size/dtype combination fails (e.g. MemoryError on a large 3D problem), it is recorded as a failed result rather than aborting the entire suite.
- **Measure what matters**: Setup costs (data generation, FFTW planning) are excluded from timing. Only the actual FFT computation is measured via `time.perf_counter` with explicit warmup.
- **Native backend readiness**: The `BackendKind` enum and three-phase lifecycle (`setup`/`execute`/`teardown`) are designed so that future CUDA/C++ backends can be added without restructuring. Native backends compile in `setup()`, run via subprocess in `execute()`, and clean up in `teardown()`.

## Directory Layout

```
src/fft_bench/
    __init__.py          # Wire main() to CLI dispatch
    cli.py               # argparse CLI: "run" and "plot" subcommands
    config.py            # SingleBenchmarkConfig, parameter grid expansion
    backends/
        __init__.py      # Backend registry (register, get, discover)
        base.py          # Backend Protocol, BackendKind enum, BackendCapabilities
        numpy_.py        # NumPy FFT backend
        scipy_.py        # SciPy FFT backend (supports workers/threads)
        pyfftw_.py       # pyFFTW backend (pre-planned, supports threads)
    runner.py            # Benchmark orchestration loop
    timing.py            # Warmup + perf_counter timing utility
    results.py           # BenchmarkResult, BenchmarkSuite, JSON serialization
    hardware.py          # HardwareInfo capture (CPU, RAM, OS, library versions)
    plotting.py          # matplotlib plot generation from result JSON
```

## Core Abstractions

### Backend Protocol (`backends/base.py`)

Three-phase lifecycle that accommodates both Python and future native (CUDA/C++) backends:

- **`setup(shape, dtype, threads)`** -- generate input data / compile native code / create FFTW plan
- **`execute()`** -- run a single FFT (this is what gets timed)
- **`teardown()`** -- release resources

Supporting types:
- **`BackendKind`** enum: `PYTHON` | `NATIVE` -- allows runner to adjust timing strategy for native backends
- **`BackendCapabilities`** dataclass: `supported_dtypes`, `supports_threading`, `max_ndim`, `kind` -- used during parameter grid expansion to filter incompatible combinations

### Backend Registry (`backends/__init__.py`)

Simple dict-based registry. Each backend module registers itself at import time. `discover_backends()` imports all backend modules (called once at CLI startup). Adding a new backend = one new file + one registration call.

### Configuration (`config.py`)

- **`SingleBenchmarkConfig`**: frozen dataclass with `backend`, `shape`, `dtype`, `threads`, `warmup`, `repetitions`
- **`expand_parameter_grid()`**: Cartesian product of all parameter axes, filtered against backend capabilities. Shape is `(size,) * ndim`. Logs warnings for skipped incompatible combos.

### Results (`results.py`)

- **`BenchmarkResult`**: per-config result with `config`, `timings: list[float]`, computed stats (`mean`, `std`, `median`, `min`, `max`), `success: bool`, `error: str | None`
- **`BenchmarkSuite`**: collection of results + `HardwareInfo` + timestamp + version. Has `save(path)` / `load(path)` for JSON round-tripping.

### Hardware (`hardware.py`)

- **`HardwareInfo`**: CPU model, core counts, RAM, OS, Python version, library versions
- Uses `platform` module + `/proc/cpuinfo` + `/proc/meminfo` on Linux. No extra dependencies.

## Data Flow

```
CLI args (argparse)
    |
    v
expand_parameter_grid() --> list[SingleBenchmarkConfig]
    |                         (filters incompatible combos)
    v
run_benchmarks(configs)
    |  for each config:
    |    backend.setup(shape, dtype, threads)
    |    timings = benchmark_single(backend.execute, warmup, reps)
    |    backend.teardown()
    |    --> BenchmarkResult
    v
BenchmarkSuite(results, hardware, timestamp)
    |
    v
suite.save("results.json")
    ======================================
suite = BenchmarkSuite.load("results.json")
    |
    v
generate_all_plots(suite, output_dir)
    |
    v
PNG/PDF/SVG files
```

## CLI Interface

```
fft-bench run --backends numpy scipy pyfftw \
              --sizes 64 128 256 512 1024 \
              --ndims 1 2 3 \
              --dtypes float64 complex128 \
              --threads 1 2 4 \
              --warmup 3 --repetitions 10 \
              -o results.json

fft-bench plot results.json -o plots/ --format png --dpi 150
```

## Plots Generated

1. **Runtime vs Size** -- per ndim, lines colored by backend, styled by dtype (log-log scale). Error bars for variability.
2. **Backend Comparison Bars** -- grouped bar chart per (ndim, dtype), groups = sizes, bars = backends.
3. **Threading Scaling** -- per (backend, ndim, dtype) where threading supported. X=threads, Y=runtime, lines per size.
4. **Dtype Impact** -- per (backend, ndim). X=size, Y=runtime, lines per dtype.

## Backend Details

| Backend | Threading | FFT functions | Notes |
|---------|-----------|---------------|-------|
| numpy   | No        | `numpy.fft.fft/fft2/fftn` | Simplest, no thread param |
| scipy   | Yes       | `scipy.fft.fft/fft2/fftn(workers=N)` | Thread support via `workers` |
| pyfftw  | Yes       | `pyfftw.builders.fft/fft2/fftn(threads=N)` | Plan created in `setup()`, callable plan in `execute()`. Planning cost excluded from timing. |

## Input Data Generation

- Real dtypes (`float32`, `float64`): `np.random.standard_normal(shape).astype(dtype)`
- Complex dtypes (`complex64`, `complex128`): `(np.random.standard_normal(shape) + 1j * np.random.standard_normal(shape)).astype(dtype)`
- Generated once in `setup()`, reused across all repetitions.

## Edge Case Handling

- **Threading incompatibility**: `expand_parameter_grid` emits only `threads=1` for non-threaded backends, warns about skipped values.
- **Unsupported dtypes**: filtered during grid expansion with logged warning.
- **MemoryError**: caught per-config, recorded as failed result.
- **pyFFTW planning**: done in `setup()`, excluded from timed region.

## Future CUDA/C++ Extension

The `BackendKind.NATIVE` path is structurally present. A native backend would:
- Compile in `setup()` (or use a cached binary)
- Run via subprocess in `execute()`, parse timing from stdout
- Clean up temp files in `teardown()`
- The runner can check `capabilities.kind` to optionally defer timing to the native program itself

No structural changes needed -- just a new file in `backends/`.

## Implementation Order

1. `config.py` -- SingleBenchmarkConfig + grid expansion
2. `hardware.py` -- HardwareInfo
3. `results.py` -- central data model + JSON serialization
4. `backends/base.py` -- Backend protocol + capabilities
5. `backends/__init__.py` -- registry
6. `backends/numpy_.py` -- first concrete backend
7. `backends/scipy_.py`
8. `backends/pyfftw_.py`
9. `timing.py` -- timing utility
10. `runner.py` -- orchestrator
11. `cli.py` -- argparse CLI
12. `plotting.py` -- plot generation
13. `__init__.py` -- wire main() to CLI
