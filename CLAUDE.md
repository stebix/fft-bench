# fft-bench

This project is concerned with benchmarking different FFT backends (i.e. fast fourier transform supporting libraries) that are usable from Python.
We want to get runtime estimations for different problem setups with variable problem sizes. The runtime estimations should ultimately produce
graphs.

## project standards

- use numpy-style docstrings
- this project is generally managed with `uv`