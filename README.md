# fftbench
FFT benchmarking code

## Supported languages
**fftbench** can be used to benchmark shared memory libraries providing Fourier based transforms.
Current C++ implementation allows one to choose among MKL, FFTW and cuFFT backends, whereas Matlab
language scripts rely on `fftn()` and `ifftn()` Matlab functions.

### C++ interface compilation
A number of environment variables can be used to target libraries to be supported.

#### fftw3
- `FFTW3ROOT` can be used to store fftw3 installation path (i.e. the folder that contains fftw3 `lib` and `include` folders)
- `FFTW3_INCLUDE_PATH` can be used to store fftw3 library path (i.e. the absolute path of fftw3 `include` folder)
- `FFTW3_LIBRARY_PATH` can be used to store fftw3 library path (i.e. the absolute path of fftw3 `lib` folder)

#### mkl

#### cuFFT
