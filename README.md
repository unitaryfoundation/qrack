# Qrack

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.5812507.svg)](https://doi.org/10.5281/zenodo.5812507) [![Mentioned in Awesome awesome-quantum-computing](https://awesome.re/mentioned-badge.svg)](https://github.com/desireevl/awesome-quantum-computing) [![Unitary Fund](https://img.shields.io/badge/Supported%20By-UNITARY%20FUND-brightgreen.svg?style=for-the-badge)](http://unitary.fund)

## About

The open source vm6502q/qrack library and its associated plugins and projects under the vm6502q organization header comprise a framework for full-stack quantum computing development, via high performance and fundamentally optimized simulation. The intent of "Qrack" is to provide maximum performance for the simulation of an ideal, virtually error-free quantum computer, across the broadest possible set of hardware and operating systems.

Using the C++11 standard, at base, Qrack has an external-dependency-free CPU simulator "engine," as well as a GPU simulator engine that depends only on OpenCL. The `QUnit` layer provides novel, fundamental optimizations in the simulation algorithm, based on "[Schmidt decomposition](https://arxiv.org/abs/1710.05867)," transformation of basis, 2 qubit controlled gate buffer caching, the physical nonobservability of arbitrary global phase factors on a state vector, and many other "synergistic" and incidental points of optimization between these approaches and in addition to them. `QUnit` can be placed "on top" of either CPU, GPU, or hybrid engine types, and an additional `QPager` layer can sit between these, or in place of `QUnit`. Optimizations and hardware support are highly configurable, particularly at build time.

A `QInterface` can be thought of as like simply a one-dimensional array of qubits, within which any qubit has the capacity to directly and fully entangle with any and all others. Bits can be manipulated on by a single bit gate at a time, or gates and higher level quantum instructions can be acted over arbitrary contiguous sets of bits. A qubit start index and a length is specified for parallel operation of gates over bits or for higher level instructions, like arithmetic on arbitrary width registers. Some methods are designed for (bitwise and register-like) interface between quantum and classical bits. See the Doxygen for the purpose of gate-like and register-like functions.

Qrack was originally integrated with a [MOS 6502 emulator](https://github.com/vm6502q/vm6502q), demonstrating its originally intended user for developing chip-like quantum computer emulators. (The base 6502 emulator to which Qrack was added for that project is by Marek Karcz, many thanks to Marek! See https://github.com/makarcz/vm6502.)

Qrack compiles like a library. To include in your project:

1. In your source code:

```cpp
#include "qrack/qfactory.hpp"
```

2. On the command line, in the project directory

```sh
$ mkdir _build && cd _build && cmake .. && make all install
```

For more information, compile the `doxygen.config` in the root folder, and then check the `doc` folder.

## Documentation

Live version of the documentation, including API reference, can be obtained at: https://qrack.readthedocs.io/en/latest/

## Community

Qrack has a community home at the Advanced Computing Topics server on Discord, at: https://discordapp.com/invite/Gj3CHDy

For help getting started with contributing, see our [CONTRIBUTING.md](CONTRIBUTING.md).

## Quick start

**Firstly, are you looking for the applications promised by quantum computing?** Do you want to run tens of thousands of nodes or more of _transverse field Ising model (TFIM),_ or _quadratic unconstrained binary optimization (QUBO)_ in the form of _MAXCUT,_ or the _Traveling Salesman Problem (TSP),_ at quality and scaling performance beyond _quantum annealers,_ _VQE,_ and _QAOA,_ at the scale of a consumer laptop, with _"quantum-inspired"_ heuristics? Then, **Qrack** will do a lot of things, but you likely want [PyQrackIsing](https://github.com/vm6502q/PyQrackIsing) for that. Otherwise, we'll tell you how to get started with **Qrack**.

To get up-and-running most easily with maximum performance, if Python is the appropriate language for what you want to do, just `pip install pyqrack`. (We also offer the [Qook](https://crates.io/crates/qook/0.1.1/dependencies) Rust language bindings.) That's it, basically! At minimum, you could stop here: it should _just work._

If Qrack tells you, on running any script or program with Qrack, its "default device" is the not the device you actually want to use (i.e. you have a NVIDIA GPU, but Qrack selects an Intel HD device instead) [change the default device environment variable setting](https://github.com/unitaryfund/qrack?tab=readme-ov-file#changing-default-opencl-device). If there are any devices at all that you don't want to use in the list Qrack prints whenever you run, this implies that you also needs to set manual device lists for `QPager` and `QUnitMulti` (which is explained below in this README). This won't remove them from Qrack's list of detected devices, but it will exclude them from being used. (Qrack will attempt to use all available devices by default, which can actually hurt performance.) If there's only one device you want to use at all (like a single NVIDIA GPU, for example), find its ID in the list Qrack prints at run time, and, if that is ID "`0`," for example, set environment variables `QRACK_QPAGER_DEVICES=0` and `QRACK_QUNITMULTI_DEVICES=0`.

If you just have a single NVIDIA GPU, but Qrack detects other devices that you don't even want to bother with, you could uninstall all other versions of Qrack and then `pip install pyqrack-cuda`, but note that OpenCL performance tends to be better than CUDA (despite erroneous "conventional wisdom," even though the CUDA and OpenCL APIs are almost line-by-line equivalent in terms of their implementations in the Qrack library internals).

You could probably squeeze out more performance by following the "power user considerations," and other special considerations might be necessary for approximate simulation techniques or custom builds.

## Power user considerations

1. Run a preliminary round of CPU-only benchmarks (such as on the [quantum Fourier transform algorithm](https://github.com/vm6502q/pyqrack-examples/blob/main/qft.py)) to decide how to **tune the [`PSTRIDEPOW`](https://github.com/unitaryfund/qrack?tab=readme-ov-file#build-and-environment-options-for-cpu-engines) environment variable for best performance** (by trial-and-error comparison of timing results). This can make a _big_ difference to CPU-based simulation performance, and be aware that CPU-based and GPU-based algorithms and implementations in Qrack work **together** to return the best possible performance for even GPU-based benchmarks.
2. Try ["precompiling" your OpenCL kernels](https://github.com/unitaryfund/qrack?tab=readme-ov-file#precompiled-opencl-kernels), to skip the OpenCL JIT compilation step, every time you run. (This doesn't work on all systems.)
2. Consider [tuning light-cone optimization qubit threshold](https://github.com/unitaryfund/qrack?tab=readme-ov-file#qtensornetwork-options) and [setting manual system resource limit variables](https://github.com/unitaryfund/qrack?tab=readme-ov-file#maximum-allocation-guard) (though these are not strictly required and might be recognized accurately by default by Qrack).
3. Set `QRACK_DISABLE_QUNIT_FIDELITY_GUARD=1` as environment variable to disengage Qrack's fidelity estimate "limiter," if using approximate simulation. (See the README section on [approximate simulation](https://github.com/unitaryfoundation/qrack?tab=readme-ov-file#approximation-and-noise-options), first.)

**Want to challenge _"quantum supremacy"_ and _"quantum advantage"_ benchmarks from hardware quantum computers, on your laptop?**
- **For hard optimization problems**, like MAXCUT, Traveling Salesman Problem, and transverse field ising model (TFIM), see [PyQrackIsing](https://github.com/vm6502q/PyQrackIsing).
- **For random circuit sampling (RCS) cross-entropy benchmark (XEB)** fidelity experiments, like Google's Sycamore, Willow, and related benchmarks, you want _automatic circuit elision (ACE)_ approximation. Set environment variable `QRACK_DISABLE_QUNIT_FIDELITY_GUARD=1` to turn off the "limiter" for first-principles fidelity estimate, because the estimate becomes grossly pessimistic for _anti-concentration_ in RCS. ACE with "Schmidt decomposition" (`QUnit`) will break the simulation down into separable subsystems limited to `m` qubits for (environment variable setting) `QRACK_MAX_PAGING_QB=m` (or `QRACK_MAX_CPU_QB=m`), so you might set subsystem size to exactly half, or a quarter, or some small, simple fraction of your total count of qubits. For a 2D square lattice with 4 subsystems (so `QRACK_MAX_PAGING_QB=m` with overall qubit width of `4*m` for the benchmark), like in the [PyQrack RCS examples](https://github.com/vm6502q/pyqrack-examples/tree/main/rcs), you might expect **roughly 50%** XEB fidelity for large circuits (almost entirely independent of circuit width or depth, seemingly). The only situations where this seems not to work is with **highly symmetric** geometries, like _perfect-square_ 2D layouts (4-by-4 qubits, 5-by-5, 6-by-6, etc.) of _fully-connected geometries._
- **For these remaining hard symmetric cases**, you likely want to experiment with a mixture of _ACE_ and _sparse truncation._ The sparse truncation approximation features are currently CPU-only, but they act somewhat similarly to _matrix product state_ simulation by retaining only a (sparse) set of probability amplitudes with the highest relative probabilities, retaining much of the fidelity of the ideal circuit. Environment variable `QRACK_SPARSE_TRUNCATION_THRESHOLD` sets the minimum probability threshold below which amplitudes will be ("proactively") discarded to save space, and [we suggest](https://github.com/vm6502q/pyqrack-examples/blob/main/rcs/fc_qiskit_validation_sparse.sh) a threshold like `1/(z*(2**n))` where "`n`" is the total number of qubits and "`z`" is the number of other qubits connected to any one qubit (so `z=n-1` for _fully-connected_ or `z=4` for a square lattice). Set environment variable `QRACK_SPARSE_MAX_ALLOC_MB` to ("reactively") cap the maximum allocation footprint of any single sparse simulator. (You might get surprisingly high XEB with very low allocation limits, which is helpful, since CPU-based sparse methods can be slow with large maximum allocation limits).
- **For even better RCS XEB results**, see [this script](https://github.com/vm6502q/pyqrack-examples/blob/main/rcs/fc_tn_qiskit_validation.py). The idea is to use _ACE_ and _sparse truncation_ to point to where an (ideal) _tensor network tree simulator_ like `quimb` should _sieve_ for a small number of amplitudes that are very likely to be _"heavy,"_ with high relative probability weight in the overall distribution. Find about `n^2`-or-so count of heavy amplitudes for `n` count of qubits, then renormalize the subset and superpose with _maximal-entropy mean field,_ just like the script does.
- Similarly, these same techniques are applicable to **[OTOC](https://github.com/vm6502q/pyqrack-examples/tree/main/otoc)** _"quantum advantage"_ experiments based on either _TFIM_ or _RCS._

## Intel HD (or integrated graphics) power user considerations

No GPU? No problem! Qrack supports (and loves) "integrated graphics coprocessors" like the Intel HD, not just "discrete" GPUs! Use the OpenCL version of Qrack or PyQrack, and it just works!

In fact, there's a certain (limited) advantage to many "integrated graphics" accelerators like the Intel HD: they often rely on _general_ memory rather than _dedicated_ memory. If you explicitly tell Qrack you want to use general memory, you might see a significant performance boost! If all you have for an accelerator is an Intel HD (or maybe another type of "integrated graphics"), try setting this option (to enable "zero-copy" mode): `QRACK_QPAGER_DEVICES_HOST_POINTER=1`

For PyQrack, you'll also want to set environment variable `PYQRACK_HOST_POINTER_DEFAULT_ON=1`, which changes the default for `isHostPointer` option in the constructor of the `QrackSimulator` class to `True`. (You can, instead, control this on a case-by-case basis, per simulator, by passing `isHostPointer=True` to `QrackSimulator` on initialization.)

## Changing default OpenCL device
OpenCL device(s) can be specified by index in `Qrack::QInterface` subclass constructors. The global default device can also be overridden with the environment variable `QRACK_OCL_DEFAULT_DEVICE=n`, where `n` is the index of the OpenCL device you want to use, as reported by the OpenCL initialization header.

## Build and environment options for CPU engines
`QEngineCPU` and `QHybrid` batch work items in groups of 2^`PSTRIDEPOW` before dispatching them to single CPU threads, potentially greatly reducing waiting on mutexes without signficantly hurting utilization and scheduling. The default for this option can be controlled at build time, by passing `-DPSTRIDEPOW=n` to CMake, with "n" being an integer greater than or equal to 0. This can be overridden at run time by the enviroment variable `QRACK_PSTRIDEPOW=n`. If an environment variable is not defined for this option, the default from CMake build will be used. (The default is meant to work well across different typical consumer systems, but it can benefit greatly from system-tailored tuning via the environment variable. **This can be critical to performance of CPU-based simulation**, so you should _always_ tune it when deploying Qrack on a new system.)

## Precompiled OpenCL kernels

```sh
$ qrack_cl_compile [path]
```
Precompile the OpenCL programs for all available devices, and save them to the optional "path" parameter location. By default, programs will be saved to a folder in the "home" directory, such as `~/.qrack/` on most Linux systems. (The default path can also be specified as an environment variable, `QRACK_OCL_PATH`.) Also by default, Qrack will attempt to load precompiled binaries from the same path, but the library will fall back to JIT compilation if program binaries are not available or are corrupt. To turn off default loading of binaries, one can simply delete the programs from this folder.

## QTensorNetwork options
`QTensorNetwork` has a qubit threshold up to which it is able to reuse more work in measurement and probability calculations, controlled by environment variable `QRACK_QTENSORNETWORK_THRESHOLD_QB`. By default, this is the same as qubit width that Qrack detects to be the maximum for state vector simulation (environment variables `QRACK_MAX_PAGING_QB` or `QRACK_MAX_CPU_QB`). Above (and not including) this threshold, `QTensorNetwork` will use techniques like restricting to "past light cones" for measurement and probablity calculation, in an attempt to reduce overall memory footprint at the cost of additional execution time.

## Maximum allocation guard
Set the maximum allowed allocation (in MB) for the global OpenCL pool with environment variable `QRACK_MAX_ALLOC_MB`. Per OpenCL device, this sets each maximum allocation limit with the same syntax as `QRACK_QPAGER_DEVICES`. (Succesive entries in the list are MB limits numbered according to Qrack's device IDs print-out on launch.) This limit includes (VRAM) state vectors and auxiliary buffers. This should also include out-of-place single duplication of any state vector. This does **not** include non-OpenCL general heap or stack allocation.

`QRACK_MAX_PAGING_QB` and `QRACK_MAX_CPU_QB` environment variables set a maximum on how many qubits can be allocated on a single `QPager` or `QEngineCPU` instance, respectively. This qubit limit is for maximum single `QPager` or `QEngineCPU` allocation, whereas `QUnit` and `QUnitMulti` might allocate _more_ qubits than this as separable subsystems, while requiring that no individual separable subsystem exceeds the qubit limit environment variables. (`QEngineOCL` limits are automatically maximal according to a query Qrack makes of maximum allocation segment on a given OpenCL device.)

`QRACK_QBDT_MAX_ALLOC_MB` controls the **total process memory** limit before `QBdt` (quantum binary decision tree) instances will `throw std::bad_alloc()`. This is particularly useful in combination with `QUnit` simulation layer over `QBdt`, which will automatically elide any gates that cross the allocation limit at first point of failure, without relying the operating system to raise `std::bad_alloc()` due to heap memory limits, which might work, but cannot be guaranteed to ultimately prevent the process from being stopped by the operating system.

`QRACK_SPARSE_MAX_ALLOC_MB` controls the maximum peak memory usage (roughly but very closely) before _sparse simulation_ in `QEngineCPU` begins to _approximately truncate._ For a worst-case fidelity estimate, when simulating with the sparse option at close to the limits of the this value, you should check `QInterface::GetUnitaryFidelity()` on your instances to know how much fidelity could have been lost.

## Near-Clifford simulation

Qrack has a novel near-Clifford simulation method based on the "reverse gadget" for non-Clifford phase injection in Appendix A of [PRX Quantum 3, 020361 – Published 23 June 2022](https://journals.aps.org/prxquantum/abstract/10.1103/PRXQuantum.3.020361). To use it, simply restrict your gate set to Clifford gates and only `RZ` for a non-Clifford gate. You probably also want to disable `QTensorNetwork` and `QUnit` layers, to directly expose `QStabilizerHybrid` at the "top" of your simulation method "layer stack."

## Approximation and noise options
`QUnit` can optionally round qubit subsystems proactively or on-demand to the nearest single or double qubit eigenstate with the `QRACK_QUNIT_SEPARABILITY_THRESHOLD=[0.0 - 1.0]` environment variable, with a value between `0.0` and `1.0`. When trying to find separable subsystems, Qrack will start by making 3-axis (independent or conditional) probability measurements. Based on the probability measurements, under the assumption that the state _is_ separable, an inverse state preparation to |0> procedure is fixed. If inverse state preparation would bring any single qubit Bloch sphere projection within parameter range of the edge of the Bloch sphere (with unit length, `1.0`), then the subsystem will be rounded (i.e. **post-selected** for measurement) to that state, then "uncomputed" with the corresponding (forward) state preparation, effectively "hyperpolarizing" one and two qubit separable substates by replacing entanglement with local qubit Bloch sphere extent. (If 3-axis probability is _not_ within rounding range, nothing is done directly to the substate.)

Similar functionality to above is available for `QBdt` with `QRACK_QBDT_SEPARABILITY_THRESHOLD=[0.0 - 0.5]`. In the case of this parameter, any branch with less than the parameter value for probability is rounded to 0, and its partner branch is renormalized to unit length. This same value is also used for branch equality comparison.

Environment variable `QRACK_NONCLIFFORD_ROUNDING_THRESHOLD` sets the non-Clifford phase gate magnitude, as a fraction of `T` gate phase angle (from the closest Clifford state Bloch sphere orientation), that will be rounded to 0 in terminal measurement and sampling operations. (For `0`/default value, all non-Clifford phase gates are exactly preserved.)

Environment variable `QRACK_SPARSE_TRUNCATION_THRESHOLD` controls the threshold of probability below which amplitudes are truncated during any `Compose()`-type operation under `QUnit` is sparse (CPU) simulation is in use. During a `Compose()` operation, if an amplitude multiplies out to a value with a norm below this threshold, it will be truncated to `0` (to save memory), and the first-principles, worst-case fidelity estimate will be updated to reflect the lost fidelity. For random circuit sampling (RCS), we recommend a value of `1 / (z * 2^n)` for this variable, where `n` is the overall qubit count and `z` is the (average) number of connections from one qubit to others (like `n - 1` for full-connectivity benchmarks).

For single-qubit depolarizing noise channels, `QInterfaceNoisy` exposes environment variable `QRACK_GATE_DEPOLARIZATION`, which is a per-gate, single-qubit (applied to all qubits in every gate) noise parameter for depolarizing noise, typically taking a floating-point value between `0` and `1`.

When `QUnit` encounters a situation where acting a coupler gate would require exceeding user-specified maximum system memory resources, it will automatically resort to replacing this gate with a "classical shadow" that will not require additional allocation. (We call this **"automatic circuit elision," "ACE."**) In the process, `QUnit` will decrease its internal first-principles unitary fidelity estimation under assumption that the classical shadow replacement effect is worst-case. If the fidelity estimate drops below floating-point rounding epsilon, `QUnit` will automatically `throw` to exit simulation early. However, particularly in cases of "anti-concentration," such as with universal random circuit sampling (or, alternatively, when qubits tend to be **very** polarized, hence the classical shadow replacement should work better), this first-principles worst-case fidelity estimate might be very overly severe. In these cases, the worst-case fidelity estimate cannot necessarily be trusted not to be overly pessimistic, but the `throw` condition on low fidelity estimate can be disabled by assigning any value to environment variable `QRACK_DISABLE_QUNIT_FIDELITY_GUARD`. **If you intend to work with ACE subystems that should be half of the overall circuit width, for example, you probably also want to set `QRACK_MAX_PAGING_QB` or `QRACK_MAX_CPU_QB` appropriately, so that your qubit subsystem limit is half (rounded-up) of the actual overall qubit width, or correspondingly by analogy for 4 subsystems, 8 subsystems, etc..**  

## Installing Qrack

If you're on Ubuntu 18.04, 20.04, 22.04, or 24.04 LTS, you're in luck: Qrack manages a PPA that provides binary installers for _all_ available CPU architectures (except any that require administrative attention from Ubuntu or Canonical).

```sh
    $ sudo add-apt-repository ppa:wrathfulspatula/vm6502q
    $ sudo apt update
    $ sudo apt install libqrack-dev
```

You might need to install the `add-apt-repository` tool first, through `apt` itself. There is also a `pyqrack` package on the PPA, or you can get PyQrack from PyPi with `pip install pyqrack`.

Otherwise, standardized builds are available on the [releases](https://github.com/unitaryfund/qrack/releases) page. (Operating system targets include Linux, Windows, and Mac, alongside WebAssmembly. Qrack source also builds for native Android and iOS.)

## Installing OpenCL on Mac

While the OpenCL framework is available by default on most modern Macs, the C++ header `cl.hpp` is usually not. One option for building for OpenCL on Mac is to download this header file and include it in the Qrack project folder under include/OpenCL (as `cl.hpp`). The OpenCL C++ header can be found at the Khronos OpenCL registry:

https://www.khronos.org/registry/OpenCL/

Otherwise, Homebrew offers a package with the headers: [opencl-clhpp-headers](https://formulae.brew.sh/formula/opencl-clhpp-headers) is the preferred method of installing headers, if `brew` is available.

## Building and Installing Qrack on Windows

Qrack supports building on Windows, but some special configuration is required. Windows 10 usually comes with default OpenCL libraries for Intel (or AMD) CPUs and their graphics coprocessors, but NVIDIA graphics card support might require the CUDA Toolkit. The CUDA Toolkit also provides an OpenCL development environment, which is generally necessary to build Qrack.

Qrack requires the `xxd` command to convert its OpenCL kernel code into hexadecimal format for building. `xxd` is not natively available on Windows systems, but Windows executables for it are provided by sources including the [Vim editor Windows port](https://www.vim.org/download.php).

```sh
    $ mkdir _build
    $ cd _build
    $ cmake -DXXD_BIN="C:/Program Files (x86)/Vim/vim82/xxd.exe" ..
```

After CMake, the project must be built in Visual Studio. Once installed, the `qrack_pinvoke` DLL is compatible with the Qrack Q# runtime fork, to provide `QrackSimulator`.

## test/tests.cpp

The included `test/tests.cpp` contains unit tests and usage examples. The unittests themselves can be executed:

```sh
    $ _build/unittest
```

Similarly, benchmarks are in `test/benchmarks.cpp`:

```sh
    $ _build/benchmarks [--optimal] [--max-qubits=30] [test_qft_cosmology]
```

## Performing code coverage

```sh
    $ cd _build
    $ cmake -DENABLE_CODECOVERAGE=ON ..
    $ make -j 8 unittest
    $ ./unittest
    $ make coverage
    $ cd coverage_results
    $ python -m http.server
```

## OpenCL on systems prior to OpenCL v2.0

Particularly on older hardware, it is possible that you do not have OpenCL v2.0 available. In theory, Qrack should work off-the-shelf anyway. However, if the OpenCL implementation isn't even aware of the existence of v2.0, use the following option to completely manually force all v2.0 functionality off (of which, Qrack uses only "out-of-order" queueing, hence "OOO") and to set the target OpenCL API level expressly to target v1.2 and minimum level v1.1:

```sh
    $ cmake -DENABLE_OOO_OCL=OFF ..
```

If you're using a pre-packaged build artifact, like in PyQrack, set environment variable "`DISABLE_OOO_OCL`" to any value instead:

```sh
    $ export DISABLE_OOO_OCL=1
```

## C++ language standard

To change the C++ language standard language with which Qrack is compiled, use `-DCPP_STD=n`, where "`n`" is the two-digit standard year:

```sh
cmake -DCPP_STD=11 ..
```

Qrack is backwards compatible as far as C++11. By default, Qrack builds for C++14. For minimum support for all optional dependencies, C++14 or later might be required.

## Optional CUDA instead of OpenCL

Theoretically, building with CUDA for your native supported architectures is as simple as installing the CUDA toolkit and compiler and using this CMake command:

```sh
cmake -DENABLE_CUDA=ON [-DENABLE_OPENCL=OFF] [-DQRACK_CUDA_ARCHITECTURES=86] ..
```

where `-DENABLE_CUDA=ON` is required to enable CUDA, `-DENABLE_OPENCL=OFF` will cause CUDA to be used in the default optimal simulation layer stack instead of OpenCL, and `-DQRACK_CUDA_ARCHITECTURES` optionally specifies an explicit list of CUDA architectures for which to build. (If `-DQRACK_CUDA_ARCHITECTURES` is not set, Qrack will attempt to detect your native installed GPU architectures and build for exactly that set.)

## WebAssembly (WASM) builds

By nature of its pure C++11 design, Qrack happens to offer excellent compatibility with Emscripten ("WebAssembly") projects. See [the qrack.net repository](https://github.com/vm6502q/qrack.net) for an example and [qrack.net](https://qrack.net) for a live demo. OpenCL GPU operation is not yet available for WASM builds. While CPU multithreading might be possible in WASM, it is advisable that `pthread` usage and linking is disabled for most conventional Web applications, with `-DENABLE_PTHREAD=OFF` in CMake:

```sh
emcmake cmake -DENABLE_RDRAND=OFF -DENABLE_PTHREAD=OFF -DSEED_DEVRAND=OFF -DUINTPOW=5 ..
```

`-DUINTPOW=5` is optional, but WASM RAM limitations currently preclude >=32 qubits of potentially entangled state vector, so 64 bit ket addressing is not necessary. However, `-DQBCAPPOW=10` could be added to the above to support high-width stabilizer and Schmidt decomposition cases, with the appropriate build of the Boost headers for the toolchain.

## Tune OpenCL preferred concurrency
Preferred concurrency has a tunable offset with default value of `3`, with the environment variable setting `export QRACK_GPU_OFFSET_QB=[m]` for some (positive or negative) integer `m`. For each integer increment of `m`, the preferred concurrency is multiplied by 2. (Preferred concurrency is calculated as `pow2(ceil(log2(([GPU processing element count] * [preferred group size for the single qubit gate kernel, usually warp size])))) << QRACK_GPU_OFFSET_QB`.)

## QPager distributed simulation options
`QPager` attempts to intelligently allocate low qubit widths for maximum performance. For heterogeneous GPU simulation, based on `clinfo`, you can set a ceiling on your maximum OpenCL accelerator state vector page allocation with the environment variable `QRACK_MAX_PAGE_QB=n`, where n is an integer >=0. The default `n` is max integer, meaning that maximum allocation segment of your GPU RAM is always a single hardware page.

To set a maximum on how many qubits can be allocated on a single `QPager` instance, use the environment variable `QRACK_MAX_PAGING_QB`, for example, `export QRACK_MAX_PAGING_QB=30` to cause `QPager` to throw an exception that can be caught if it is asked to allocate 31 or more qubits. 

To set the `QPager` device ID list, use the `QRACK_QPAGER_DEVICES` environment variable. This variable should contain an ordered list of Qrack OpenCL device IDs that should be automatically used in all `QPager` instances. Note that device IDs may be included multiple times in this list in order to achieve a simple form of load balancing. For example, since NVIDIA GPUs typically have 4 maximum allocation segments, a device list like `1,1,1,1,0,0,0,0` will allocate the first 4 maximum allocation segments on device `1` first, such that device `1` will be (roughly) 100% utilized before including any segments on device `0`. This list can also be written `4.1,4.0`, which means that 4 segments of `1` should be repeated before 4 segments of `0` are repeated. To repeat a pattern of multiple IDs, follow the multiplier with multiple `.` characters separating every ID in the pattern, like `4.1.0,4.2` for 4 repetitions of `1,0` followed by 4 repetitions of `2`. If device IDs are exhausted in the device list, `QPager` will automatically cycle the list as many times as it needs, to attempt higher segment count allocation.

There are two special device IDs that can be specified in these lists: `-1` is global Qrack default device. `-2` indicates that `QInterface`-local constructor-specified device ID should be used. (For example, a device list argument of just `-2` will indicate that distribution choices should defer to those of `QUnitMulti`, if in use.)

`QRACK_QPAGER_DEVICES_HOST_POINTER` corresponds to each device ID in `QRACK_QPAGER_DEVICES`, per sequential item in that other variable, (with the same syntax and list wrapping behavior). If the value of this is `0` for a page, that page attempts OpenCL _device_ RAM allocation; if the value is `1` for a page, that page attempts OpenCL _host_ RAM allocation. `0` value, device RAM, is suggested for GPUs; `1` value, host RAM, is suggested for CPUs and APUs (which use general host RAM, anyway). By default, all devices attempt on-device RAM allocation, if this environment variable is not specified.

## QUnitMulti device list
Specify a device list for `QUnitMulti` the same way you would for `QPager`, with environment variable `QRACK_QUNITMULTI_DEVICES`. Corresponding to each entry in `QRACK_QUNITMULTI_DEVICES`, use `QRACK_QUNITMULTI_DEVICES_MAX_QB` to (optionally) specify a per-entry ceiling on device usage. For smaller-width devices like CPUs, it might make sense to set the qubit ceiling to about the CPU `PSTRIDEPOW` plus logarithm base 2 of your hyperthread count.

## QBdt and QBdtHybrid options
`QBdtHybrid` sets a threshold for "hybridization" between "quantum binary decision diagrams" (see Acknowledgements at bottom of document) and state vector simulation, based on how efficiently the "diagram" or "tree" can be "compressed." The environment variable `QRACK_QBDT_HYBRID_THRESHOLD` (typically taking values between 0 and 1) sets a multiplicative fraction for maximally-compressed size of the tree, as fraction of node count vs. equivalent state vector amplitude count, before switching over to state vector simulation. Note that maximum `QBdt` node count is _twice_ the count of amplitudes in the equivalent state vector simulation, so set the variable to 2 or higher to completely suppress switching and recover `QBdt`-only simulation in all cases.

## Enable OpenCL device redistribution
Setting the environment variable `QRACK_ENABLE_QUNITMULTI_REDISTRIBUTE` to any value except a null string enables reactive load redistribution or balancing, in `QUnitMulti`. Otherwise, `QUnitMulti` only tries to balance load as opportunity arises when new separable `QEngineShard` instances are created.

## Vectorization optimization

```sh
$ cmake -DENABLE_COMPLEX_X2=ON ..
```
Multiply complex numbers two at a time instead of one at a time. Requires AVX for double and SSE 1.0 (with optional SSE 3.0) for float. On by default, but can be turned off for double accuracy without the AVX requirement, or to completely remove vectorization with single float accuracy.

If `-DENABLE_COMPLEX_X2=ON`, then SSE 3.0 is used by default (for 32-bit and 64-bit floating-point builds), AVX2 is used for 64-bit floating-point builds, and FMA instruction is emitted. For cross-platform compatibility of newer binaries to older `x86_64` targets, each of these can be separately toggled on-or-off:

```sh
$ cmake -DENABLE_SSE3=OFF -DENABLE_AVX=OFF -DENABLE_FMA=OFF ..
```
## Random number generation options (on-chip by default)

```sh
$ cmake -DENABLE_RDRAND=OFF ..
```
Turn off the option to attempt using on-chip hardware random number generation, which is on by default. If the option is on, Qrack might still compile to attempt using hardware random number generation, but fall back to software generation if the RDRAND opcode is not actually available. Some systems' compilers, such as that of the Raspberry Pi 3, do not recognize the compilation flag for enabling RDRAND, in which case this option needs to be turned off.

```sh
$ cmake [-DENABLE_RDRAND=OFF] -DENABLE_DEVRAND=ON ..
```
Instead of RDRAND, use Linux `/dev/urandom/` as the Qrack random number source. (The necessary system call will only be available on Linux systems.)

```sh
$ cmake -DSEED_DEVRAND=OFF ..
```
If pure software pseudo-random number generator is used, it will be seeded from `/dev/random` by default. `-DSEED_DEVRAND=OFF` will use the system clock for Mersenne twister seeding, instead of `/dev/random`.

If you prefer to "bring your own RNG," from any source, then save it as a binary bit string file or files in `~/.qrack/rng/` (or override that path with environment variable `QRACK_RNG_PATH`), and Qrack will use these for random bit streams, in lexigraphical order of file name in the directory, with the following CMake options:

```sh
$ cmake -DENABLE_RNDFILE=ON [-DENABLE_RDRAND=OFF] [-DENABLE_DEVRAND=OFF] ..
```

Also see [scripts/qrng.py](https://github.com/unitaryfund/qrack/blob/main/scripts/qrng.py) for an example of how one could request quantum random bit strings from [https://quantumnumbers.anu.edu.au](https://quantumnumbers.anu.edu.au) and use them as a source of random numbers for Qrack.

## Reduced or increased coherent qubit addressing

```sh
$ cmake [-DUINTPOW=n] [-DQBCAPPOW=n] ..
```
Qrack uses an unsigned integer primitive for ubiquitous qubit masking operations, for "local" qubits (`QEngine`) and "global" qubits (`QUnit` and `QPager`). This limits the maximum qubit capacity of any coherent `QInterface` to the total number of bits in the global (or local) masking type. By default, a 64-bit unsigned integer is used, corresponding to a maximum of 64 qubits in any coherent `QInterface` (if attainable, such as in limited cases with `QUnit`). `-DUINTPOW=n` reduces the "local" masking type to 2^n bits (ex.: for max OpenCL sub-unit or page qubit width), which might also be important with accelerators that might not support 64-bit types. `-DQBCAPPOW=n` sets the maximum power of "global" qubits in "paged" or `QUnit` types as potentially larger than single "pages" or "sub-units," for "n" >= 5, with n=5 being 2^5=32 qubits. Large "n" is possible with the Boost big integer header. (Setting "n" the same for both build options can avoid casting between "subunit" and "global qubit" masking types, if larger "paging" or `QUnit` widths than `QEngine` types are not needed.)

## Variable floating point precision

```sh
$ cmake [-DFPPOW=n] ..
```
Like for unsigned integer masking types, this sets the floating point accuracy for state vectors to 2^n. By default, n=5, for 2^5=32 bit floating point precision per real or imaginary part. "half," "double," and "quad," availability depend on the system, but n=6 for "double" is commonly supported on modern hardware. n=4 for half is supported by GCC on ARM, header-only on x86_64, and by device pragma if available for OpenCL kernels. "quad" is supported on CPU only, if available.

## Turn on/off optional API components

```sh
$ cmake -DENABLE_BCD=OFF -DENABLE_REG_GATES=OFF -DENABLE_ROT_API=OFF -DENABLE_ALU=ON ..
```

Prior to the Qrack v7 API, a larger set of convenience methods were included in all builds, which increased the size of the library binary. By default, `ENABLE_REG_GATES`, `ENABLE_ROT_API` and  `ENABLE_BCD` all default to `OFF`, while `ENABLE_ALU` for arithmetic logic unit methods defaults to `ON`.

`ENABLE_REG_GATES` adds various looped-over-width gates to the API, like the lengthwise `CNOT(control, target, length)` method. This method is a convenience wrapper on a loop of `CNOT` operations for `length`, starting from `control` and `target`, to `control + length - 1` and `target + length - 1`. These methods were seen as opportunities for optimization, at a much earlier point, but they have fallen out of internal use, and basically none of them are optimized as special cases, anymore. Disabling `ENABLE_REG_GATES` does **not** remove lengthwise `X(target, length)` and `H(target, length)` methods, as these specific convenience methods are still commonly used in the protected API, for negating or superposing across register width.

`ENABLE_ROT_API` adds many less common **rotation** methods to the API, like dyadic fraction rotations. These never found common use in the protected API, while they add significant size to compiled binaries.

"BCD" arithmetic ("binary coded decimal") is necessary to support emulation based on the MOS-6502. However, this is an outmoded form of binary arithmetic for most or all conceivable purposes for which one would want a quantum computer. (It stores integers as base 10 digits, in binary.) Off by default, turning this option on will slightly increase binary size by including BCD ALU operations from the API, but this is necessary to support the VM6502Q chip-like emulator project.

## Copyright, License, and Acknowledgements

Copyright (c) Daniel Strano and the Qrack contributors 2017-2025. All rights reserved.

Daniel Strano would like to specifically note that Benn Bollay is almost entirely responsible for the initial implementation of `QUnit` and tooling, including unit tests, in addition to large amounts of work on the documentation and many other various contributions in intensive reviews. Special thanks go to Aryan Blaauw for his extensive systematic benchmark program, leading to much debugging and design feedback, while he spreads general good will about our community discussion space. Also, thank you to Marek Karcz for supplying an awesome base classical 6502 emulator for proof-of-concept. For unit tests and benchmarks, Qrack uses Catch v2.13.7 under the Boost Software License, Version 1.0. The `QStabilizer` partial simulator "engine" is adapted from CHP by Scott Aaronson, for non-commercial use. `QBdt` is Qrack's "hand-rolled" take on "quantum binary decision diagrams" ("QBDD," or "quantum binary decision trees") inspired largely by a talk Dan attended from Jülich Supercomputing Center at IEEE Quantum Week, in 2021, and later followed up with reading into work of authors including Robert Wille. The half precision floating point header is provided by [http://half.sourceforge.net/](http://half.sourceforge.net/), and the fixed-point header is provided by [https://github.com/eteran/cpp-utilities](https://github.com/eteran/cpp-utilities), with our thanks and appreciation. GitHub user [paniash](https://github.com/paniash) has kindly contributed `README.md` styling and standardization. Some commits might be written with the assistance of OpenAI's ChatGPT, though the commit messages should note all such specific cases, and 0 commits used direct ChatGPT assistance or any direct AI assistance for authorship before April 15, 2023. Thank you to all our PR contributors, tracked in GitHub, and thank you to the OSS community in general for supporting code, including [Adam Kelly](https://github.com/libtangle/qcgpu) and the [qulacs team](https://github.com/qulacs/qulacs), for Qiskit and Cirq interfaces. (Additionally, the font for the Qrack logo is "Electrickle," distributed as "Freeware" from [https://www.fontspace.com/fontastic/electrickle](https://www.fontspace.com/fontastic/electrickle).)

We thank the Unitary Foundation for its generous support, in a project to help standardize benchmarks across quantum computer simulator software!  Thank you to any and all contributors!

Licensed under the GNU Lesser General Public License V3.

See [LICENSE.md](https://github.com/vm6502q/qrack/blob/main/LICENSE.md) in the project root or https://www.gnu.org/licenses/lgpl-3.0.en.html for details.
