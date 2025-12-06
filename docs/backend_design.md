# Backend Module Detailed Design (Traits & Responsibilities)

## 0. Goal

**Goal of this document**

> Define a clear and implementation-ready trait layer for the `backend` module of `rust-pyscf-pbc-bands`, encapsulating all interactions with CubeCL (CPU/WGPU), including linear algebra (via `cubecl-matmul`), 3D FFT (via Cooley–Tukey), reductions, and memory management.
> The design must:
>
> * Hide CubeCL details from SCF/Band/Cell modules.
> * Support both CPU and WGPU backends.
> * Be testable and extensible (e.g., later CUDA backends).
> * Explicitly instruct implementers to use **docs rs mcp server**, **Obsidian mcp server**, and the **`./vendor`** directory for documentation and examples.
> * Require that backend code undergoes **at least three review-and-improvement iterations**.

---

## 1. Role and Scope of the `backend` Module

The `backend` module is the sole integration point between the high-level electronic-structure logic (Cell, SCF, Band, FFTDF) and the lower-level compute infrastructure (CubeCL runtimes, CPU/WGPU devices, matmul/FFT/reduce kernels, and memory pools).

**Responsibilities:**

* Provide abstract traits for:

  * **Linear algebra** (matrix multiplication and eigen-decomposition).
  * **3D FFT** (forward/inverse, Cooley–Tukey).
  * **Reductions** (sum over device buffers).
  * **Memory management** (allocate/free device buffers, report limits).
* Provide concrete implementations for:

  * `CpuBackend` (CubeCL CPU runtime).
  * `WgpuBackend` (CubeCL WGPU runtime).
* Centralize all direct dependencies on:

  * `cubecl-core`
  * `cubecl-wgpu`
  * `cubecl-matmul`
  * `cubecl-reduce`
  * `cubecl-runtime`
* Enforce CubeCL’s type constraints:

  * Kernels operate only on `CubeType`-compatible primitives (`f32`, `f64`, integer types) and their buffers.
  * Complex numbers are passed as **split-complex arrays** (`re: DeviceBuffer<f64>`, `im: DeviceBuffer<f64>`).

**Non-responsibilities:**

* SCF convergence logic, density construction, Fock building, or band structure workflows (handled in `scf` / `band` modules).
* PySCF-compatible I/O; these use `backend` only indirectly via higher-level modules.

---

## 2. Core Abstractions and Type Overview

### 2.1 BackendKind

An enum used at the builder/API level to select a backend:

```rust
pub enum BackendKind {
    Cpu,
    Wgpu,
    // Future: Cuda, Rocm, etc.
}
```

### 2.2 BackendHandle and Runtime Type Parameter

Each concrete backend will wrap a CubeCL runtime and compute client:

```rust
pub trait Backend {
    type Runtime; // e.g., cubecl_cpu::CpuRuntime or cubecl_wgpu::WgpuRuntime

    fn linalg(&self)  -> &dyn BackendLinalg;
    fn fft(&self)     -> &dyn BackendFft;
    fn reduce(&self)  -> &dyn BackendReduce;
    fn memory(&self)  -> &dyn BackendMemory;
    fn info(&self)    -> BackendInfo;
}
```

`BackendInfo` is a small struct with runtime metadata:

```rust
pub struct BackendInfo {
    pub kind: BackendKind,
    pub device_name: String,
    pub max_memory_mb: usize,
}
```

Concrete types:

* `CpuBackend`
* `WgpuBackend`

will provide concrete associated `Runtime` types and hold a `ComputeClient<Self::Runtime>`.

> **Implementation guideline:**
> When unsure about the exact types and initialization patterns (e.g., `ComputeClient`, `WgpuRuntime`, `CpuRuntime`, `TensorHandle`), you **must** consult:
>
> * **docs rs mcp server** for `cubecl-runtime`, `cubecl-matmul`, `cubecl-wgpu`, `cubecl-core`.
> * **Obsidian mcp server** for existing CubeCL sample code and internal notes.
> * The local `./vendor` directory (e.g., `./vendor/cubecl_matmul/examples`) for canonical usage patterns.

---

## 3. Trait: `BackendLinalg`

### 3.1 Purpose

Encapsulate linear algebra operations required by SCF and band structure computations, primarily:

* Dense **matrix multiplication** (GEMM) via `cubecl-matmul`.
* Dense **eigen-decomposition** (for Fock / Hamiltonian diagonalization).

### 3.2 Data Structures

Backend-independent, minimal representations:

```rust
pub struct Matrix {
    pub rows: usize,
    pub cols: usize,
    pub ld: usize, // leading dimension (row-major: ld = cols)
    pub data: DeviceBuffer<f64>,
}

pub struct Vector {
    pub len: usize,
    pub data: DeviceBuffer<f64>,
}
```

`DeviceBuffer<T>` is an opaque handle to device memory (see `BackendMemory`).

### 3.3 Trait Definition

```rust
pub trait BackendLinalg {
    /// Perform C = alpha * A * B + beta * C (row-major GEMM).
    fn matmul(
        &self,
        a: &Matrix,
        b: &Matrix,
        c: &mut Matrix,
        alpha: f64,
        beta: f64,
    ) -> Result<()>;

    /// Solve A * v_i = λ_i * v_i (Hermitian eigen-decomposition).
    /// For v0.1.0, CPU backend may call LAPACK; GPU backend may fall back
    /// to CPU or be unimplemented (return ResourceError / InternalError).
    fn eig_hermitian(
        &self,
        a: &Matrix,
        evals: &mut Vector,
        evecs: &mut Matrix,
    ) -> Result<()>;
}
```

### 3.4 Mapping to `cubecl-matmul`

The `matmul` implementation for WGPU should:

* Wrap `Matrix` buffers into `TensorHandle<WgpuRuntime, f32>` / `f64>` and `MatmulInputHandle`, analogously to the sample:

```rust
use cubecl_matmul::{launch, MatmulInputHandle, Strategy};
use cubecl_runtime::client::ComputeClient;
use cubecl_std::tensor::TensorHandle;

// Pseudocode inside WgpuLinalg::matmul:
fn matmul(...) -> Result<()> {
    let client: &ComputeClient<WgpuRuntime> = self.client();

    // Convert Matrix -> TensorHandle
    let lhs = MatmulInputHandle::Normal(TensorHandle::<WgpuRuntime, f64>::new(
        a.data.handle(),
        vec![a.rows, a.cols],
        a_strides,
    ));
    let rhs = MatmulInputHandle::Normal(TensorHandle::<WgpuRuntime, f64>::new(
        b.data.handle(),
        vec![b.rows, b.cols],
        b_strides,
    ));
    let out = TensorHandle::<WgpuRuntime, f64>::new(
        c.data.handle(),
        vec![c.rows, c.cols],
        c_strides,
    );

    launch::<WgpuRuntime, f64>(&Strategy::Auto, client, lhs, rhs, out)
        .map_err(|e| Error::InternalError { msg: format!("{e:?}") })?;

    Ok(())
}
```

> **IMPORTANT:**
> When implementing this mapping, do **not** guess the exact types or function signatures.
> Instead:
>
> 1. Look up `cubecl_matmul` on **docs rs mcp server**.
> 2. Search for matmul examples in **Obsidian mcp server**.
> 3. Inspect the provided sample and any examples in `./vendor/cubecl-matmul/` to confirm exact usage.

---

## 4. Trait: `BackendFft` (3D Cooley–Tukey FFT)

### 4.1 Purpose

Provide high-level 3D FFT operations required by FFTDF and related routines:

```rust
pub struct ComplexGrid3D {
    pub nx: usize,
    pub ny: usize,
    pub nz: usize,
    pub re: DeviceBuffer<f64>,
    pub im: DeviceBuffer<f64>,
}

pub trait BackendFft {
    /// Forward 3D FFT: real-space -> reciprocal-space.
    /// Uses radix-2 Cooley–Tukey algorithm along x, y, z.
    fn fft3d_forward(&self, grid: &mut ComplexGrid3D) -> Result<()>;

    /// Inverse 3D FFT: reciprocal-space -> real-space.
    /// Must apply 1/(Nx * Ny * Nz) normalization so that
    /// fft3d_inverse(fft3d_forward(f)) ≈ f within floating-point error.
    fn fft3d_inverse(&self, grid: &mut ComplexGrid3D) -> Result<()>;
}
```

### 4.2 Algorithmic Requirements (Cooley–Tukey Radix-2)

For each dimension (x, y, z):

* Let the size along that axis be (N = 2^m).
* Data is conceptually indexed as (f[x,y,z]), but stored in a flat buffer:

  [
  n = (z \cdot N_y + y) \cdot N_x + x
  ]

#### 4.2.1 1D FFT per Axis

For a fixed pair of indices:

* X-axis: fixed (y,z), compute a 1D FFT over x.
* Y-axis: fixed (x,z) or (k_x,z), compute a 1D FFT over y.
* Z-axis: fixed (x,y) or (k_x,k_y), compute a 1D FFT over z.

Each 1D FFT must:

1. Arrange inputs in **bit-reversed order** or access them via bit-reversed indices.
2. For each stage (s = 0..m-1):

   * Block size (L = 2^{s+1}), half-block (H = 2^s), twiddle stride (R = N/L).
   * For all blocks (b) and positions (j):

     * Pair indices: (n_1 = bL + j), (n_2 = n_1 + H).
     * Twiddle index: (k = jR).
     * Angle: (\theta = 2\pi k / N).
     * Twiddle: (W_N^k = \cos\theta - i\sin\theta).
3. Perform the butterfly:

   Let (a = X^{(s)}[n_1]), (b = X^{(s)}[n_2]).
   Split-complex:

   [
   t_{\mathrm{re}} =
   b_{\mathrm{re}}^{(s)}\cos\theta + b_{\mathrm{im}}^{(s)}\sin\theta
   ]
   [
   t_{\mathrm{im}} =
   -b_{\mathrm{re}}^{(s)}\sin\theta + b_{\mathrm{im}}^{(s)}\cos\theta
   ]

   [
   X_{\mathrm{re}}^{(s+1)}[n_1] = a_{\mathrm{re}}^{(s)} + t_{\mathrm{re}},
   \quad
   X_{\mathrm{im}}^{(s+1)}[n_1] = a_{\mathrm{im}}^{(s)} + t_{\mathrm{im}}
   ]

   [
   X_{\mathrm{re}}^{(s+1)}[n_2] = a_{\mathrm{re}}^{(s)} - t_{\mathrm{re}},
   \quad
   X_{\mathrm{im}}^{(s+1)}[n_2] = a_{\mathrm{im}}^{(s)} - t_{\mathrm{im}}
   ]

#### 4.2.2 3D Composition

The full 3D FFT must apply 1D FFTs in three passes:

1. X-pass: for each (y,z) pair.
2. Y-pass: for each (k_x,z) pair.
3. Z-pass: for each (k_x,k_y) pair.

The final result must be mathematically equivalent to the 3D DFT definition. The inverse transform must use the opposite twiddle sign and multiply by (1/(N_x N_y N_z)) to ensure approximate round-trip identity:

[
\text{fft3d_inverse}(\text{fft3d_forward}(f)) \approx f
]

> **Implementation guidance:**
> The actual FFT kernels should be implemented as `#[cube]` CubeCL kernels. When unsure:
>
> * Use **docs rs mcp server** to inspect CubeCL kernel syntax and math routines (e.g., cosine/sine).
> * Search Obsidian via **Obsidian mcp server** for existing FFT or tensor kernels.
> * Inspect `./vendor` directories (e.g., `./vendor/cubecl_core`, `./vendor/cubecl_std`) for reference kernels.

---

## 5. Trait: `BackendReduce`

### 5.1 Purpose

Expose reduction operations implemented with `cubecl-reduce` and/or custom kernels.

### 5.2 Trait Definition

```rust
pub trait BackendReduce {
    /// Sum over all elements in a buffer (device-side reduction).
    fn sum_f64(&self, buf: &DeviceBuffer<f64>) -> Result<f64>;

    /// Future: reductions along an axis over complex grids, norms, etc.
    // fn sum_axis(...);
}
```

Implementation will:

* Launch a `cubecl-reduce` sum kernel on the buffer.
* Read back the result to host.

> When implementing the `sum` kernel:
>
> * Use **docs rs mcp server** to inspect `cubecl-reduce` APIs.
> * Use **Obsidian mcp server** and `./vendor/cubecl-reduce` examples to confirm launch patterns and buffer layout.

---

## 6. Trait: `BackendMemory`

### 6.1 Purpose

Provide a uniform abstraction over memory allocation and device buffer lifetimes.

### 6.2 Types

```rust
pub enum MemoryUsage {
    Persistent,
    Transient,
}

pub struct DeviceBuffer<T> {
    pub handle: cubecl_runtime::client::AllocationHandle,
    pub len: usize,
    pub _marker: std::marker::PhantomData<T>,
}
```

### 6.3 Trait Definition

```rust
pub trait BackendMemory {
    /// Allocate device buffer for len elements of T, possibly using memory pools.
    fn alloc<T>(&self, len: usize, usage: MemoryUsage) -> Result<DeviceBuffer<T>>
    where
        T: CubeType;

    /// Free a previously allocated buffer.
    fn free<T>(&self, buf: DeviceBuffer<T>) -> Result<()>
    where
        T: CubeType;

    /// Return the maximum usable device memory in MB.
    fn max_memory_mb(&self) -> usize;

    /// Estimate memory usage for a given SCF/Band configuration.
    fn estimate_scf_memory_mb(&self, params: &ScfMemoryParams) -> usize;
}
```

`ScfMemoryParams` is a small struct capturing:

* number of k-points
* number of AOs
* number of bands
* FFT mesh dimensions
* etc.

Persistent vs transient usage directs the backend to allocate from appropriate pools (e.g., CubeCL’s persistent storage vs ephemeral allocations).

---

## 7. Error Handling and Result Types

All backend traits use the library’s standard error types:

```rust
use crate::error::{Error, Result};

pub enum Error {
    InputError { msg: String },
    ConvergenceError { msg: String },
    ResourceError { msg: String },
    InternalError { msg: String },
}
```

Backend trait methods must:

* Map underlying CubeCL errors (e.g., from `cubecl-matmul`, `cubecl-reduce`, runtime launch errors) to:

  * `ResourceError` for allocation/limit issues.
  * `InternalError` for unexpected failures.
* Avoid panics for recoverable conditions.

---

## 8. Implementation Sketch: CPU and WGPU Backends

### 8.1 CPU Backend

```rust
pub struct CpuBackend {
    client: ComputeClient<cubecl_cpu::CpuRuntime>,
    linalg: CpuLinalg,
    fft: CpuFft,
    reduce: CpuReduce,
    memory: CpuMemory,
}

impl Backend for CpuBackend {
    type Runtime = cubecl_cpu::CpuRuntime;

    fn linalg(&self)  -> &dyn BackendLinalg { &self.linalg }
    fn fft(&self)     -> &dyn BackendFft    { &self.fft }
    fn reduce(&self)  -> &dyn BackendReduce { &self.reduce }
    fn memory(&self)  -> &dyn BackendMemory { &self.memory }
    fn info(&self)    -> BackendInfo        { /* ... */ }
}
```

For v0.1.0:

* `matmul` may use a CPU-only implementation (e.g., BLAS or a simple loop), with `cubecl-matmul` optionally used for uniformity.
* `eig_hermitian` is expected to use LAPACK or a Rust linear algebra library (not via CubeCL).

### 8.2 WGPU Backend

```rust
pub struct WgpuBackend {
    client: ComputeClient<cubecl_wgpu::WgpuRuntime>,
    linalg: WgpuLinalg,
    fft: WgpuFft,
    reduce: WgpuReduce,
    memory: WgpuMemory,
}

impl Backend for WgpuBackend {
    type Runtime = cubecl_wgpu::WgpuRuntime;

    fn linalg(&self)  -> &dyn BackendLinalg { &self.linalg }
    fn fft(&self)     -> &dyn BackendFft    { &self.fft }
    fn reduce(&self)  -> &dyn BackendReduce { &self.reduce }
    fn memory(&self)  -> &dyn BackendMemory { &self.memory }
    fn info(&self)    -> BackendInfo        { /* ... */ }
}
```

Construction of `WgpuBackend` must follow patterns similar to the sample:

```rust
use cubecl_wgpu::{init_device, init_setup, AutoGraphicsApi, RuntimeOptions, WgpuDevice, WgpuRuntime};

let device = WgpuDevice::DefaultDevice;
let options = RuntimeOptions::default();
let setup = init_setup::<AutoGraphicsApi>(&device, options);
let device = init_device(setup, RuntimeOptions::default());
let client: ComputeClient<WgpuRuntime> = ComputeClient::load(&device);
```

To confirm exact APIs and options, implementers must use:

* **docs rs mcp server** for `cubecl-wgpu` and `cubecl-runtime`.
* **Obsidian mcp server** for internal best-practice notes.
* `./vendor/cubecl_wgpu` example sources.

---

## 9. Documentation and Information Sources (Mandatory Usage)

All implementers (including AI-based tools) must adhere to the following when working on the `backend` module:

1. **API and crate documentation:**

   * Use **docs rs mcp server** to look up:

     * `cubecl-core`
     * `cubecl-matmul`
     * `cubecl-wgpu`
     * `cubecl-runtime`
     * `cubecl-reduce`
   * Do not guess function signatures or types if documentation is available.

2. **Sample code and internal design notes:**

   * Use **Obsidian mcp server** to locate:

     * Existing CubeCL kernel examples.
     * Notes on WGPU/CPU runtime setup and memory management.
     * Previous FFT or matmul integration code.

3. **Concrete reference implementation:**

   * Use the local **`./vendor`** directory to inspect:

     * Actual crate source code.
     * Official examples (e.g., `./vendor/cubecl-matmul/examples`).
     * Tests that demonstrate correct usage of runtimes and kernels.

4. **Build and run verification:**

   * After modifications to `backend`:

     * Run `cargo build` and `cargo test` for the entire workspace.
     * Ensure that no warnings or errors are introduced in the public API.
     * For WGPU-specific features, run tests on a machine with a supported GPU.

---

## 10. Review and Improvement Process (3 Iterations Required)

The `backend` module is critical infrastructure. To ensure robustness and maintainability:

### 10.1 Required Review Cycles for Implementation

For any substantial change or initial implementation of the backend:

1. **Review Cycle 1: API Shape and Semantics**

   * Check that trait names, method signatures, and error semantics match the requirements in this document.
   * Ensure the separation of concerns (linalg/fft/reduce/memory) is respected.
   * Adjust names and signatures for clarity and consistency.

2. **Review Cycle 2: Correctness and Algorithmic Soundness**

   * Validate that:

     * `matmul` behaves correctly for non-square and non-contiguous shapes (if supported).
     * `fft3d_forward` and `fft3d_inverse` implement Cooley–Tukey correctly and satisfy round-trip accuracy tests.
     * `sum_f64` matches CPU reference results.
   * Review the Cooley–Tukey implementation for correct indexing, bit-reversal, and normalization.

3. **Review Cycle 3: Performance and Resource Usage**

   * Evaluate:

     * Memory behavior (persistent vs transient buffers).
     * Overheads in kernel launches and data transfers.
     * Parallelism and utilization (CPU thread pools / WGPU queues).
   * Apply micro-optimizations only where they measurably improve performance without harming clarity.

Each review cycle should be documented (e.g., in PR comments or design notes) and must be completed before merging into the main branch.

### 10.2 Required Review Cycles for This Design Document

This **backend trait design document itself** must also undergo **at least three review-and-improvement iterations**:

1. **Design Review 1: Coverage and Consistency**

   * Check that all backend responsibilities are covered.
   * Ensure there are no contradictions with the higher-level requirements specification.

2. **Design Review 2: Implementability**

   * Confirm that all traits can be implemented using the current CubeCL APIs and runtime abstractions.
   * Adjust or simplify as necessary based on feedback from implementers.

3. **Design Review 3: Long-term Extensibility**

   * Validate that the abstraction can support future backends (e.g., CUDA).
   * Ensure that introducing new operations (e.g., batched eigensolvers, additional reductions) will not require breaking changes to existing traits.

Results of each design review should be captured in versioned design notes (e.g., Obsidian pages or design docs in the repo).

---

## 11. Backend Implementer Cheat Sheet (local + Obsidian references)

Use this as the fastest, copy-paste-friendly path to wire the traits above using the verified samples.

**Matmul (CubeCL WGPU)**
- Follow `Cubecl/cubecl_matmul_gemm_example.md` (Obsidian). Steps: `WgpuDevice::DefaultDevice` ➜ `init_setup::<AutoGraphicsApi>` ➜ `init_device` ➜ `ComputeClient::load`. Upload host arrays with `create_tensor`, allocate output via `empty_tensor`, wrap each allocation in `TensorHandle::<WgpuRuntime, f32|f64>::new(shape, strides)`, then `MatmulInputHandle::Normal(...)`. Launch with `launch::<WgpuRuntime, T>(&Strategy::Auto, &client, lhs, rhs, out.clone())`; read back via `client.read_tensor(vec![out.as_copy_descriptor()])`.

**Reduce (sum f64, CPU runtime)**
- Pattern from `Cubecl/cubecl_reduce_sum.md`: build `TensorHandleRef::<CpuRuntime>::from_raw_parts` for input/output, call `reduce::<CpuRuntime, Sum>(&client, input, output, axis=0, None, (), ReduceDtypes { input/output/accum = Float64 })`, then `read_tensor` using `copy_descriptor`. Mirror for `sum_f64` implementation; return host scalar.

**3D FFT (Cooley–Tukey, split-complex)**
- Reference `Cubecl/cubecl_3d_dft.md`: three passes `fft_x`, `fft_y`, `fft_z` written as `#[cube(launch_unchecked)]` kernels; host launches with `CubeDim::new_1d(N)` / `CubeCount::new_1d(1)` and split-complex buffers. Use as template for `fft3d_forward`/`fft3d_inverse`; ensure inverse scales by `1/(nx*ny*nz)`.

**WGPU runtime init knobs**
- From `vendor/cubecl-wgpu/src/runtime.rs`: `RuntimeOptions { tasks_max, memory_config }`, default `tasks_max = 32` (1 in tests). `ComputeClient::load(&device)` after `init_device(setup, options)`. Prefer `MemoryConfiguration::SubSlices` unless backend requires `ExclusivePages`.

**Matmul handle requirements**
- From `vendor/cubecl-matmul/src/base.rs`: `MatmulInputHandle` wraps `TensorHandle`; strides must be contiguous per `Runtime::can_read_tensor`. `Strategy::Auto` tries Simple then SimpleUnit.

**Memory pools and limits**
- From `vendor/cubecl-runtime/src/memory_management/mod.rs`: use `MemoryDeviceProperties` for `max_page_size`/`alignment`; choose pool type via `MemoryConfiguration::{SubSlices, ExclusivePages, Custom { pool_options }}` to implement `max_memory_mb` and `alloc` heuristics.

**Testing quickchecks**
- GEMM: compare against CPU reference for small shapes (2x3·3x2, non-square) using `client.read_tensor`.
- Reduce: sum `[1.,2.,3.,4.]` equals 10 within 1e-12.
- FFT: round-trip error `max(|f - ifft(fft(f))|) < 1e-9` on small 4×4×4 mesh.

**Mandatory sources to consult before coding**
- Obsidian MCP: `Cubecl/cubecl_matmul_gemm_example.md`, `Cubecl/cubecl_reduce_sum.md`, `Cubecl/cubecl_3d_dft.md`.
- Local `vendor`: `cubecl-wgpu/src/runtime.rs`, `cubecl-matmul/src/base.rs`, `cubecl-runtime/src/memory_management/mod.rs`.
- docs.rs via docs-mcp: `cubecl-core`, `cubecl-wgpu`, `cubecl-matmul`, `cubecl-reduce`, `cubecl-runtime`.

**Review cycle hook**
- Attach checklist above to Review Cycles 2 and 3: confirm numerical correctness (GEMM/reduce/FFT) and resource behavior (memory pools, tasks_max) before merging.
