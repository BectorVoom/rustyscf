# Fock / FFT Kernels – Detailed Design

This document defines the data layout, numerical algorithms, and kernel interfaces for periodic HF Fock-matrix construction using FFTDF and Cooley–Tukey 3D FFTs in `rust-pyscf-pbc-bands`. It complements `docs/backend_design.md` and must be iterated through at least three review cycles together with its implementation.

## 0. Goal

Reproduce PySCF PBC + FFTDF HF behavior (H + GTH) with CubeCL (CPU/WGPU) backends, using split-complex 3D FFT kernels and integrating with `BackendFft`, `BackendLinalg`, `BackendReduce`, and `BackendMemory`. Enforce CubeCL type constraints and mandate use of:

- PySCF (as reference) when algorithmic details are unclear.
- docs rs mcp server for crate docs.
- Obsidian mcp server for sample code/notes.
- `./vendor` for canonical crate sources.

Both implementation and this document require **≥3 review cycles**.

## 1. Scope and Assumptions

- KRHF, 3D periodic, GTH pseudopotentials, FFTDF Coulomb, minimal exchange (H-only).
- Smearing 0, `exxdiv="ewald"`.
- Backends: CPU (may be partially non-accelerated for K), WGPU (accelerate FFT/J; K may fall back).

## 2. Mathematical Background (HF with FFTDF)

Fock per k: `F = h + J - 0.5 K`, with density `D(k) = Σ_n f_n C C†`.

### Coulomb J via FFTDF
1) Build real-space density ρ(r) from AO pairs and D(k).
2) FFT → ρ(G).
3) Multiply by Coulomb kernel `v(G)=4π/|G|^2`, `v(0)=0` (neutral cell / Ewald).
4) iFFT → V_H(r).
5) Project back: `J_μν(k) = ∑_g φ*_μ(g,k) V_H(g) φ_ν(g,k) w_g`.

### Exchange K (v0.1.0)
Simplified CPU path acceptable; GPU may be unimplemented with clear `ResourceError`. Use PySCF (`pyscf.pbc.df.fft_jk`, `pyscf.pbc.scf.khf.get_veff`) as reference.

## 3. Data Layout

- Grid dims `(nx, ny, nz)` power-of-two; flattened index `n = (z*ny + y)*nx + x`.
- Split-complex grid:
  ```rust
  struct ComplexGrid3D { nx, ny, nz: usize, re: DeviceBuffer<f64>, im: DeviceBuffer<f64> }
  ```
- AO grid values (per k):
  - Layout: `values[mu][n]` contiguous in `n`; flattened `[mu * ngrid + n]`.

## 4. Kernel Decomposition

### 4.1 High-Level Fock Build
1) Core h(k) (outside scope).
2) Density matrices D(k).
3) J pipeline:
   - density grid build
   - forward FFT
   - apply Coulomb kernel in G
   - inverse FFT
   - project J
4) K pipeline (CPU baseline).
5) Assemble F.

### 4.2 Kernels

**Density build (AO×AO→ρ)**
```rust
#[cube(launch_unchecked)]
fn build_density(
    ao_vals: &Array<f64>,   // [mu][grid] per k
    d_mats: &Array<f64>,    // [k][mu][nu]
    rho: &mut Array<f64>,   // [grid] (optionally per k)
    #[comptime] nx: usize, #[comptime] ny: usize, #[comptime] nz: usize,
    #[comptime] nao: usize, #[comptime] nk: usize,
) { /* O(nao^2) per grid point */ }
```
Grid-parallel; acceptable O(nao²) for H-only; later optimize/screen.

**3D FFT (forward/inverse)**
- Implement radix-2 Cooley–Tukey per axis; split-complex buffers.
- Inverse multiplies by `1/(nx*ny*nz)`.
- Implemented via `BackendFft::fft3d_forward/inverse`, calling CubeCL `#[cube]` kernels.

**Coulomb kernel multiply in G**
```rust
#[cube(launch_unchecked)]
fn apply_coulomb_kernel(
    rho_re: &Array<f64>, rho_im: &Array<f64>,
    v_g: &Array<f64>,
    out_re: &mut Array<f64>, out_im: &mut Array<f64>,
    #[comptime] ngrid: usize,
) { /* elementwise multiply */ }
```

**J projection (V_H grid → J matrices)**
```rust
#[cube(launch_unchecked)]
fn project_j(
    ao_vals: &Array<f64>,   // [mu][grid] per k
    v_h: &Array<f64>,       // [grid]
    j_out: &mut Array<f64>, // [k][mu][nu]
    weights: &Array<f64>,
    #[comptime] nx: usize, #[comptime] ny: usize, #[comptime] nz: usize,
    #[comptime] nao: usize, #[comptime] nk: usize,
) { /* grid sum over mu,nu */ }
```
Parallelize over (k, μ, ν) or over grid blocks depending on occupancy.

## 5. Host-Side Orchestration (per backend)

J-build pseudo:
```rust
fn build_j(...) -> Result<()> {
    // rho grid alloc
    launch_build_density_kernel(...);
    backend.fft().fft3d_forward(grid)?;
    apply_coulomb_kernel(...); // host helper wraps kernel
    backend.fft().fft3d_inverse(grid)?;
    launch_project_j_kernel(...);
    Ok(())
}
```
Backend differences: allocation, launch config (CubeDim/CubeCount), CPU fallbacks.

K-build v0.1.0: CPU-only acceptable; GPU may return `ResourceError` until implemented.

## 6. CubeCL Constraints

- Only `CubeType` primitives; split-complex buffers.
- Flattened buffers; shapes passed as comptime/runtime scalars.
- No complex structs across kernel boundary; encode params explicitly.

## 7. Testing & Validation

- FFT round-trip tests (1D/2D/3D small grids) max error < 1e-10 (f64).
- Density/J small-system checks vs direct O(nao⁴) CPU reference.
- Integration vs PySCF (H-only): match Fock, total energy, eigenvalues within max 1e-6 Ha, RMS 1e-7 Ha.
- When unclear, inspect PySCF (`pyscf.pbc.df.FFTDF`, `pyscf.pbc.scf.KRHF`).

## 8. Documentation / Source Lookup Rules

- **PySCF**: investigate when FFTDF/J/K details are unclear (kernel form, G=0 handling, exchange).
- **docs rs mcp server**: `cubecl-core`, `cubecl-runtime`, `cubecl-wgpu`, `cubecl-matmul`, `cubecl-reduce`.
- **Obsidian mcp server**: CubeCL examples and notes (matmul, reduce, FFT prototypes).
- **./vendor**: authoritative crate sources (e.g., `vendor/cubecl_core`, `vendor/cubecl_matmul`, `vendor/cubecl_reduce`).

## 9. Review & Improvement (≥3 cycles)

### Implementation cycles
1) Algorithmic correctness: Cooley–Tukey indexing, density/J math, documented approximations vs PySCF.
2) Validation: unit tests, PySCF comparisons; fix discrepancies.
3) Performance: memory/launch tuning, batching; keep clarity.

### Design document cycles
1) Completeness (all components specified, aligned with backend traits).
2) Implementability (signatures/layout workable with CubeCL + PySCF behavior).
3) Extensibility (future K, other pseudopotentials/SOC without breaking traits).

Record outcomes of each review (Obsidian/PR notes).
