# rustyscf

Library-first crate for band-structure workflows built on top of `cubecl`. Task 1 establishes the public API skeleton only; numerical kernels arrive in later milestones.

## Status
- Crate type: library (`lib`)
- Edition: 2024, MSRV 1.85
- Platforms: macOS arm64 (primary); Linux/Windows x86_64/aarch64 (beta)

## Quick start
```rust
use rustyscf::{CellBuilder, Unit, KMesh, ScfBuilder, Method, BandStructureBuilder};

let cell = CellBuilder::new()
    .with_atom("H 0 0 0; H 1 1 1")
    .with_a([[2.0,0.0,0.0],[0.0,2.0,0.0],[0.0,0.0,2.0]])
    .with_unit(Unit::Angstrom)
    .with_basis("gth-dzvp")
    .with_pseudo("gth-pbe")
    .build()?;

let scf = ScfBuilder::new(&cell)
    .with_method(Method::KRHF)
    .with_kmesh(KMesh::new([4,4,4]))
    .run()?;

// BandStructureBuilder currently returns an internal error stub until the
// band kernels land in a later milestone.
let _bands = BandStructureBuilder::new(&cell, &scf).run();
# Ok::<_, rustyscf::Error>(())
```

## Development notes
- Dependencies: `cubecl = "0.8.1"`, `log`, `thiserror`.
- Feature flags: `cpu-backend` (default), `wgpu-backend`, `logging` (enables `env_logger`).
- Documentation is written for docs.rs; use the `docs-rs` MCP server to inspect upstream crate APIs during development.
- Backend architecture and implementation checklist: see `docs/backend_design.md`.

## License
MIT OR Apache-2.0
