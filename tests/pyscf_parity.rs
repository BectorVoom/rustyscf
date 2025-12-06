//! Regression harness to compare rustyscf band energies against reference PySCF data.
//! The test is ignored by default until numerical kernels are implemented.

use rustyscf::{
    band::BandStructureBuilder,
    band::BandLinalgBackend,
    cell::CellBuilder,
    kpoints::{KPath, KPoint},
    scf::{Method, ScfBuilder},
};
use serde::Deserialize;

#[derive(Deserialize)]
struct PyscfBandRef {
    kpts_frac: Vec<[f64; 3]>,
    energies_ha: Vec<Vec<f64>>, // [k][band]
    nband: usize,
    nk: usize,
}

#[ignore = "requires implemented PBC integrals and eigensolver"]
#[test]
fn parity_with_pyscf_hydrogen() {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("pyscf_test/pyscf_band_h.json");
    if !path.exists() {
        eprintln!("Skipping: reference data missing at {}", path.display());
        return;
    }

    let data: PyscfBandRef = serde_json::from_reader(std::fs::File::open(path).unwrap()).unwrap();

    let cell = CellBuilder::new()
        .with_atom("H 0 0 0; H 1 1 1")
        .with_a([[2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 2.0]])
        .with_basis("gth-dzvp")
        .with_pseudo("gth-pbe")
        .build()
        .unwrap();

    let scf = ScfBuilder::new(&cell)
        .with_method(Method::KRHF)
        .with_kmesh(rustyscf::kpoints::KMesh::new([2, 2, 2]))
        .run()
        .unwrap();

    // Build explicit k-point list from reference data
    let kpoints: Vec<KPoint> = data
        .kpts_frac
        .iter()
        .map(|frac| KPoint { frac: *frac, label: None })
        .collect();

    let bands = BandStructureBuilder::new(&cell, &scf)
        .with_kpoints(kpoints)
        .with_linalg_backend(BandLinalgBackend::Cpu)
        .run()
        .unwrap();

    let tol = 1e-4;
    for (k_idx, ref_row) in data.energies_ha.iter().enumerate() {
        for (band_idx, &ref_e) in ref_row.iter().enumerate() {
            let got = bands.energy(k_idx, band_idx).unwrap();
            let diff = (got - ref_e).abs();
            assert!(diff < tol, "k {k_idx} band {band_idx} diff {diff}");
        }
    }
}
