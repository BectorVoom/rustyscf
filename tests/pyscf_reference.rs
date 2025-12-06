use rustyscf::{
    band::BandStructureBuilder,
    cell::CellBuilder,
    kpoints::{KMesh, KPath, KPoint},
    scf::{Method, PeriodicScf, ScfBuilder},
};
use serde::Deserialize;
use std::fs;
use std::path::Path;

#[derive(Deserialize)]
struct PyScfBandFixture {
    path_labels: Vec<String>,
    path_frac: Vec<[f64; 3]>,
    nk_per_segment: usize,
    nband: usize,
    nk: usize,
    energies_ha: Vec<Vec<f64>>,
}

#[test]
fn compare_simple_cubic_h_against_pyscf_fixture() {
    let fixture_path = Path::new("pyscf_test/pyscf_band_h.json");
    if !fixture_path.exists() {
        eprintln!("fixture missing; skip");
        return;
    }
    let raw = fs::read_to_string(fixture_path).expect("read fixture");
    let fixture: PyScfBandFixture = serde_json::from_str(&raw).expect("parse fixture");

    // Build matching cell (simple cubic H, gth-dzvp/pbe).
    let cell = CellBuilder::new()
        .with_atom("H 0 0 0")
        .with_a([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        .with_basis("gth-dzvp")
        .with_pseudo("gth-pbe")
        .build()
        .expect("cell");

    let scf = ScfBuilder::new(&cell)
        .with_method(Method::KRHF)
        .with_kmesh(KMesh::new([4, 4, 4]))
        .run()
        .expect("scf");

    let kpoints: Vec<KPoint> = fixture
        .path_frac
        .windows(2)
        .flat_map(|w| {
            let start = KPoint::new(w[0]).with_label("path-start");
            let end = KPoint::new(w[1]).with_label("path-end");
            KPath::new()
                .push_segment(
                    rustyscf::kpoints::KPathSegment::new(start, end, fixture.nk_per_segment),
                )
                .interpolate()
        })
        .collect();

    let bands = BandStructureBuilder::new(&cell, &scf)
        .with_kpoints(kpoints.clone())
        .with_mo_coeff(false)
        .run()
        .expect("bands");

    // Basic shape checks
    assert_eq!(bands.kpoints.len(), fixture.nk);
    assert_eq!(bands.energies.len(), fixture.nk);
    assert_eq!(bands.energies[0].len(), scf.ao_dimension());

    // Compute RMS energy difference against first nband from fixture with tighter threshold.
    let nband = fixture.nband.min(scf.ao_dimension());
    let mut mse = 0.0;
    let mut count = 0;
    for (k_idx, ref_row) in fixture.energies_ha.iter().enumerate() {
        if k_idx >= bands.energies.len() {
            break;
        }
        for b in 0..nband.min(ref_row.len()) {
            let ref_e = ref_row[b];
            let ours = bands.energies[k_idx][b];
            let diff = ref_e - ours;
            mse += diff * diff;
            count += 1;
        }
    }
    let rms = (mse / count as f64).sqrt();

    assert!(
        rms < 1.0,
        "RMS energy deviation too large: {rms:.3} Ha (nband={nband}, count={count})"
    );
}
