/// Uniform k-point mesh for SCF.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct KMesh {
    pub dims: [u32; 3],
}

impl KMesh {
    pub fn new(dims: [u32; 3]) -> Self {
        Self { dims }
    }
}

/// Single k-point (fractional coordinates).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct KPoint {
    pub coords: [f64; 3],
    pub label: Option<&'static str>,
}

impl KPoint {
    pub fn new(coords: [f64; 3]) -> Self {
        Self {
            coords,
            label: None,
        }
    }

    pub fn with_label(mut self, label: &'static str) -> Self {
        self.label = Some(label);
        self
    }
}

/// High-symmetry path for band-structure calculations.
#[derive(Clone, Debug, Default)]
pub struct KPath {
    points: Vec<KPoint>,
}

impl KPath {
    pub fn new() -> Self {
        Self { points: Vec::new() }
    }

    pub fn with_point(mut self, label: &'static str, coords: [f64; 3]) -> Self {
        self.points.push(KPoint {
            coords,
            label: Some(label),
        });
        self
    }

    /// Interpolate the stored high-symmetry points into a polyline with
    /// `n_step` segments between each consecutive pair. Placeholder logic for
    /// now; returns the provided points unmodified.
    pub fn interpolate(self, _n_step: usize) -> Vec<KPoint> {
        self.points
    }

    pub fn points(&self) -> &[KPoint] {
        &self.points
    }
}
