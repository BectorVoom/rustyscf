/// Uniform k-point mesh for SCF (Monkhorst–Pack style).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct KMesh {
    pub dims: [u32; 3],
}

impl KMesh {
    pub fn new(dims: [u32; 3]) -> Self {
        Self { dims }
    }

    /// Total number of sampling k-points.
    pub fn len(&self) -> usize {
        self.dims[0] as usize * self.dims[1] as usize * self.dims[2] as usize
    }

    /// Generate a simple Gamma-centered Monkhorst–Pack grid (no offset).
    pub fn generate_points(&self) -> Vec<KPoint> {
        let [nx, ny, nz] = self.dims;
        let mut out = Vec::with_capacity(self.len());

        for ix in 0..nx {
            for iy in 0..ny {
                for iz in 0..nz {
                    let coords = [
                        ix as f64 / nx as f64,
                        iy as f64 / ny as f64,
                        iz as f64 / nz as f64,
                    ];
                    out.push(KPoint::new(coords));
                }
            }
        }

        out
    }
}

/// Single k-point (fractional coordinates in reciprocal lattice basis).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct KPoint {
    pub frac: [f64; 3],
    pub label: Option<&'static str>,
}

impl KPoint {
    pub fn new(frac: [f64; 3]) -> Self {
        Self {
            frac,
            label: None,
        }
    }

    pub fn gamma() -> Self {
        Self::new([0.0, 0.0, 0.0]).with_label("G")
    }

    pub fn with_label(mut self, label: &'static str) -> Self {
        self.label = Some(label);
        self
    }
}

/// Path segment between two k-points, inclusive, with a fixed number of
/// interpolation points.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct KPathSegment {
    pub start: KPoint,
    pub end: KPoint,
    /// Number of points along the segment (inclusive). Must be >= 2.
    pub n_points: usize,
}

impl KPathSegment {
    pub fn new(start: KPoint, end: KPoint, n_points: usize) -> Self {
        let n_points = n_points.max(2);
        Self {
            start,
            end,
            n_points,
        }
    }
}

/// High-symmetry path for band-structure calculations.
#[derive(Clone, Debug, Default)]
pub struct KPath {
    segments: Vec<KPathSegment>,
}

impl KPath {
    pub fn new() -> Self {
        Self { segments: Vec::new() }
    }

    /// Add a segment with a fixed number of interpolation points (>=2).
    pub fn push_segment(mut self, segment: KPathSegment) -> Self {
        self.segments.push(segment);
        self
    }

    /// Convenience builder: create a path from an ordered list of high-symmetry
    /// points using a uniform number of points per segment.
    pub fn from_points(points: Vec<KPoint>, n_points_per_segment: usize) -> Self {
        let mut path = Self::new();
        if points.len() < 2 {
            return path;
        }

        for pair in points.windows(2) {
            let start = pair[0];
            let end = pair[1];
            path = path.push_segment(KPathSegment::new(
                start,
                end,
                n_points_per_segment,
            ));
        }

        path
    }

    /// Total number of k-points across all segments (with shared endpoints
    /// counted once).
    pub fn total_points(&self) -> usize {
        if self.segments.is_empty() {
            return 0;
        }

        let mut total = 1; // first point of first segment
        for seg in &self.segments {
            total += seg.n_points.saturating_sub(1);
        }
        total
    }

    /// Interpolate the full set of k-points along the path.
    pub fn interpolate(&self) -> Vec<KPoint> {
        let mut out = Vec::with_capacity(self.total_points());

        for (seg_idx, seg) in self.segments.iter().enumerate() {
            let n = seg.n_points.max(2);

            for i in 0..n {
                // avoid duplicating the first point of subsequent segments
                if seg_idx > 0 && i == 0 {
                    continue;
                }

                let t = i as f64 / (n - 1) as f64;
                let frac = [
                    (1.0 - t) * seg.start.frac[0] + t * seg.end.frac[0],
                    (1.0 - t) * seg.start.frac[1] + t * seg.end.frac[1],
                    (1.0 - t) * seg.start.frac[2] + t * seg.end.frac[2],
                ];

                let label = if i == 0 {
                    seg.start.label
                } else if i == n - 1 {
                    seg.end.label
                } else {
                    None
                };

                out.push(KPoint { frac, label });
            }
        }

        out
    }

    pub fn segments(&self) -> &[KPathSegment] {
        &self.segments
    }

    pub fn is_empty(&self) -> bool {
        self.segments.is_empty()
    }
}
