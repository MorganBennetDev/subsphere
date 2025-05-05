//! Contains functionality for [BaseTriSphere].
use crate::basetri::{icosa_point_index, face::Face, half_edge::HalfEdge, vertex::Vertex, C_1};

/// A lookup table which identifies which [`Face`] corresponds to a particular result from
/// [`icosa_point_index`].
///
/// For indices that don't correspond exactly to a face, this will provide an arbitrary
/// nearby [`Face`].
const ICOSA_FACE_AT: [Face; 243] = const {
    let mut res = [Face(u8::MAX); 243];
    
    // Assign each face to its proper index
    let sphere = BaseTriSphere::Icosa;
    let mut i = sphere.first_face_inner();
    while i < sphere.last_face_inner() {
        let face = Face(i);
        let j = icosa_point_index(face.center());
        assert!(res[j as usize].0 == u8::MAX, "index already assigned");
        res[j as usize] = face;
        i += 1;
    }
    
    // Fill remaining indices by iteratively copying the contents of a "nearby" index which is
    // already filled.
    let mut next = res;
    let mut all_filled = false;
    while !all_filled {
        let mut index = 0;
        all_filled = true;
        while index < 243 {
            if res[index].0 == u8::MAX {
                all_filled = false;
                
                // Try to find a component we can change slightly to get a filled index.
                let mut mul = 1;
                while mul <= 83 {
                    let comp = (index / mul) % 3;
                    if comp == 0 {
                        if res[index + mul].0 != u8::MAX {
                            next[index] = res[index + mul];
                            break;
                        }
                        if res[index + 2 * mul].0 != u8::MAX {
                            next[index] = res[index + 2 * mul];
                            break;
                        }
                    } else if res[index - comp * mul].0 != u8::MAX {
                        next[index] = res[index - comp * mul];
                        break;
                    }
                    mul *= 3;
                }
            }
            index += 1;
        }
        res = next;
    }
    res
};

/// Given a point on the unit sphere, gets an index which can be used to identify which
/// octahedron [`Face`] contains it.
///
/// The indexing scheme is arbitrary, but points on different [`Face`]s must have different
/// indices.
const fn octa_point_index([x, y, z]: [f64; 3]) -> u8 {
    (((x >= 0.0) as u8) << 2) | (((y >= 0.0) as u8) << 1) | ((z >= 0.0) as u8)
}

/// A compact lookup table which identifies which [`Face`] corresponds to a particular result
/// from [`octa_point_index`].
const OCTA_FACE_AT: u64 = const {
    let sphere = BaseTriSphere::Octa;
    let mut i = sphere.first_face_inner();
    let mut res = 0;
    while i < sphere.last_face_inner() {
        let face = Face(i);
        let j = octa_point_index(face.center());
        res |= (i as u64) << (j * 8);
        i += 1;
    }
    res
};

/// A tessellation of the unit sphere constructed by projecting a triangular platonic solid
/// onto it.
#[repr(u8)]
#[derive(Clone, Copy, Default, Debug, PartialEq, Eq, Hash)]
pub enum BaseTriSphere {
    /// A tessellation of the unit sphere constructed by projecting an icosahedron onto it.
    #[default]
    Icosa,
    
    /// A tessellation of the unit sphere constructed by projecting an octahedron onto it.
    Octa,
    
    /// A tessellation of the unit sphere constructed by projecting a tetrahedron onto it.
    Tetra,
}

impl BaseTriSphere {
    pub(crate) const INDEX_MAP: [BaseTriSphere; 3] = [ Self::Icosa, Self::Octa, Self::Tetra ];
    
    /// The degree of the vertices in this base shape.
    pub const fn vertex_degree(self) -> usize {
        self.lookup::<5, 4, 3>() as usize
    }
    
    /// The length of any edge on this base shape, or equivalently, the angle between any
    /// two adjacent vertices.
    pub const fn edge_length(self) -> f64 {
        [1.1071487177940904, std::f64::consts::FRAC_PI_2][self as usize]
    }
    
    /// The cosine of [`edge_length`](BaseTriSphere::edge_length`), or equivalently, the dot
    /// product between any two adjacent vertices.
    pub const fn edge_cos_length(self) -> f64 {
        [C_1, 0.0][self as usize]
    }
    
    /// The internal representation of the first face of this base shape.
    pub(crate) const fn first_face_inner(self) -> u8 {
        self.lookup::<0, 20, 28>()
    }
    
    /// One more than the internal representation of the last face of this base shape.
    pub(crate) const fn last_face_inner(self) -> u8 {
        self.lookup::<20, 28, 32>()
    }
    
    /// The internal representation of the first vertex of this base shape.
    pub(crate) const fn first_vertex_inner(self) -> u8 {
        self.lookup::<0, 12, 18>()
    }
    
    /// One more than the internal representation of the last vertex of this base shape.
    pub(crate) const fn last_vertex_inner(self) -> u8 {
        self.lookup::<12, 18, 22>()
    }
    
    /// Returns the constant value corresponding to this base shape.
    const fn lookup<const ICOSA: u8, const OCTA: u8, const TETRA: u8>(self) -> u8 {
        match self {
            BaseTriSphere::Icosa => ICOSA,
            BaseTriSphere::Octa => OCTA,
            BaseTriSphere::Tetra => TETRA
        }
    }
    
    /// Determines which face contains the given point on the unit sphere.
    pub const fn face_at(self, point: [f64; 3]) -> Face {
        match self {
            BaseTriSphere::Icosa => {
                let index = icosa_point_index(point);
                ICOSA_FACE_AT[index as usize]
            }
            BaseTriSphere::Octa => {
                let index = octa_point_index(point);
                Face((OCTA_FACE_AT >> (index * 8)) as u8)
            }
            BaseTriSphere::Tetra => todo!(),
        }
    }
    
    /// The number of vertices on the sphere.
    pub const fn num_vertices(self) -> usize {
        self.lookup::<12, 6, 4>() as usize
    }
    
    /// Gets the [`Vertex`] with the specified index.
    pub const fn vertex(self, index: usize) -> Vertex {
        assert!(index < self.num_vertices(), "index out of bounds");
        Vertex(self.first_vertex_inner() + index as u8)
    }
}

impl crate::Sphere for BaseTriSphere {
    type Face = Face;
    type Vertex = Vertex;
    type HalfEdge = HalfEdge;
    
    fn num_faces(&self) -> usize {
        self.lookup::<20, 8, 4>() as usize
    }
    
    fn face(&self, index: usize) -> Face {
        assert!(index < self.num_faces(), "index out of bounds");
        Face(self.first_face_inner() + index as u8)
    }
    
    fn faces(&self) -> impl Iterator<Item = Face> {
        (self.first_face_inner()..self.last_face_inner()).map(Face)
    }
    
    fn face_at(&self, point: [f64; 3]) -> Face {
        (*self).face_at(point)
    }
    
    fn num_vertices(&self) -> usize {
        (*self).num_vertices()
    }
    
    fn vertex(&self, index: usize) -> Vertex {
        (*self).vertex(index)
    }
    
    fn vertices(&self) -> impl Iterator<Item = Vertex> {
        (self.first_vertex_inner()..self.last_vertex_inner()).map(Vertex)
    }
}