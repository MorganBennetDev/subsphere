//! Contains functionality for [HalfEdge]
use crate::basetri::face::Face;
use crate::basetri::{INDICES, NUM_FACES, NUM_VERTS};
use crate::basetri::vertex::Vertex;
use crate::BaseTriSphere;

/// Table used to implement [`crate::HalfEdge::twin`].
const TWIN: [[HalfEdge; 3]; NUM_FACES] = const {
    // Build adjacent mapping for potential edges
    let mut adjacent: [[u8; NUM_VERTS]; NUM_VERTS] = [[u8::MAX; NUM_VERTS]; NUM_VERTS];
    let mut i = 0;
    while i < NUM_FACES {
        let v_0 = INDICES[i][0] as usize;
        let v_1 = INDICES[i][1] as usize;
        let v_2 = INDICES[i][2] as usize;
        assert!(adjacent[v_0][v_1] == u8::MAX, "duplicate edge detected");
        assert!(adjacent[v_1][v_2] == u8::MAX, "duplicate edge detected");
        assert!(adjacent[v_2][v_0] == u8::MAX, "duplicate edge detected");
        adjacent[v_0][v_1] = (i as u8) << 2;
        adjacent[v_1][v_2] = ((i as u8) << 2) | 1;
        adjacent[v_2][v_0] = ((i as u8) << 2) | 2;
        i += 1;
    }
    
    // Convert to adjacency table for faces
    let mut res: [[HalfEdge; 3]; NUM_FACES] = [[HalfEdge(0); 3]; NUM_FACES];
    i = 0;
    while i < NUM_FACES {
        let v_0 = INDICES[i][0] as usize;
        let v_1 = INDICES[i][1] as usize;
        let v_2 = INDICES[i][2] as usize;
        assert!(adjacent[v_1][v_0] != u8::MAX, "hanging edge detected");
        assert!(adjacent[v_2][v_1] != u8::MAX, "hanging edge detected");
        assert!(adjacent[v_0][v_2] != u8::MAX, "hanging edge detected");
        res[i][0] = HalfEdge(adjacent[v_1][v_0]);
        res[i][1] = HalfEdge(adjacent[v_2][v_1]);
        res[i][2] = HalfEdge(adjacent[v_0][v_2]);
        i += 1;
    }
    res
};

/// A half-edge of a [`BaseTriSphere`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct HalfEdge(pub(crate) u8);

impl HalfEdge {
    /// Gets the [`BaseTriSphere`] this face belongs to.
    pub const fn sphere(self) -> BaseTriSphere {
        self.inside().sphere()
    }
    
    /// The index of this half-edge within the [`sides`](crate::Face::sides) list of its
    /// [`inside`](HalfEdge::inside).
    pub const fn side_index(self) -> usize {
        (self.0 & 0b11) as usize
    }
    
    /// Gets the [`Face`] whose interior boundary contains this half-edge.
    pub const fn inside(self) -> Face {
        Face(self.0 >> 2)
    }
    
    /// Gets the [`Vertex`] at the "start" of this half-edge.
    pub const fn start(self) -> Vertex {
        Vertex(INDICES[self.inside().0 as usize][self.side_index()])
    }
    
    /// Gets the complementary half-edge on the opposite side of the edge.
    ///
    /// The returned half-edge will go in the opposite direction along the same edge.
    pub const fn twin(self) -> Self {
        TWIN[self.inside().0 as usize][self.side_index()]
    }
    
    /// Gets the half-edge which shares the [`inside`](HalfEdge::inside) face of this half-edge and
    /// precedes it in counter-clockwise order around the face.
    pub const fn prev(self) -> Self {
        HalfEdge(if self.0 & 0b11 == 0 {
            self.0 + 2
        } else {
            self.0 - 1
        })
    }
    
    /// Gets the half-edge which shares the [`inside`](HalfEdge::inside) face of this half-edge and
    /// follows it in counter-clockwise order around the face.
    pub const fn next(self) -> Self {
        HalfEdge(if self.0 & 0b11 == 2 {
            self.0 - 2
        } else {
            self.0 + 1
        })
    }
}

impl crate::HalfEdge for HalfEdge {
    type Face = Face;
    type Vertex = Vertex;
    
    fn side_index(&self) -> usize {
        (*self).side_index()
    }
    
    fn length(&self) -> f64 {
        self.sphere().edge_length()
    }
    
    fn angle(&self) -> f64 {
        // TODO: Replace with lookup table
        let v_a = self.prev().start().pos();
        let v_b = self.start().pos();
        let v_c = self.next().start().pos();
        crate::util::angle(v_a, v_b, v_c)
    }
    
    fn inside(&self) -> Face {
        (*self).inside()
    }
    
    fn start(&self) -> Vertex {
        (*self).start()
    }
    
    fn twin(&self) -> Self {
        (*self).twin()
    }
    
    fn prev(&self) -> Self {
        (*self).prev()
    }
    
    fn next(&self) -> Self {
        (*self).next()
    }
}