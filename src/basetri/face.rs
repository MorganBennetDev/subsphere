//! Contains functionality for [Face]
use crate::basetri::{half_edge::HalfEdge, vertex::Vertex, OWNERSHIP};
use crate::BaseTriSphere;
use crate::math::vec;

/// A face of a [`BaseTriSphere`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Face(pub(crate) u8);

impl Face {
    /// Gets the [`BaseTriSphere`] this face belongs to.
    pub const fn sphere(self) -> BaseTriSphere {
        BaseTriSphere::INDEX_MAP[(self.0.saturating_sub(12) / 8) as usize]
    }
    
    /// Indicates whether this face [owns](OwnershipInfo) its second vertex.
    pub(crate) fn owns_vertex_1(self) -> bool {
        OWNERSHIP.owns_vert_1 & (1 << self.0) != 0
    }
    
    /// Indicates whether this face [owns](OwnershipInfo) its third edge.
    pub(crate) fn owns_edge_2(self) -> bool {
        OWNERSHIP.owns_edge_2 & (1 << self.0) != 0
    }
    
    /// Gets the number of vertices that are owned by the faces preceding this face in
    /// iteration order, not counting the first vertex of the shape.
    pub(crate) fn num_owned_vertices_before(self) -> usize {
        let start = self.sphere().first_face_inner();
        let before_mask = (1u32 << self.0) - 1;
        ((OWNERSHIP.owns_vert_1 & before_mask) >> start).count_ones() as usize
    }
    
    /// Gets the number of edges that are owned by the faces preceding this face in iteration
    /// order.
    pub(crate) fn num_owned_edges_before(self) -> usize {
        let start = self.sphere().first_face_inner();
        let index = self.0 - start;
        let before_mask = (1u32 << self.0) - 1;
        index as usize + ((OWNERSHIP.owns_edge_2 & before_mask) >> start).count_ones() as usize
    }
    
    /// Gets the [`HalfEdge`] which has the given [`index`](HalfEdge::side_index) and this face as
    /// its [`inside`](HalfEdge::inside).
    pub const fn side(self, index: usize) -> HalfEdge {
        assert!(index < 3, "index out of bounds");
        HalfEdge((self.0 << 2) | index as u8)
    }
    
    /// Gets the point at the center of this face.
    pub const fn center(self) -> [f64; 3] {
        let v_0 = self.side(0).start().pos();
        let v_1 = self.side(1).start().pos();
        let v_2 = self.side(2).start().pos();
        let mul = [0.4194695241216063, 0.5773502691896257][self.sphere() as usize]; // TODO
        vec::mul(vec::add(vec::add(v_0, v_1), v_2), mul)
    }
}

#[test]
fn test_center_face_at() {
    use crate::Sphere;
    // TODO: Extend to other spheres
    for sphere in [BaseTriSphere::Icosa, BaseTriSphere::Octa] {
        for face in sphere.faces() {
            let center = face.center();
            assert!((vec::dot(center, center) - 1.0).abs() < 1.0e-12);
            assert_eq!(sphere.face_at(center), face);
        }
    }
}

impl crate::Face for Face {
    type Vertex = Vertex;
    type HalfEdge = HalfEdge;
    
    fn index(&self) -> usize {
        (self.0 - self.sphere().first_face_inner()) as usize
    }
    
    fn area(&self) -> f64 {
        // TODO: Replace with lookup table
        let v_0 = self.side(0).start().pos();
        let v_1 = self.side(1).start().pos();
        let v_2 = self.side(2).start().pos();
        crate::util::tri_area([v_0, v_1, v_2])
    }
    
    fn num_sides(&self) -> usize {
        3
    }
    
    fn side(&self, index: usize) -> HalfEdge {
        (*self).side(index)
    }
}