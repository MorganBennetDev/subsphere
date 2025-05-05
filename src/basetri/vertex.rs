//! Contains functionality for [Vertex]
use crate::basetri::face::Face;
use crate::basetri::{half_edge::HalfEdge, NUM_VERTS, OWNERSHIP, VERTS};
use crate::BaseTriSphere;

/// Table used to implement [`crate::Vertex::outgoing`].
static OUTGOING: [[HalfEdge; 5]; NUM_VERTS] = const {
    let mut res: [[HalfEdge; 5]; NUM_VERTS] = [[HalfEdge(u8::MAX); 5]; NUM_VERTS];
    let mut j = 0;
    while j < NUM_VERTS {
        let vert = Vertex(j as u8);
        let owner = vert.owner();
        
        // Determine a unique first outgoing edge for each vertex
        let first_outgoing = {
            let mut k = 0;
            loop {
                let edge = owner.side(k);
                if edge.start().0 == vert.0 {
                    break edge;
                }
                k += 1;
                if k == 3 {
                    panic!("owner does not contain vertex");
                }
            }
        };
        
        // Build outgoing edge list
        let mut k = 0;
        let mut outgoing = first_outgoing;
        loop {
            res[j][k] = outgoing;
            k += 1;
            outgoing = outgoing.prev().twin();
            assert!(
                outgoing.start().0 == vert.0,
                "outgoing edge does not start at vertex"
            );
            if outgoing.0 == first_outgoing.0 {
                break;
            }
        }
        assert!(
            k == Vertex(j as u8).sphere().vertex_degree(),
            "degree mismatch"
        );
        j += 1;
    }
    res
};

/// A vertex of a [`BaseTriSphere`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Vertex(pub(crate) u8);

impl Vertex {
    /// Gets the [`BaseTriSphere`] this vertex belongs to.
    pub const fn sphere(self) -> BaseTriSphere {
        BaseTriSphere::INDEX_MAP[(self.0.saturating_sub(6) / 6) as usize]
    }
    
    /// Gets the face which [owns](OwnershipInfo) this vertex.
    pub(crate) const fn owner(self) -> Face {
        OWNERSHIP.vert_owner[self.0 as usize]
    }
    
    /// The position of this vertex.
    pub const fn pos(self) -> [f64; 3] {
        VERTS[self.0 as usize]
    }
}

impl crate::Vertex for Vertex {
    type Face = Face;
    type HalfEdge = HalfEdge;
    
    fn index(&self) -> usize {
        (self.0 - self.sphere().first_vertex_inner()) as usize
    }
    
    fn pos(&self) -> [f64; 3] {
        (*self).pos()
    }
    
    fn degree(&self) -> usize {
        self.sphere().vertex_degree()
    }
    
    fn outgoing(&self, index: usize) -> HalfEdge {
        assert!(index < self.degree(), "index out of bounds");
        OUTGOING[self.0 as usize][index]
    }
}