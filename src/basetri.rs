//! Contains types related to [`BaseTriSphere`].

pub mod base_tri_sphere;
pub mod face;
pub mod vertex;
pub mod half_edge;

use crate::math::vec;
use base_tri_sphere::BaseTriSphere;
use face::Face;

/// Provides information about the "ownership" relationships between faces, edges, and vertices.
///
/// The rules for ownership are as follows:
///  * Every vertex and edge is owned by exactly one face.
///  * Faces must own their first edge.
///  * Faces may own their third edge.
///  * The first face in a [`BaseTriSphere`] must own its first vertex, which must be the first
///    vertex in the [`BaseTriSphere`].
///  * Faces may own their second vertex.
struct OwnershipInfo {
    vert_owner: [Face; NUM_VERTS],
    owns_vert_1: u32,
    owns_edge_2: u32,
}

/// Provides information about the "ownership" relationships between faces, edges, and vertices.
const OWNERSHIP: OwnershipInfo = const {
    // Assign ownership of first vertex in each shape
    let mut vert_owner: [Face; NUM_VERTS] = [Face(u8::MAX); NUM_VERTS];
    vert_owner[0] = Face(0);
    vert_owner[12] = Face(20);

    // Assign ownership of each vertex to the first face that contains it as it's 1 vertex
    let mut owns_vert_1 = 0;
    let mut i = 0;
    while i < NUM_FACES {
        let v_1 = INDICES[i][1] as usize;
        if vert_owner[v_1].0 == u8::MAX {
            vert_owner[v_1] = Face(i as u8);
            owns_vert_1 |= 1 << i;
        }
        i += 1;
    }

    // Verify that every vertex has an owner
    let mut j = 0;
    while j < NUM_VERTS {
        assert!(vert_owner[j].0 != u8::MAX, "vertex does not have an owner");
        j += 1;
    }

    // Assign ownership of each edge. First by edge 0, then optionally by edge 2.
    let mut edge_has_owner: [[bool; NUM_VERTS]; NUM_VERTS] = [[false; NUM_VERTS]; NUM_VERTS];
    let mut num_owned_edges = 0;
    let mut i = 0;
    while i < NUM_FACES {
        let v_0 = INDICES[i][0] as usize;
        let v_1 = INDICES[i][1] as usize;
        let edge_has_owner = if v_0 < v_1 {
            &mut edge_has_owner[v_0][v_1]
        } else {
            &mut edge_has_owner[v_1][v_0]
        };
        assert!(!*edge_has_owner, "edge already has an owner");
        *edge_has_owner = true;
        num_owned_edges += 1;
        i += 1;
    }
    let mut owns_edge_2 = 0;
    let mut i = 0;
    while i < NUM_FACES {
        let v_2 = INDICES[i][2] as usize;
        let v_0 = INDICES[i][0] as usize;
        let edge_has_owner = if v_0 < v_2 {
            &mut edge_has_owner[v_0][v_2]
        } else {
            &mut edge_has_owner[v_2][v_0]
        };
        if !*edge_has_owner {
            *edge_has_owner = true;
            num_owned_edges += 1;
            owns_edge_2 |= 1 << i;
        }
        i += 1;
    }

    // Verify that every edge has an owner
    assert!(
        num_owned_edges == NUM_FACES * 3 / 2,
        "not all edges have an owner"
    );

    // Finalize ownership info
    OwnershipInfo {
        vert_owner,
        owns_vert_1,
        owns_edge_2,
    }
};

/// Given a point on the unit sphere, gets an index which can be used to identify which
/// icosahedron [`Face`] contains it.
///
/// The indexing scheme is arbitrary, but points on different [`Face`]s must have different
/// indices.
const fn icosa_point_index(point: [f64; 3]) -> u8 {
    // The index is constructed by testing the dot product of `point` with 5 distinct (and
    // non-antipodal) vertices. For each test, we get one of 3 results. This gives us 243
    // "possible" indices, with each face corresponding to exactly one of these.
    let mut res = 0;
    let mut i = 0;
    while i < 5 {
        let dot = vec::dot(point, BaseTriSphere::Icosa.vertex(i).pos());
        let comp = (dot.is_sign_positive() as u8 + 1) * ((dot.abs() > C_1) as u8);
        res = res * 3 + comp;
        i += 1;
    }
    res
}

/// The total number of vertices across all [`BaseTriSphere`]s.
const NUM_VERTS: usize = 12 + 6;

/// The total number of faces across all [`BaseTriSphere`]s.
const NUM_FACES: usize = 20 + 8;

/// The vertex position data for all potential vertices on a [`BaseTriSphere`].
const VERTS: [[f64; 3]; NUM_VERTS] = [
    // Icosahedron top apex
    [0.0, 0.0, 1.0],
    // Icosahedron top ring
    [C_0, 0.0, C_1],
    [C_2, C_3, C_1],
    [-C_4, C_5, C_1],
    [-C_4, -C_5, C_1],
    [C_2, -C_3, C_1],
    // Icosahedron bottom ring
    [C_4, -C_5, -C_1],
    [C_4, C_5, -C_1],
    [-C_2, C_3, -C_1],
    [-C_0, 0.0, -C_1],
    [-C_2, -C_3, -C_1],
    // Icosahedron bottom apex
    [0.0, 0.0, -1.0],
    // Octahedron
    [0.0, 0.0, 1.0],
    [1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
    [-1.0, 0.0, 0.0],
    [0.0, -1.0, 0.0],
    [0.0, 0.0, -1.0],
];

/// The face index data for all potential faces on a [`BaseTriSphere`].
const INDICES: [[u8; 3]; NUM_FACES] = [
    // Icosahedron top cap
    [0, 1, 2],
    [0, 2, 3],
    [0, 3, 4],
    [0, 4, 5],
    [0, 5, 1],
    // Icosahedron central ring
    [5, 6, 1],
    [6, 7, 1],
    [1, 7, 2],
    [7, 8, 2],
    [2, 8, 3],
    [8, 9, 3],
    [3, 9, 4],
    [9, 10, 4],
    [4, 10, 5],
    [10, 6, 5],
    // Icosahedron bottom cap
    [10, 11, 6],
    [6, 11, 7],
    [7, 11, 8],
    [8, 11, 9],
    [9, 11, 10],
    // Octahedron top cap
    [12, 13, 14],
    [12, 14, 15],
    [12, 15, 16],
    [12, 16, 13],
    // Octahedron bottom cap
    [16, 17, 13],
    [13, 17, 14],
    [14, 17, 15],
    [15, 17, 16],
];

/// `sqrt(4 / 5)`
const C_0: f64 = 0.8944271909999159;

/// `sqrt(1 / 5)`
const C_1: f64 = 0.4472135954999579;

/// `(5 - sqrt(5)) / 10`
const C_2: f64 = 0.276393202250021;

/// `sqrt((5 + sqrt(5)) / 10)`
const C_3: f64 = 0.8506508083520399;

/// `(5 + sqrt(5)) / 10`
const C_4: f64 = 0.7236067977499789;

/// `sqrt((5 - sqrt(5)) / 10)`
const C_5: f64 = 0.5257311121191336;
