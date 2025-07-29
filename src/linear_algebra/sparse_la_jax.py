import jax.numpy as jnp
import scipy.sparse as sp
# from jax.experimental import sparse
from jax.experimental.sparse import BCOO

# Add two sparse matrices
def sparse_add(a, b):
        data = jnp.concatenate([a.data, b.data])
        indices = jnp.concatenate([a.indices, b.indices])
        return BCOO((data, indices), shape=a.shape)


def empty_sparse(shape):
    return BCOO((jnp.array([]), jnp.zeros((0, 2), dtype=int)), shape=shape)


# Helper function to create diagonal sparse matrix
def create_diag_sparse(diag, size):
        indices = jnp.stack([jnp.arange(size), jnp.arange(size)], axis=1)
        return BCOO((diag, indices), shape=(size, size))
    

# Block diagonal assembly
def block_diag_sparse(matrices):
    shapes = [m.shape for m in matrices]
    total_shape = (sum(s[0] for s in shapes), sum(s[1] for s in shapes))
    
    data = jnp.concatenate([m.data for m in matrices])
    
    indices = []
    row_offset = 0
    col_offset = 0
    for m in matrices:
        m_indices = m.indices
        m_indices = m_indices.at[:, 0].add(row_offset)
        m_indices = m_indices.at[:, 1].add(col_offset)
        indices.append(m_indices)
        row_offset += m.shape[0]
        col_offset += m.shape[1]
    
    indices = jnp.concatenate(indices)
    return BCOO((data, indices), shape=total_shape)
    

def convert_to_scipy(bcoo_jax_matrix: BCOO):
    rows, cols = bcoo_jax_matrix.indices.T  # unpack from shape (nnz, 2)
    return sp.coo_matrix((bcoo_jax_matrix.data, (rows, cols)), shape=bcoo_jax_matrix.shape)


def construct_skew_symmetric_block(D: BCOO) -> BCOO:
    """
    Construct the block matrix J = [[0, D], [-D.T, 0]] from a JAX BCOO matrix D.
    
    Args:
        D (sparse.BCOO): A JAX sparse matrix of shape (n, m).
    
    Returns:
        J (sparse.BCOO): A JAX sparse matrix of shape (m+n, m+n), with blocks:
                         [ 0  D]
                         [-Dᵗ 0]
    """
    if not isinstance(D, BCOO):
        raise TypeError("Input must be a jax.experimental.sparse.BCOO matrix.")

    n, m = D.shape

    # Off-diagonal block positions
    D_row, D_col = D.indices.T
    D_indices_top_right = jnp.stack([D_row, D_col + n], axis=1)

    D_T = D.transpose()
    DT_row, DT_col = D_T.indices.T
    D_indices_bottom_left = jnp.stack([DT_row + n, DT_col], axis=1)

    # Combine data and indices
    data_combined = jnp.concatenate([D.data, -D_T.data])
    indices_combined = jnp.concatenate([D_indices_top_right, D_indices_bottom_left], axis=0)

    # Construct final matrix
    J_shape = (n + m, n + m)
    

    return BCOO((data_combined, indices_combined), shape=J_shape)


def sparse_outer_product(a: BCOO, b: BCOO) -> BCOO:
    """
    Compute the sparse outer product of two 1D BCOO vectors.

    Parameters:
    - a: BCOO with shape (m,1)
    - b: BCOO with shape (n,1)

    Returns:
    - BCOO sparse matrix of shape (m, n), representing a ⊗ b
    """
    assert a.ndim == 2 and b.ndim == 2, "Inputs must be (n_rows,1) arrays."
    assert a.shape[1] == 1 and b.shape[1] == 1, "Inputs must be column vectors."

    # Extract data and indices
    a_idx = a.indices[:, 0]
    b_idx = b.indices[:, 0]
    a_data = a.data
    b_data = b.data

    # Compute outer product indices and values
    ii, jj = jnp.meshgrid(a_idx, b_idx, indexing='ij')
    coords = jnp.stack([ii.ravel(), jj.ravel()], axis=-1)

    aa, bb = jnp.meshgrid(a_data, b_data, indexing='ij')
    data = (aa * bb).ravel()

    shape = (a.shape[0], b.shape[0])
    return BCOO((data, coords), shape=shape)