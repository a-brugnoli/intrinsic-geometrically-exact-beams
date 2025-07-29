import jax.numpy as jnp
from jax.experimental import sparse
from src.linear_algebra.sparse_la_jax import construct_skew_symmetric_block
# Example sparse matrix A (shape m x n)
data = jnp.array([1.0, 2.0, 3.0, 5])
row = jnp.array([0, 1, 1, 1])
col = jnp.array([0, 1, 2, 3])
indices = jnp.stack([row, col], axis=1)
D = sparse.BCOO((data, indices), shape=(2, 4))  # Replace with your actual A
D_dense = D.todense()
J = construct_skew_symmetric_block(D)
J_dense = J.todense()

n, m = D_dense.shape
J_dense_manual = jnp.block([[jnp.zeros((n, n)), D_dense], 
                            [-D_dense.T, jnp.zeros((m, m))]])

assert jnp.allclose(J_dense, J_dense_manual), "J is not correctly constructed"