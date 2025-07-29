from jax.experimental.sparse import BCOO
import jax.numpy as jnp
from src.linear_algebra.sparse_la_jax import sparse_outer_product
# Create sparse vectors
a_dense = jnp.array([1.0, 0.0, 2.0]).reshape(-1, 1)
b_dense = jnp.array([0.0, 3.0, 0.0, 4.0]).reshape(-1, 1)

C_dense = jnp.outer(a_dense, b_dense)

a_sparse = BCOO.fromdense(a_dense)
b_sparse = BCOO.fromdense(b_dense)

C_sparse = sparse_outer_product(a_sparse, b_sparse)

print("Dense outer product:\n", C_dense)
print("Sparse outer product:\n", C_sparse.todense())

assert jnp.allclose(C_dense, C_sparse.todense()), "Outer product is not correct"
