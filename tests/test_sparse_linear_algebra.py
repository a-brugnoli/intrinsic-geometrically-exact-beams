import unittest
import jax.numpy as jnp
from jax.experimental.sparse import BCOO
from src.linear_algebra.sparse_la_jax import sparse_outer_product, construct_skew_symmetric_block


class TestSparseLinearAlgebra(unittest.TestCase):

    def setUp(self):
        # Setup for sparse outer product
        self.a_dense = jnp.array([1.0, 0.0, 2.0]).reshape(-1, 1)
        self.b_dense = jnp.array([0.0, 3.0, 0.0, 4.0]).reshape(-1, 1)
        self.C_dense = jnp.outer(self.a_dense, self.b_dense)

        self.a_sparse = BCOO.fromdense(self.a_dense)
        self.b_sparse = BCOO.fromdense(self.b_dense)

        # Setup for skew-symmetric block
        data = jnp.array([1.0, 2.0, 3.0, 5.0])
        row = jnp.array([0, 1, 1, 1])
        col = jnp.array([0, 1, 2, 3])
        indices = jnp.stack([row, col], axis=1)
        self.D = BCOO((data, indices), shape=(2, 4))
        self.D_dense = self.D.todense()

    def test_outer_product_correctness(self):
        C_sparse = sparse_outer_product(self.a_sparse, self.b_sparse)
        C_sparse_dense = C_sparse.todense()
        self.assertTrue(
            jnp.allclose(self.C_dense, C_sparse_dense),
            msg="Sparse outer product does not match dense outer product."
        )

    def test_construct_skew_symmetric_block(self):
        J = construct_skew_symmetric_block(self.D)
        J_dense = J.todense()

        n, m = self.D_dense.shape
        J_dense_manual = jnp.block([
            [jnp.zeros((n, n)), self.D_dense],
            [-self.D_dense.T, jnp.zeros((m, m))]
        ])

        self.assertTrue(
            jnp.allclose(J_dense, J_dense_manual),
            msg="Skew-symmetric block matrix is not correctly constructed."
        )


if __name__ == '__main__':
    unittest.main()
