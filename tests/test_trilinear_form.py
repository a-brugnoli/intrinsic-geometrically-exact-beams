import unittest
import jax.numpy as jnp

from src.fem.triplet_scalar_form import assemble_trilinear_scalar_form_cg, assemble_trilinear_scalar_form_dg

class TestTrilinearScalarFormAssembly(unittest.TestCase):
    """
    The P1 interpolation is working correctly on P1 on P0 test functions.
    Test cases using DG functions are still under development.
    """
    def test_trilinear_scalar_form(self):
        N_els = 2
        length = 1.0
        x_nodes = jnp.linspace(0.0, length, N_els + 1)  # 3 nodes
        x_mids = 0.5 * (x_nodes[:-1] + x_nodes[1:])     # 2 mids

        alpha = 3.0
        u_dofs_p1 = alpha * x_nodes  # P1 interpolation
        u_dofs_p0 = alpha * x_mids   # P0 interpolation (not used here but defined)

        beta = 5.0
        w_dofs_p1 = beta * x_nodes
        w_dofs_p0 = beta * jnp.pi * x_mids  # not used here

        # P1 test
        Lcg = assemble_trilinear_scalar_form_cg(x_nodes, u_dofs_p1, w_dofs_p1)
        expected_Lcg = alpha * beta * length**3 * jnp.array([1/96, 7/48, 17/96])
        self.assertTrue(jnp.allclose(Lcg, expected_Lcg), msg=f"P1 form incorrect:\nGot {Lcg}\nExpected {expected_Lcg}")

        # P0 test
        Ldg = assemble_trilinear_scalar_form_dg(x_nodes, u_dofs_p1, w_dofs_p1)
        expected_Ldg = alpha * beta * length**3 * jnp.array([1/24, 7/24])
        self.assertTrue(jnp.allclose(Ldg, expected_Ldg), msg=f"P0 form incorrect:\nGot {Ldg}\nExpected {expected_Ldg}")

if __name__ == '__main__':
    unittest.main()
