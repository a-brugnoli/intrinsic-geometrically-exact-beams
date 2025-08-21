import unittest
import jax
import jax.numpy as jnp

from src.models.intrinsic_free_beam import IntrinsicPlanarFreeBeam

class TestIntrinsicPlanarFreeBeam(unittest.TestCase):
    def setUp(self):
        # Random key for reproducibility
        self.key = jax.random.PRNGKey(500)

        # Initialize beam
        n_elements = 10

        # Beam and material parameters
        length = 1.0              # Total length (m)
        E = 210e9            # Young's modulus (Pa)
        G = 80.77e9          # Shear modulus (Pa)
        rho = 7800           # Density (kg/m³)
        b = 0.1             # Width (m)
        h = 0.1             # Height (m)
        A = b * h            # Cross-sectional area (m²)
        I = b * h**3 / 12    # Second moment of area (m^4)
        kappa = 5/6          # Shear correction factor

        geometrical_props = {
            'L': length,   # Length of the beam
            'A': A,  # Cross-sectional area
            'I': I,  # Second moment of area of the beam
            'k': kappa,   # Shear coefficient for rectungular cross-section
        }

        material_props = {
            'E': E,  # Young's modulus
            'G': G,  # Shear modulus 
            'rho': rho,  # Density
        }
        self.beam = IntrinsicPlanarFreeBeam(n_elements, geometrical_props, material_props)

        self.J_linear_matrix = self.beam.assemble_linear_interconnection_matrix()
        

    def test_skew_symmetry_linear_term(self):
        J = self.J_linear_matrix

        # --- (1) Matrix-level check (authoritative) ---
        sym_part = J + J.T
        norm_J = jnp.linalg.norm(J, ord='fro')
        norm_sym = jnp.linalg.norm(sym_part, ord='fro')
        rel_residual = norm_sym / (norm_J + 1e-300)

        # Require skew-symmetry to within working precision
        self.assertLess(
            rel_residual, 1e-12,
            msg=f"J not skew-symmetric enough: relative residual {rel_residual}"
        )

        # --- (2) Diagnostic quadratic probe (non-fatal) ---
        max_probe = 0.0
        for i in range(5):
            key, self.key = jax.random.split(self.key)
            x = jax.random.normal(key, shape=(J.shape[0],))
            x = x / jnp.linalg.norm(x)
            val = float(x.T @ J @ x)
            max_probe = max(max_probe, abs(val))

        print(
            f"[diagnostic] Skew-symmetry residual = {rel_residual:.2e}, "
            f"max quadratic probe = {max_probe:.2e}"
        )


    def test_skew_symmetry_nonlinear_term(self):
        """
        Checks that the nonlinear term of the dynamics is skew-symmetric:
        For random u, (u^T @ f(u)) should vanish up to numerical precision.
        """
        max_abs_violation = 0.0
        max_rel_violation = 0.0

        for i in range(10):
            # Sample random test vector u
            key, self.key = jax.random.split(self.key)
            u = jax.random.normal(key, shape=(self.beam.n_dofs,))
            u = u / jnp.linalg.norm(u)

            # Evaluate f(u) = J(u) u
            f = self.beam.assemble_nonlinear_dynamics(u)

            # Quadratic form u^T f(u)
            val = float(u.T @ f)

            # Absolute violation
            max_abs_violation = max(max_abs_violation, abs(val))

            # Relative violation: scale by the magnitude of f(u)
            f_norm = float(jnp.linalg.norm(f))
            if f_norm > 1e-15:
                rel_val = abs(val) / f_norm
                max_rel_violation = max(max_rel_violation, rel_val)

        # --- Assertion (absolute tolerance) ---
        self.assertLess(
            max_abs_violation, 1e-10,
            msg=(
                f"Nonlinear J(u) not skew-symmetric enough: "
                f"max abs violation {max_abs_violation}, "
                f"max rel violation {max_rel_violation}"
            )
        )

        # --- Diagnostic logging ---
        print(
            f"[diagnostic] Nonlinear skew-symmetry check: "
            f"max abs violation = {max_abs_violation:.2e}, "
            f"max rel violation = {max_rel_violation:.2e}"
        )



if __name__ == "__main__":
    unittest.main()
