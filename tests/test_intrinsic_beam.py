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

        # Randomized state vectors
        self.lin_vel_0 = jax.random.uniform(self.key, shape=(self.beam.n_lin_vel,))
        self.ang_vel_0 = jax.random.uniform(self.key, shape=(self.beam.n_ang_vel,))
        self.force_0 = jax.random.uniform(self.key, shape=(self.beam.n_force,))
        self.moment_0 = jax.random.uniform(self.key, shape=(self.beam.n_moment,))

        self.coenergy_vector_0 = jnp.concatenate([
            self.lin_vel_0,
            self.ang_vel_0,
            self.force_0,
            self.moment_0
        ])

        self.H_matrix, self.J_linear_matrix = self.beam.assemble_matrices()
        self.energy_0 = self.coenergy_vector_0.T @ self.H_matrix @ self.coenergy_vector_0


    def test_skew_symmetry_linear_term(self):
        
        value = (self.coenergy_vector_0.T @ self.J_linear_matrix @ self.coenergy_vector_0) / self.energy_0
        #print("Linear part skew-symmetric: ", value, "\n")
        self.assertTrue(jnp.isclose(value, 0.0, atol=1e-7), msg=f"Linear term not skew-symmetric, got {value}")

    def test_skew_symmetry_nonlinear_term(self):
        f = self.beam.assemble_nonlinear_dynamics(self.coenergy_vector_0)
        value = (self.coenergy_vector_0.T @ f) / self.energy_0
        #print("Nonlinear part skew-symmetric: ", value, "\n")
        self.assertTrue(jnp.isclose(value, 0.0, atol=1e-7), msg=f"Nonlinear term not skew-symmetric, got {value}")


if __name__ == "__main__":
    unittest.main()
