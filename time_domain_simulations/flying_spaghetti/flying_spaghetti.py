from src.models.intrinsic_free_beam import IntrinsicPlanarFreeBeam
from data import geometrical_props, material_props
import jax.numpy as jnp
n_elements = 50  # Number of finite elements
intrinsic_beam = IntrinsicPlanarFreeBeam(n_elements, geometrical_props, material_props)

coenergy_vector = jnp.zeros(intrinsic_beam.n_dofs)

H_matrix, J_linear_matrix = intrinsic_beam.assemble_matrices()
f = intrinsic_beam.assemble_nonlinear_dynamics(coenergy_vector)


