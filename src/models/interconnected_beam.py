import jax.numpy as jnp
import jax.scipy.linalg as jla
from src.fem.matrices_assembly import mass_matrix_lagrange, discrete_gradient, inner_product_cg_dg
from src.fem.trilinear_vector_form import assemble_2d_cross_form_cg, assemble_2d_cross_form_dg, \
                                        assemble_2d_cross_skew_adjoint_form_dg
      
from src.linear_algebra.la_jax import skew_block_matrix
import jax
from functools import partial


class IntrinsicPlanarInterconnectedBeam():
    pass