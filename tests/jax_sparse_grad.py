import jax
import jax.numpy as jnp
from jax.experimental import sparse

# Create a sparse matrix in BCOO (Block Coordinate) format
# Let's create a 5x5 sparse matrix with a few non-zero elements
rows = jnp.array([0, 1, 2, 3, 4, 1, 2])
cols = jnp.array([0, 1, 2, 3, 4, 3, 1])
data = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0, 0.5, 1.5])
sparse_shape = (5, 5)

# Create indices matrix - shape should be (n_sparse, n_batch, n_dense)
# For standard COO format, we need shape (n_entries, n_dims)
indices = jnp.column_stack([rows, cols])  # Shape: (7, 2)

# Create sparse matrix using correct BCOO format
A_sparse = sparse.BCOO((data, indices), shape=sparse_shape)

print("Sparse matrix A:")
print("Non-zero elements:", A_sparse.data)
print("Indices (row, col):", A_sparse.indices.T)
print("Dense representation:")
print(A_sparse.todense())

# Create a dense vector
x = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
print(f"\nVector x: {x}")

# Define a function that uses sparse matrix operations
def sparse_quadratic_form(x, A_sparse):
    """Compute x^T @ A @ x using sparse operations"""
    Ax = A_sparse @ x  # Sparse matrix-vector multiplication
    return x @ Ax      # x^T @ (A @ x)

# Compute the function value
result = sparse_quadratic_form(x, A_sparse)
print(f"\nQuadratic form x^T @ A @ x = {result}")

# Compute gradient with respect to x using automatic differentiation
grad_fn = jax.grad(sparse_quadratic_form, argnums=0)  # Differentiate w.r.t. first argument (x)
gradient = grad_fn(x, A_sparse)
print(f"Gradient ∇_x(x^T @ A @ x) = {gradient}")

# Let's also compute the Jacobian for a more complex function
def sparse_transform(x, A_sparse):
    """Transform vector through sparse matrix and apply nonlinearity"""
    y = A_sparse @ x
    return jnp.tanh(y)  # Apply tanh element-wise

# Compute Jacobian
jacobian_fn = jax.jacfwd(sparse_transform, argnums=0)
jacobian = jacobian_fn(x, A_sparse)
print(f"\nJacobian of tanh(A @ x):")
print(jacobian)

# Example with trainable sparse matrix values
# Let's differentiate w.r.t. the sparse matrix data
def loss_function(sparse_data, x):
    """Loss function that depends on sparse matrix values"""
    A_sparse_param = sparse.BCOO((sparse_data, indices), shape=sparse_shape)
    y = A_sparse_param @ x
    return jnp.sum(y**2)  # L2 loss

# Compute gradient w.r.t. sparse matrix values
grad_sparse_fn = jax.grad(loss_function, argnums=0)
grad_sparse_data = grad_sparse_fn(data, x)
print(f"\nGradient w.r.t. sparse matrix values: {grad_sparse_data}")

# Demonstration of JIT compilation with sparse operations
@jax.jit
def compiled_sparse_op(x, sparse_data):
    """JIT-compiled sparse operation"""
    A = sparse.BCOO((sparse_data, indices), shape=sparse_shape)
    return A @ x

# This will be compiled on first call, then very fast
compiled_result = compiled_sparse_op(x, data)
print(f"\nJIT-compiled sparse matrix-vector product: {compiled_result}")

# Verify our gradient computation manually
# For quadratic form x^T @ A @ x, gradient should be (A + A^T) @ x
A_dense = A_sparse.todense()
manual_gradient = (A_dense + A_dense.T) @ x
print(f"\nManual gradient verification: {manual_gradient}")
print(f"JAX gradient:                {gradient}")
print(f"Difference: {jnp.abs(manual_gradient - gradient).max()}")