from scipy.sparse.linalg import spsolve
from scipy.sparse import csr_matrix, lil_matrix
import numpy as np


def solve_bcs(A, b, bcs : dict = {}):
    if bcs:
        try:
            assert all(key <= A.shape[0] for key in bcs.keys())
        except AssertionError:
            raise ValueError("Boundary conditions are out of range")

        raise NotImplementedError("Feature to be implemented")
            
    return spsolve(A, b)


def remove_row_column(mat, row_i, column_i):
    list_wo_row = [i for i in range(mat.shape[0]) if i != column_i]
    modified_mat = lil_matrix(mat[:, list_wo_row])
    modified_mat.rows = np.delete(modified_mat.rows, row_i)
    modified_mat.data = np.delete(modified_mat.data, row_i)
    modified_mat._shape = (modified_mat._shape[0] - 1, modified_mat._shape[1])

    return modified_mat.tocsr()

if __name__=="__main__":

    # Example sparse matrix
    sparse_matrix = csr_matrix(np.array([
        [1, 2, 13],
        [0, 3, 4],
        [5, 6, 15]
    ]))

    # Row and column to remove
    row_to_remove = 1
    col_to_remove = 1

    # Remove the specified row and column
    new_sparse_matrix = remove_row_column(sparse_matrix, row_to_remove, col_to_remove)

    assert (new_sparse_matrix.toarray()==np.array([[1, 13], [5, 15]])).all()
    
    print("Successful test")
