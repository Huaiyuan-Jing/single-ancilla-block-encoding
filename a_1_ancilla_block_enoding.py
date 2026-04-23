import numpy as np
import pennylane as qml
from scipy.linalg import svd, fractional_matrix_power

from algo import algorithm1 

def get_polar_decomposition(A):
    u, s, vh = svd(A)
    U = u @ vh
    H = vh.T.conj() @ np.diag(s) @ vh
    return U, H

def general_matrix_block_encoding(A, error_limit=1e-3):
    norm_A = np.linalg.norm(A, ord=2)
    A_scaled = A / norm_A if norm_A > 1.0 else A
    
    U_matrix, H_matrix = get_polar_decomposition(A_scaled)
    
    V_H, err_H, queries = algorithm1(H_matrix, error_limit)
    
    total_dim = V_H.shape[0]
    sys_dim = A.shape[0]
    ancilla_dim = total_dim // sys_dim
    
    U_full = np.kron(np.eye(ancilla_dim), U_matrix)
    
    V_A = U_full @ V_H
    
    A_extracted = V_A[:sys_dim, :sys_dim]
    if norm_A > 1.0:
        A_extracted *= norm_A
        
    final_err = np.linalg.norm(A - A_extracted, ord=2)
    
    return V_A, final_err

if __name__ == "__main__":
    A = np.array([[0.1, 0.4], 
                  [-0.2, 0.3]])
    
    try:
        V_final, err = general_matrix_block_encoding(A, error_limit=1e-2)
        print(f"\nSuccess, error: {err:.2e}")
        print(f"dim: {V_final.shape[0]}")
        
        print("\nA_approx:")
        print(np.real(V_final[:2, :2]))
        print("\n", np.real(V_final))
    except Exception as e:
        print(f"\nwrong: {e}")