import numpy as np
import matplotlib.pyplot as plt
import time
import itertools
import sys
import multiprocessing as mp


# system parameters
Lx = 25
Ly = 25
A = Lx*Ly



def create_sparse_matrix(A):
    return np.zeros((A, A), dtype=complex)



# main function to run the simulation for a specific set of parameters
def run_simulation(params):
    mi_val, alpha_val, gamma_val = params

    # SX2D
    SX2D = create_sparse_matrix(A)
    for i in range(Ly):
        for j in range(Lx - 1):
            SX2D[j + i*Lx, j + 1 + i*Lx] = 0.5j
            SX2D[j + 1 + i*Lx, j + i*Lx] = -0.5j
    
    
    # SY2D
    SY2D = create_sparse_matrix(A)
    for i in range(Ly - 1):
        for j in range(Lx):
            SY2D[i*Lx + j, (i + 1)*Lx + j] = 0.5j
            SY2D[(i + 1)*Lx + j, i*Lx + j] = -0.5j
    
    
    # CX2D
    CX2D = create_sparse_matrix(A)
    for i in range(Ly):
        for j in range(Lx - 1):
            CX2D[j + i*Lx, j + 1 + i*Lx] = 0.5
            CX2D[j + 1 + i*Lx, j + i*Lx] = 0.5
    
    
    # CY2D
    CY2D = create_sparse_matrix(A)
    for i in range(Ly - 1):
        for j in range(Lx):
            CY2D[i*Lx + j, (i + 1)*Lx + j] = 0.5
            CY2D[(i + 1)*Lx + j, i*Lx + j] = 0.5
    
    
    
    
    # Cons (constant matrix)
    Cons = np.eye(A)
    
    
    # building the two band hamiltonian in real space with OBC
    
    # Pauli Matrices
    sigma_x = np.array([[0, 1],
                        [1, 0]], dtype=complex)
    
    sigma_y = np.array([[0, -1j],
                        [1j, 0]], dtype=complex)
    
    sigma_z = np.array([[1, 0],
                        [0, -1]], dtype=complex)
    sigma_0 = np.eye(2)
    
    
    # constructing the reference topological edge mode against which we will check for fidelity
    m0_edg = 1
    
    H_edg = np.kron(SX2D,sigma_x) + np.kron(SY2D, sigma_y) + np.kron((CX2D + CY2D - m0_edg*Cons), sigma_z)
    
    
    eigvals_edg, eigvecs_edg = np.linalg.eig(H_edg)
    eig_edg = np.sort(np.real(eigvals_edg))
    sorted_indices_edg = np.argsort(np.real(eigvals_edg))
    sorted_eigvals_edg = eigvals_edg[sorted_indices_edg]
    sorted_eigvecs_edg = eigvecs_edg[:, sorted_indices_edg]
    
    # two eigenvalues closest to zero with opposite signs
    positive_index_edg = np.where(sorted_eigvals_edg > 0)[0][0]  
    negative_index_edg = positive_index_edg - 1 
    
    eigval_below_zero_edg = sorted_eigvals_edg[negative_index_edg]
    eigvec_below_zero_edg = sorted_eigvecs_edg[:, negative_index_edg]
    
    eigval_above_zero_edg = sorted_eigvals_edg[positive_index_edg]
    eigvec_above_zero_edg = sorted_eigvecs_edg[:, positive_index_edg]
    
    
    v0 = (eigvec_above_zero_edg + eigvec_below_zero_edg)/np.sqrt(2)
    
    
    # constructing the initial mixed density matrix
    Hi = np.kron(SX2D, sigma_x) + np.kron(SY2D, sigma_y) + np.kron((CX2D + CY2D - mi_val*Cons), sigma_z)
    eigvals, eigvecs = np.linalg.eig(Hi)
    
    
    sorted_indices = np.argsort(np.real(eigvals))
    sorted_eigvals = eigvals[sorted_indices]
    sorted_eigvecs = eigvecs[:, sorted_indices]
    
    
    negative_indices = np.where(sorted_eigvals < 0)[0]  # Indices of negative eigenvalues
    N_rho = len(negative_indices)
    
    positive_index = np.where(sorted_eigvals > 0)[0][0]  
    eigvec_above_zero = sorted_eigvecs[:, positive_index]
    eigvec_below_zero = sorted_eigvecs[:, (positive_index - 1)]
    
    
    rho = np.zeros((sorted_eigvecs.shape[0], sorted_eigvecs.shape[0]), dtype=complex)
    
    
    for i in negative_indices:
        v_neg = sorted_eigvecs[:, i]  
        rho += np.outer(v_neg, np.conjugate(v_neg))  
    
    # contribution from the zero mode right above zero
    rho += np.outer(eigvec_above_zero, np.conjugate(eigvec_above_zero))
    rho /= (N_rho + 1)
    
    
    
    # Time evolution parameters
    T = 50
    dt = 0.001
    N = int(T/dt)
    mf_val = 1.0

    # Time-independent and time-dependent parts of the system hamiltonian
    H0 = np.kron(SX2D, sigma_x) + np.kron(SY2D, sigma_y) + np.kron((CX2D + CY2D), sigma_z)
    Ht = -np.kron(Cons, sigma_z)

    # Condensation Probability stored in P
    P = np.zeros(N)

    # Helper function for time dependent mass ramp
    def m_ramp(t, mi, mf, alpha):
        return mi + (mf - mi) * (1 - np.exp(-alpha * t))

    # Time evolution loop
    for n in range(N):
        m_val = m_ramp(n * dt, mi_val, mf_val, alpha_val)
        H = H0 + m_val * Ht
        C1 = H @ rho - rho @ H
        C2 = H @ C1 - C1 @ H
        dephasing_term = -gamma_val * C2
        rho = rho -dt*1j*C1 + dt * dephasing_term
        P[n] = np.real(np.vdot(v0, np.matmul(rho, v0)))

    
    
    filename = f"P_mi={mi_val}_mf={mf_val}_alpha={alpha_val}_gamma={gamma_val}euler.txt"
    filename_rho = f"rho_final_mi={mi_val}_mf={mf_val}_alpha={alpha_val}_gamma={gamma_val}euler.txt"
    np.savetxt(filename, P)
    np.savetxt(filename_rho,rho)




# Cluster compatible main 
if __name__ == '__main__':
    
    # Handle HPC array index or run locally
    if len(sys.argv) > 1:
        array_index = int(sys.argv[1])
    else:
        print("No array index passed. Running with array_index = 0.")
        array_index = 0

    # Parameter combinations
    mi_values = [-2.25, -1.0, 2.25]
    alpha_values = [0.3, 20.0]
    gamma_values = [0.0, 0.5, 1.0]

    
    parameter_combinations = list(itertools.product(mi_values, alpha_values, gamma_values))
    params = parameter_combinations[array_index]

    run_simulation(params)
