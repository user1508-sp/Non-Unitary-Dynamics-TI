import numpy as np
import matplotlib.pyplot as plt
import time
import itertools
import sys


# system parameters
Lx = 20
Ly = 20
A = Lx * Ly



def create_sparse_matrix(A):
    return np.zeros((A, A), dtype=complex)



# Function to run the computation for a specific set of parameters
def run_simulation(params):
    mi_val, alpha_val, gamma_val = params

    # Matrix definitions and initializations
    SX2D = create_sparse_matrix(A)
    CX2D = create_sparse_matrix(A)
    SY2D = create_sparse_matrix(A)
    CY2D = create_sparse_matrix(A)
    Cons = np.eye(A)

    
    # SX2D
    for i in range(Ly):
        for j in range(Lx - 1):
            SX2D[j + i*Lx, j + 1 + i*Lx] = 0.5j
            SX2D[j + 1 + i*Lx, j + i*Lx] = -0.5j


    # SY2D
    for i in range(Ly - 1):
        for j in range(Lx):
            SY2D[i*Lx + j, (i + 1)*Lx + j] = 0.5j
            SY2D[(i + 1)*Lx + j, i*Lx + j] = -0.5j


    # CX2D
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


    # building the two band hamiltonian in real space with OBC

    # Pauli Matrices
    sigma_1 = np.array([[0, 1],
                        [1, 0]], dtype=complex)

    sigma_2 = np.array([[0, -1j],
                        [1j, 0]], dtype=complex)

    sigma_3 = np.array([[1, 0],
                        [0, -1]], dtype=complex)
    sigma_0 = np.eye(2)

    # first pauli matrix operates on the spin index, second operates on the orbital index
    Gamma_31 = np.kron(sigma_3, sigma_1)
    Gamma_32 = np.kron(sigma_3, sigma_2)
    Gamma_33 = np.kron(sigma_3, sigma_3)
    Gamma_10 = np.kron(sigma_1, sigma_0)

    # building the 2-band hamiltonian with spin dofs and constructing the initial corner mode against which we will check for fidelity
    m0_cor = 1
    H_cor = np.kron(SX2D, Gamma_31) + np.kron(SY2D, Gamma_32) + np.kron((m0_cor * Cons - CX2D - CY2D), Gamma_33) + np.kron(CX2D - CY2D, Gamma_10)
    eigvals_cor, eigvecs_cor = np.linalg.eigh(H_cor)

    sorted_indices_cor = np.argsort(np.real(eigvals_cor))
    sorted_eigvals_cor = eigvals_cor[sorted_indices_cor]
    sorted_eigvecs_cor = eigvecs_cor[:, sorted_indices_cor]

    positive_index_cor = np.where(sorted_eigvals_cor > 0)[0][0]  # First positive eigenvalue
    negative_index_cor = positive_index_cor - 1  # Last negative eigenvalue

    first_eigvec_below_zero_cor = sorted_eigvecs_cor[:, negative_index_cor]
    first_eigvec_above_zero_cor = sorted_eigvecs_cor[:, positive_index_cor]
    second_eigvec_below_zero_cor = sorted_eigvecs_cor[:, negative_index_cor - 1]
    second_eigvec_above_zero_cor = sorted_eigvecs_cor[:, positive_index_cor + 1]

    # Constructing the reference topological mode against which we will check for condesation probability
    v0 = (first_eigvec_above_zero_cor + first_eigvec_below_zero_cor + second_eigvec_above_zero_cor + second_eigvec_below_zero_cor) / 2

    # Constructing the initial mixed density matrix
    Hi = np.kron(SX2D, Gamma_31) + np.kron(SY2D, Gamma_32) + np.kron((mi_val * Cons - CX2D - CY2D), Gamma_33) + np.kron(CX2D - CY2D, Gamma_10)
    eigvals, eigvecs = np.linalg.eigh(Hi)

    sorted_indices = np.argsort(np.real(eigvals))
    sorted_eigvals = eigvals[sorted_indices]
    sorted_eigvecs = eigvecs[:, sorted_indices]

    positive_index = np.where(sorted_eigvals > 0)[0][0]
    negative_index = positive_index - 1

    first_eigvec_below_zero = sorted_eigvecs[:, negative_index]
    first_eigvec_above_zero = sorted_eigvecs[:, positive_index]
    second_eigvec_below_zero = sorted_eigvecs[:, negative_index - 1]
    second_eigvec_above_zero = sorted_eigvecs[:, positive_index + 1]

    rho = np.zeros((sorted_eigvecs.shape[0], sorted_eigvecs.shape[0]), dtype=complex)
    negative_indices = np.where(sorted_eigvals < 0)[0]
    N_rho = len(negative_indices)

    # Add contributions to rho
    for i in negative_indices:
        v_neg = sorted_eigvecs[:, i]
        rho += np.outer(v_neg, np.conjugate(v_neg))

    rho += np.outer(first_eigvec_above_zero, np.conjugate(first_eigvec_above_zero))
    rho += np.outer(second_eigvec_above_zero, np.conjugate(second_eigvec_above_zero))
    rho /= (N_rho + 2)



    # Time evolution parameters
    T = 50
    dt = 0.001
    N = int(T/dt)

    # Time-independent and time-dependent parts of the system hamiltonian   
    H0 = np.kron(SX2D,Gamma_31) + np.kron(SY2D, Gamma_32) - np.kron((CX2D + CY2D), Gamma_33) + np.kron(CX2D - CY2D, Gamma_10)
    Ht = np.kron(Cons, Gamma_33)

    mf_val = m0_cor

    # Condensation Probability stored in P
    P = np.zeros(N+1)
    P[0] = np.real(np.vdot(v0, np.matmul(rho, v0)))


    # Helper function for time dependent mass ramp
    def m_ramp(t, mi, mf, alpha):
        return mi + (mf - mi) * (1 - np.exp(-alpha * t))

    # Helper function for calculating the local density of state
    def ldos(rho, i, j, lx, ly):
        index = i*lx + j
        ldos = np.real(rho[2 * index, 2 * index] + rho[2 * index + 1, 2 * index + 1])
        return ldos
        

   
    for n in range(N):
        m_val = m_ramp(n * dt, mi_val, mf_val, alpha_val)
        H = H0 + m_val * Ht
        C1 = H @ rho - rho @ H
        C2 = H @ C1 - C1 @ H
        dephasing_term = -gamma_val * C2
        rho = rho -dt*1j*C1 + dt * dephasing_term
        P[n+1] = np.real(np.vdot(v0, np.matmul(rho, v0)))

    

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

    # Parameter values
    mi_values = [2.25, -2.25]
    alpha_values = [0.3, 20.0]
    gamma_values = [2.0, 4.0, 10.0, 20.0]
    
    parameter_combinations = list(itertools.product(mi_values, alpha_values, gamma_values))

    
    params = parameter_combinations[array_index]

    run_simulation(params)
