import numpy as np
import matplotlib.pyplot as plt
import time
import sys
import itertools
import multiprocessing as mp


# system parameters
Lx = 24
Ly = 24
l2 = 8
l1 = 8
m = 13
A = Lx * Ly + l2

def create_sparse_matrix(A):
    return np.zeros((A, A), dtype=complex)


# main function to run the simulation for a specific set of parameters
def run_simulation(params):
    mi_val, alpha_val, gamma_val = params

    # Matrix definitions and initializations
    SX = create_sparse_matrix(A)
    CX = create_sparse_matrix(A)
    SY = create_sparse_matrix(A)
    CY = create_sparse_matrix(A)
    Cons = np.eye(A)

    # sin(Kx) in real space
    for i in range(Lx - 1):
        for j in range(l1):
            SX[i + j*Lx, i + 1 +j*Lx] = 0.5j
            SX[i + 1 + j*Lx, i + j*Lx] = -0.5j
    
    for i in range(Lx):
        for j in range(l2):
            SX[i + j*(Lx + 1) + l1*Lx, i + 1 + j*(Lx + 1) + l1*Lx] = 0.5j
            SX[i + 1 + j*(Lx + 1) + l1*Lx, i + j*(Lx + 1) + l1*Lx] = -0.5j
    
    for i in range(Lx -1):
        for j in range(Ly - (l1 + l2)):
            SX[i + j*Lx +(l1 + l2)*Lx + l2, i + 1 + j*Lx + (l1 + l2)*Lx + l2] = 0.5j
            SX[i + 1 + j*Lx + (l1 + l2)*Lx + l2, i + j*Lx + (l1 + l2)*Lx + l2] = -0.5j
    
    # adding PBC
    for j in range(l1):
        SX[(j + 1)*Lx - 1, j*Lx] = 0.5j
        SX[j*Lx, (j + 1)*Lx - 1] = -0.5j
    
    for j in range(l2):
        SX[(j + 1)*(Lx + 1) - 1 + l1*Lx, j*(Lx + 1) + l1*Lx] = 0.5j
        SX[j*(Lx + 1) + l1*Lx, (j + 1)*(Lx + 1) - 1 + l1*Lx] = -0.5j
    
    for j in range(Ly - (l1 + l2)):
        SX[(j + 1)*Lx - 1 + (l1 + l2)*Lx + l2, j*Lx + (l1 + l2)*Lx + l2] = 0.5j
        SX[j*Lx + (l1 + l2)*Lx + l2, (j + 1)*Lx - 1 + (l1 + l2)*Lx + l2] = -0.5j


    # cos(Kx) in real space
    for i in range(Lx - 1):
        for j in range(l1):
            CX[i + j*Lx, i + 1 + j*Lx] = 0.5
            CX[i + 1 + j*Lx, i + j*Lx] = 0.5
    
    for i in range(Lx):
        for j in range(l2):
            CX[i + j*(Lx + 1) + l1*Lx, i + 1 + j*(Lx + 1) + l1*Lx] = 0.5
            CX[i + 1 + j*(Lx + 1) + l1*Lx, i + j*(Lx + 1) + l1*Lx] = 0.5
    
    for i in range(Lx -1):
        for j in range(Ly - (l1 + l2)):
            CX[i + j*Lx +(l1 + l2)*Lx + l2, i + 1 + j*Lx + (l1 + l2)*Lx + l2] = 0.5
            CX[i + 1 + j*Lx + (l1 + l2)*Lx + l2, i + j*Lx + (l1 + l2)*Lx + l2] = 0.5
    
    # adding PBC
    for j in range(l1):
        CX[(j + 1)*Lx - 1, j*Lx] = 0.5
        CX[j*Lx, (j + 1)*Lx - 1] = 0.5
    
    for j in range(l2):
        CX[(j + 1)*(Lx + 1) - 1 + l1*Lx, j*(Lx + 1) + l1*Lx] = 0.5
        CX[j*(Lx + 1) + l1*Lx, (j + 1)*(Lx + 1) - 1 + l1*Lx] = 0.5
    
    for j in range(Ly - (l1 + l2)):
        CX[(j + 1)*Lx - 1 + (l1 + l2)*Lx + l2, j*Lx + (l1 + l2)*Lx + l2] = 0.5
        CX[j*Lx + (l1 + l2)*Lx + l2, (j + 1)*Lx - 1 + (l1 + l2)*Lx + l2] = 0.5


    # sin(Ky) in real space
    for i in range(Lx):
        for j in range(l1 - 1):
            SY[i + j*Lx, i + (j + 1)*Lx] = 0.5j
            SY[i + (j + 1)*Lx, i + j*Lx] = -0.5j
    
    for i in range(Lx + 1):
        for j in range(l2 - 1):
            SY[i + j*(Lx + 1) + l1*Lx, i + (j + 1)*(Lx + 1) + l1*Lx] = 0.5j
            SY[i + (j + 1)*(Lx + 1) + l1*Lx, i + j*(Lx + 1) + l1*Lx] = -0.5j
    
    for i in range(Lx):
        for j in range(Ly - (l1 + l2) -1):
            SY[i + j*Lx +(l1 + l2)*Lx + l2, i + (j + 1)*Lx + (l1 + l2)*Lx + l2] = 0.5j
            SY[i + (j + 1)*Lx + (l1 + l2)*Lx + l2, i + j*Lx + (l1 + l2)*Lx + l2] = -0.5j
    
    # adding PBC
    for i in range (Lx):
        SY[Lx*(Ly - 1) + l2 + i, i ]  = 0.5j
        SY[i, Lx*(Ly - 1) + l2 + i] = -0.5j
    
    # adding connectors
    for i in range(m):
        SY[ i + (l1 -1)*Lx, i + l1*Lx] = 0.5j
        SY[i + l1*Lx, i + (l1 -1)*Lx] = -0.5j
        SY[i + (l1 + l2 -1 )*Lx + l2 -1, i + (l1 + l2)*Lx + l2] = 0.5j
        SY[i + (l1 + l2)*Lx + l2, i + (l1 + l2 -1)*Lx + l2 -1] = -0.5j
    
    for i in range(Lx - m):
        SY[i + m + (l1 - 1)*Lx, i + m + 1 + l1*Lx] = 0.5j
        SY[i + m + 1 + l1*Lx, i + m + (l1 - 1)*Lx] = -0.5j
        SY[i + m + 1 + (l1 + l2 - 1)*Lx + l2 - 1, i + m + (l1 + l2)*Lx + l2] = 0.5j
        SY[i + m + (l1 + l2)*Lx + l2, i + m + 1 + (l1 + l2 - 1)*Lx + l2 - 1] = -0.5j


    # cos(Ky) in real space
    for i in range(Lx):
        for j in range(l1 - 1):
            CY[i + j*Lx, i + (j + 1)*Lx] = 0.5
            CY[i + (j + 1)*Lx, i + j*Lx] = 0.5
    
    for i in range(Lx + 1):
        for j in range(l2 - 1):
            CY[i + j*(Lx + 1) + l1*Lx, i + (j + 1)*(Lx + 1) + l1*Lx] = 0.5
            CY[i + (j + 1)*(Lx + 1) + l1*Lx, i + j*(Lx + 1) + l1*Lx] = 0.5
    
    for i in range(Lx):
        for j in range(Ly - (l1 + l2) -1):
            CY[i + j*Lx +(l1 + l2)*Lx + l2, i + (j + 1)*Lx + (l1 + l2)*Lx + l2] = 0.5
            CY[i + (j + 1)*Lx + (l1 + l2)*Lx + l2, i + j*Lx + (l1 + l2)*Lx + l2] = 0.5
    
    # adding PBC
    for i in range (Lx):
        CY[Lx*(Ly - 1) + l2 + i, i ]  = 0.5
        CY[i, Lx*(Ly - 1) + l2 + i] = 0.5   
    
    # adding connectors
    for i in range(m):
        CY[ i + (l1 -1)*Lx, i + l1*Lx] = 0.5
        CY[i + l1*Lx, i + (l1 -1)*Lx] = 0.5
        CY[i + (l1 + l2 -1 )*Lx + l2 -1, i + (l1 + l2)*Lx + l2] = 0.5
        CY[i + (l1 + l2)*Lx + l2, i + (l1 + l2 -1)*Lx + l2 -1] = 0.5    
    
    for i in range(Lx - m):
        CY[i + m + (l1 - 1)*Lx, i + m + 1 + l1*Lx] = 0.5
        CY[i + m + 1 + l1*Lx, i + m + (l1 - 1)*Lx] = 0.5
        CY[i + m + 1 + (l1 + l2 - 1)*Lx + l2 - 1, i + m + (l1 + l2)*Lx + l2] = 0.5
        CY[i + m + (l1 + l2)*Lx + l2, i + m + 1 + (l1 + l2 - 1)*Lx + l2 - 1] = 0.5
    
    
    # Cons (constant matrix)
    Cons = np.eye(A)

    # Pauli Matrices
    sigma_x = np.array([[0, 1],
                        [1, 0]], dtype=complex)
    
    sigma_y = np.array([[0, -1j],
                        [1j, 0]], dtype=complex)
    
    sigma_z = np.array([[1, 0],
                        [0, -1]], dtype=complex)
    sigma_0 = np.eye(2)
    

    # constructing the reference dislocation mode against which we will check for fidelity

    m0_dis = -1

    H_dis = np.kron(SX,sigma_x) + np.kron(SY, sigma_y) + np.kron((CX + CY - m0_dis*Cons), sigma_z)
    eigvals_dis, eigvecs_dis = np.linalg.eig(H_dis)
    eig_dis = np.sort(np.real(eigvals_dis))
    
    sorted_indices_dis = np.argsort(np.real(eigvals_dis))
    sorted_eigvals_dis = eigvals_dis[sorted_indices_dis]
    sorted_eigvecs_dis = eigvecs_dis[:, sorted_indices_dis]
    
    # two eigenvalues closest to zero with opposite signs
    positive_index_dis = np.where(sorted_eigvals_dis > 0)[0][0]  
    negative_index_dis = positive_index_dis - 1 
    
    eigval_below_zero_dis = sorted_eigvals_dis[negative_index_dis]
    eigvec_below_zero_dis = sorted_eigvecs_dis[:, negative_index_dis]
    
    eigval_above_zero_dis = sorted_eigvals_dis[positive_index_dis]
    eigvec_above_zero_dis = sorted_eigvecs_dis[:, positive_index_dis]
    
    # topological reference mode
    v0 = (eigvec_above_zero_dis + eigvec_below_zero_dis)/np.sqrt(2)
    
    

    # constructing the initial mixed density matrix
    Hi = np.kron(SX, sigma_x) + np.kron(SY, sigma_y) + np.kron((CX + CY - mi_val*Cons), sigma_z)
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
    mf_val = -1

    # Time-independent and time-dependent parts of the system hamiltonian   
    H0 = np.kron(SX, sigma_x) + np.kron(SY, sigma_y) + np.kron((CX + CY), sigma_z)
    Ht = -np.kron(Cons, sigma_z)


    # Condensation Probability stored in P
    P = np.zeros(N)

    # Helper function for time dependent mass ramp
    def m_ramp(t, mi, mf, alpha):
        return mi + (mf - mi) * (1 - np.exp(-alpha * t))


    # First loop: To handle the case when gamma = 0
    if gamma_val == 0:
        for n in range(N):
            m_val = m_ramp(n * dt, mi_val, mf_val, alpha_val)
            H = H0 + m_val * Ht
            C1 = H @ rho - rho @ H
            rho = rho - dt * 1j * C1
            P[n] = np.real(np.vdot(v0, np.matmul(rho, v0)))

    # Second loop: To handle the case when gamma != 0
    else:
        for n in range(N):
            m_val = m_ramp(n * dt, mi_val, mf_val, alpha_val)
            H = H0 + m_val * Ht
            C1 = H @ rho - rho @ H
            C2 = H @ C1 - C1 @ H  # Only calculate C2 if gamma != 0
            dephasing_term = -gamma_val * C2
            rho = rho - dt * 1j * C1 + dt * dephasing_term
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
    mi_values = [-2.25, 2.10, 0.10]
    alpha_values = [0.3, 20.0]
    gamma_values = [0.0, 0.5, 1.0, 4.0, 5.0, 10.0]
    
    parameter_combinations = list(itertools.product(mi_values, alpha_values, gamma_values))

    
    params = parameter_combinations[array_index]

    run_simulation(params)
