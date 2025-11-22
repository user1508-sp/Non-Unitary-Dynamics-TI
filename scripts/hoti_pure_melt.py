import numpy as np
import matplotlib.pyplot as plt
import time
import itertools
import sys

# System parameters
Lx = 20
Ly = 20
A = Lx * Ly


def create_sparse_matrix(A):
    return np.zeros((A, A), dtype=complex)

# main function to run the simulation for a specific set of parameters
def run_simulation(params):
    mf_val, alpha_val, gamma_val = params

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

    H_cor = np.kron(SX2D,Gamma_31) + np.kron(SY2D, Gamma_32) + np.kron((m0_cor*Cons - CX2D -CY2D), Gamma_33) + np.kron(CX2D - CY2D, Gamma_10)

    eigvals_cor, eigvecs_cor = np.linalg.eig(H_cor)
    eig_cor = np.sort(np.real(eigvals_cor))
    
    # Sorting eigenvalues and eigenvectors by real part of eigenvalues
    sorted_indices_cor = np.argsort(np.real(eigvals_cor))
    sorted_eigvals_cor = eigvals_cor[sorted_indices_cor]
    sorted_eigvecs_cor = eigvecs_cor[:, sorted_indices_cor]
    
    # two eigenvalues closest to zero with opposite signs
    positive_index_cor = np.where(sorted_eigvals_cor > 0)[0][0]  # First positive eigenvalue
    negative_index_cor = positive_index_cor - 1  # The last negative eigenvalue
    
    # Extracting the corresponding eigenvectors
    first_eigval_below_zero_cor = np.real(sorted_eigvals_cor[negative_index_cor])
    first_eigvec_below_zero_cor = sorted_eigvecs_cor[:, negative_index_cor]
    
    first_eigval_above_zero_cor = np.real(sorted_eigvals_cor[positive_index_cor])
    first_eigvec_above_zero_cor = sorted_eigvecs_cor[:, positive_index_cor]
    
    second_eigval_below_zero_cor = np.real(sorted_eigvals_cor[negative_index_cor-1])
    second_eigvec_below_zero_cor = sorted_eigvecs_cor[:, negative_index_cor-1]
    
    second_eigval_above_zero_cor = np.real(sorted_eigvals_cor[positive_index_cor+1])
    second_eigvec_above_zero_cor = sorted_eigvecs_cor[:, positive_index_cor+1]



    v0 = first_eigvec_above_zero_cor
    rho = np.outer(v0, np.conjugate(v0)) 

    # Time evolution parameters
    T = 20
    dt = 0.001
    N = int(T/dt)
    
    # Time-independent and time-dependent parts of the system hamiltonian
    H0 = np.kron(SX2D,Gamma_31) + np.kron(SY2D, Gamma_32) - np.kron((CX2D + CY2D), Gamma_33) + np.kron(CX2D - CY2D, Gamma_10)
    Ht = np.kron(Cons, Gamma_33)

    mi_val = m0_cor

    # Survival Probability stored in P
    P = np.zeros(N)

    # Helper function for time dependent mass ramp
    def m_ramp(t, mi, mf, alpha):
        return mi + (mf - mi) * (1 - np.exp(-alpha * t))

    # Helper funciton for calculating LDOS
    def ldos(rho, i, j, lx, ly):
        index = 4*(i*lx + j)
        ldos = np.real(rho[index, index] + rho[index + 1, index + 1] + rho[index + 2, index + 2] + rho[index + 3, index+3] )
        return ldos
        
    # arbitrarily chosen time instants to record LDOS snapshots
    time_instants = np.array([0, 0.511, 1.111, 1.511, 2.111, 2.511, 3.111, 3.511, 4.111, 4.511, 5.111, 5.511, 6.111, 6.5111, 7.111, 7.511, 8.111, 8.511, 10.111, 10.511, 12.111, 12.511, 13.511, 14.111, 16.111])  
    ldos_at_instants = []

    # Time evolution loop
    for n in range(N):
        m_val = m_ramp(n * dt, mi_val, mf_val, alpha_val)
        H = H0 + m_val * Ht
        C1 = H @ rho - rho @ H
        C2 = H @ C1 - C1 @ H
        dephasing_term = -gamma_val * C2
        rho = rho -dt*1j*C1 + dt * dephasing_term
        P[n] = np.real(np.vdot(v0, np.matmul(rho, v0)))

        current_time = n * dt
        if np.any(np.isclose(current_time, time_instants, atol=dt/2)):
            ldos_grid = np.zeros((Lx, Ly))
            for i in range(Lx):
                for j in range(Ly):
                    ldos_grid[i, j] = ldos(rho, i, j, Lx, Ly)
            ldos_at_instants.append(ldos_grid)

    
   # Save P values and LDOS grids
    filename_P = f"corner_mode_P_mi={mi_val}_mf={mf_val}_alpha={alpha_val}_gamma={gamma_val}_euler.txt"
    np.savetxt(filename_P, P)

    vmin = np.min(ldos_at_instants[1])
    vmax = np.max(ldos_at_instants[1])

    fig, axs = plt.subplots(1, len(ldos_at_instants), figsize=(20, 8), sharey=True)
    fig.subplots_adjust(right=0.85)

    cbar_ax = fig.add_axes([0.9, 0.15, 0.03, 0.7])

    for idx, (ldos_grid, ax) in enumerate(zip(ldos_at_instants, axs)):
        np.savetxt(f"corner_mode_LDOS_at_t={np.round(time_instants[idx], 1)}mi={mi_val}_mf={mf_val}_alpha={alpha_val}_gamma={gamma_val}_non_unitary_euler.txt", ldos_grid)
        im = ax.imshow(ldos_grid, origin='upper', extent=[0, Lx, 0, Ly], cmap='cividis', vmin=vmin, vmax=vmax)
        ax.set_xlabel("Lx")
        if idx == 0:
            ax.set_ylabel("Ly")
        ax.set_title(f"t={np.round(time_instants[idx], 1)}")

    fig.colorbar(im, cax=cbar_ax)
    plt.savefig(f"corner_modes_LDOS_dephasing_mi={mi_val}_mf={mf_val}_alpha={alpha_val}_gamma={gamma_val}.png")
    plt.show()


# Cluster compatible main 
if __name__ == '__main__':
    
    # Handle HPC array index or run locally
    if len(sys.argv) > 1:
        array_index = int(sys.argv[1])
    else:
        print("No array index passed. Running with array_index = 0.")
        array_index = 0
   
    mf_values = [2.25, -2.25]
    alpha_values = [0.3, 20.0]
    gamma_values = [0.0, 0.5, 1.0, 5.0]
    
    parameter_combinations = list(itertools.product(mf_values, alpha_values, gamma_values))

    
    params = parameter_combinations[array_index]

    run_simulation(params)
