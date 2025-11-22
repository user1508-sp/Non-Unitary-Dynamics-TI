import numpy as np
import matplotlib.pyplot as plt
import time
import multiprocessing as mp
import itertools
import sys


# System parameters
Lx = 25
Ly = 25
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
    sigma_x = np.array([[0, 1],
                        [1, 0]], dtype=complex)

    sigma_y = np.array([[0, -1j],
                        [1j, 0]], dtype=complex)

    sigma_z = np.array([[1, 0],
                        [0, -1]], dtype=complex)
    sigma_0 = np.eye(2)


     
    # constructing the initial edge mode against which we will check for fidelity
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


    v0 = eigvec_above_zero_edg 
    rho = np.outer(v0, np.conjugate(v0)) 

    # Time evolution parameters
    T = 20
    dt = 0.001
    N = int(T/dt)
   
    # Time-independent and time-dependent parts of the system hamiltonian
    H0 = np.kron(SX2D, sigma_x) + np.kron(SY2D, sigma_y) + np.kron((CX2D + CY2D), sigma_z)
    Ht = -np.kron(Cons, sigma_z)

    mi_val = 1

    # Survival Probability stored in P
    P = np.zeros(N)

    # Helper function for time dependent mass ramp
    def m_ramp(t, mi, mf, alpha):
        return mi + (mf - mi) * (1 - np.exp(-alpha * t))


    # Helper funciton for calculating LDOS
    def ldos(rho, i, j, lx, ly):
        index = i*lx + j
        ldos = np.real(rho[2 * index, 2 * index] + rho[2 * index + 1, 2 * index + 1])
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
    filename_P = f"Edge_modes_P_mi={mi_val}_mf={mf_val}_alpha={alpha_val}_gamma={gamma_val}_euler.txt"
    np.savetxt(filename_P, P)

    vmin = np.min(ldos_at_instants[1])
    vmax = np.max(ldos_at_instants[1])

    fig, axs = plt.subplots(1, len(ldos_at_instants), figsize=(20, 8), sharey=True)
    fig.subplots_adjust(right=0.85)

    cbar_ax = fig.add_axes([0.9, 0.15, 0.03, 0.7])

    for idx, (ldos_grid, ax) in enumerate(zip(ldos_at_instants, axs)):
        np.savetxt(f"Edge_mode_LDOS_at_t={np.round(time_instants[idx], 1)}mi={mi_val}_mf={mf_val}_alpha={alpha_val}_gamma={gamma_val}_non_unitary_euler.txt", ldos_grid)
        im = ax.imshow(ldos_grid.T, origin='lower', extent=[0, Lx, 0, Ly], cmap='cividis', vmin=vmin, vmax=vmax)
        ax.set_xlabel("Lx")
        if idx == 0:
            ax.set_ylabel("Ly")
        ax.set_title(f"t={np.round(time_instants[idx], 1)}")

    fig.colorbar(im, cax=cbar_ax)
    plt.savefig(f"Edge_mode_LDOS_dephasing_mi={mi_val}_mf={mf_val}_alpha={alpha_val}_gamma={gamma_val}.png")
    plt.show()


# Cluster compatible main 
if __name__ == '__main__':
    
    # Handle HPC array index or run locally
    if len(sys.argv) > 1:
        array_index = int(sys.argv[1])
    else:
        print("No array index passed. Running with array_index = 0.")
        array_index = 0
    
    # Parameter combinations
    mf_values = [-2.25, -1, 2.25]
    alpha_values = [20.0, 0.3]
    gamma_values = [0.0, 0.5, 1.0, 5.0, 10.0]
    
    parameter_combinations = list(itertools.product(mf_values, alpha_values, gamma_values))

    
    params = parameter_combinations[array_index]

    run_simulation(params)
