import numpy as np
import matplotlib.pyplot as plt
import time
import itertools
import sys
import multiprocessing as mp

Lx = 24
Ly = 24
l2 = 8
l1 = 8
m = 13
A = Lx * Ly + l2

def create_sparse_matrix(A):
    return np.zeros((A, A), dtype=complex)

# Function to run the computation for a specific set of parameters
def run_simulation(params):
    mf_val, alpha_val, gamma_val = params

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
    

    # constructing the initial dislocation made against which we will check for fidelity

    m0_dis = -1

    H_dis = np.kron(SX,sigma_x) + np.kron(SY, sigma_y) + np.kron((CX + CY - m0_dis*Cons), sigma_z)
    
    # Calculate eigenvalues and eigenvectors
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
    

    v0 = (eigvec_above_zero_dis + eigvec_below_zero_dis)/np.sqrt(2)
    
    
    m0i = -1
    Hi = np.kron(SX, sigma_x) + np.kron(SY, sigma_y) + np.kron((CX + CY - m0i*Cons), sigma_z)
    
    
    eigvals, eigvecs = np.linalg.eig(Hi)
    
    
    sorted_indices = np.argsort(np.real(eigvals))
    sorted_eigvals = eigvals[sorted_indices]
    sorted_eigvecs = eigvecs[:, sorted_indices]


    negative_indices = np.where(sorted_eigvals < 0)[0]  # Indices of negative eigenvalues
    N_rho = len(negative_indices)
    
    positive_index = np.where(sorted_eigvals > 0)[0][0]  
    eigvec_above_zero = sorted_eigvecs[:, positive_index]
    eigvec_below_zero = sorted_eigvecs[:, (positive_index - 1)]
    
    # constructing the mixed initial density matrix
    rho = np.zeros((sorted_eigvecs.shape[0], sorted_eigvecs.shape[0]), dtype=complex)
    
    
    for i in negative_indices:
        v_neg = sorted_eigvecs[:, i]  
        rho += np.outer(v_neg, np.conjugate(v_neg))  
    
    # contribution from the zero mode right above zero
    rho += np.outer(eigvec_above_zero, np.conjugate(eigvec_above_zero))
    rho /= (N_rho + 1)
    
    

    T = 20
    dt = 0.001
    N = int(T/dt)

    H0 = np.kron(SX, sigma_x) + np.kron(SY, sigma_y) + np.kron((CX + CY), sigma_z)
    Ht = -np.kron(Cons, sigma_z)

    P = np.zeros(N)

    def m_ramp(t, mi, mf, alpha):
        return mi + (mf - mi) * (1 - np.exp(-alpha * t))

    def ldos_custom_density(rho_ldos, i, j, Lx, Ly, l1, l2, m):
        # Handle zero density for the gaps at m+1 
        if (j < l1 or j >= l1 + l2) and i == m:
            return 0  # Set LDOS to 0 at the dislocation gap in the lower and upper sections
    
        if j < l1:  # Lower section
            if i < m:
                index = i + j * Lx
            else:
                index = i - 1 + j * Lx
        elif j < l1 + l2:  # Middle section (dislocation)
            index = l1 * Lx + i + (j - l1) * (Lx + 1)
        else:  # Upper section
            if i < m:
                index = l1 * Lx + l2 * (Lx + 1) + i + (j - l1 - l2) * Lx
            else:
                index = l1 * Lx + l2 * (Lx + 1) + i - 1 + (j - l1 - l2) * Lx
        
        ldos = np.real(rho_ldos[2 * index, 2 * index] + rho_ldos[2 * index + 1, 2 * index + 1])
        
        return ldos
        

    time_instants = np.array([0, 0.511, 1.111, 1.511, 2.111, 2.511, 3.111, 3.511, 4.111, 13.511, 14.111, 16.111])  
    ldos_at_instants = []


    for n in range(N):
        m_val = m_ramp(n * dt, m0i, mf_val, alpha_val)
        H = H0 + m_val * Ht
        C1 = H @ rho - rho @ H
        C2 = H @ C1 - C1 @ H
        dephasing_term = -gamma_val * C2
        rho = rho -dt*1j*C1 + dt * dephasing_term
        P[n] = np.real(np.vdot(v0, np.matmul(rho, v0)))

        current_time = n * dt
        if np.any(np.isclose(current_time, time_instants, atol=dt/2)):
            ldos_grid = np.zeros((Lx + 1, Ly))
            for i in range(Lx + 1):
                for j in range(Ly):
                    ldos_grid[i, j] = ldos_custom_density(rho, i, j, Lx, Ly, l1, l2, m)
            ldos_at_instants.append(ldos_grid)

    
    # Save P values and LDOS grids
    filename_P = f"P__mi={m0i}_mf={mf_val}_alpha={alpha_val}_gamma={gamma_val}_euler.txt"
    np.savetxt(filename_P, P)

    vmin = np.min(ldos_at_instants[1])
    vmax = np.max(ldos_at_instants[1])

    fig, axs = plt.subplots(1, len(ldos_at_instants), figsize=(20, 8), sharey=True)
    fig.subplots_adjust(right=0.85)

    cbar_ax = fig.add_axes([0.9, 0.15, 0.03, 0.7])

    for idx, (ldos_grid, ax) in enumerate(zip(ldos_at_instants, axs)):
        np.savetxt(f"LDOS_at_t={np.round(time_instants[idx], 1)}mi={m0i}_mf={mf_val}_alpha={alpha_val}_gamma={gamma_val}_non_unitary_euler.txt", ldos_grid)
        im = ax.imshow(ldos_grid.T, origin='lower', extent=[0, Lx + 1, 0, Ly], cmap='cividis', vmin=vmin, vmax=vmax)
        ax.set_xlabel("Lx + 1")
        if idx == 0:
            ax.set_ylabel("Ly")
        ax.set_title(f"t={np.round(time_instants[idx], 1)}")

    fig.colorbar(im, cax=cbar_ax)
    plt.savefig(f"LDOS_sephasing_mi={m0i}_mf={mf_val}_alpha={alpha_val}_gamma={gamma_val}.png")
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
    mf_values = [-2.5, 0.5, 2.5]
    alpha_values = [0.3, 20.0]
    gamma_values = [0.0, 0.5, 1.0, 5.0, 10.0]
    parameter_combinations = list(itertools.product(mf_values, alpha_values, gamma_values))

    params = parameter_combinations[array_index]
    run_simulation(params)