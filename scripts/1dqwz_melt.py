import numpy as np
import matplotlib.pyplot as plt
import time
import itertools
import sys
from scipy.linalg import eigh

def run_simulation(params):
    mf_val, alpha_val, gamma_val = params

    # system parameters
    Lx = 6
    Ly = 1
    A = Lx * Ly
    N = 2 * A
    n_particles = N // 2 + 1  # half-filling + 1 

    # Enumerating many-body basis states
    basis_states = []
    state_to_index = {}
    for bits in itertools.combinations(range(N), n_particles):
        state = np.zeros(N, dtype=int)
        state[list(bits)] = 1
        basis_states.append(state)
        state_to_index[tuple(state)] = len(basis_states) - 1

    hilbert_dim = len(basis_states)

    # Pauli matrices
    sigma_x = np.array([[0, 1], [1, 0]], dtype=complex)
    sigma_y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    sigma_z = np.array([[1, 0], [0, -1]], dtype=complex)
    sigma_0 = np.eye(2)

    # Hopping matrices
    SX1D = np.zeros((A, A), dtype=complex)
    CX1D = np.zeros((A, A), dtype=complex)
    for i in range(A - 1):
        SX1D[i, i + 1] = 0.5j
        SX1D[i + 1, i] = -0.5j
        CX1D[i, i + 1] = 0.5
        CX1D[i + 1, i] = 0.5

    I_A = np.eye(A)

    # Static part (ky = 0)
    ky = 0
    M = np.cos(ky) * sigma_z + np.sin(ky) * sigma_y
    Ht_sp = np.kron(I_A, sigma_z)
    H0_sp = np.kron(I_A, M) + np.kron(SX1D, sigma_x) + np.kron(CX1D, sigma_z)

    # Mapping single-particle Hamiltonian to many-body
    def build_many_body_hamiltonian(H_sp):
        H_MB = np.zeros((hilbert_dim, hilbert_dim), dtype=complex)
        for i in range(N):
            for j in range(N):
                if abs(H_sp[i, j]) < 1e-14:
                    continue
                if i == j:
                    for k, state in enumerate(basis_states):
                        if state[i] == 1:
                            H_MB[k, k] += H_sp[i, i]
                else:
                    for k, state in enumerate(basis_states):
                        if state[j] == 1 and state[i] == 0:
                            new_state = state.copy()
                            new_state[j] = 0
                            new_state[i] = 1
                            sign = (-1) ** np.sum(state[min(i, j)+1:max(i, j)])
                            l = state_to_index[tuple(new_state)]
                            H_MB[k, l] += H_sp[i, j] * sign
        return H_MB

    # Building the Many-Body Hamiltonian
    H_MB_0 = build_many_body_hamiltonian(H0_sp)
    H_MB_t = build_many_body_hamiltonian(Ht_sp)

    # Initial density matrix from starting m
    mi_val = 0.25
    H_MB_i = H_MB_0 - mi_val * H_MB_t
    E_i, V_i = eigh(H_MB_i)
    psi_i = V_i[:, 0]
    rho = np.outer(psi_i, psi_i.conj())

    # Reference ground state for survival probability
    m0 = mi_val
    H_MB_ref = H_MB_0 - m0 * H_MB_t
    E_ref, V_ref = eigh(H_MB_ref)
    psi_0 = V_ref[:, 0]

    # Time evolution parameters
    T = 20
    dt = 0.0001
    steps = int(T / dt)
    
    P = np.zeros(steps)

    def m_ramp(t, mi, mf, alpha):
        return mi + (mf - mi) * (1 - np.exp(-alpha * t))

    # Precomputed number operators for LDOS 
    num_ops = []
    for i in range(A):
        for beta in [0, 1]:
            orb = 2 * i + beta
            n_op = np.zeros((hilbert_dim, hilbert_dim), dtype=complex)
            for b_idx, state in enumerate(basis_states):
                if state[orb] == 1:
                    n_op[b_idx, b_idx] = 1.0
            num_ops.append(n_op)

    # Sampling instants
    time_instants = np.linspace(0, T, 21)
    sample_indices = np.round(time_instants / dt).astype(int)
    density_at_instants = []

    # Evolution loop
    start = time.time()
    for n in range(steps):
        t = n * dt
        m_val = m_ramp(t, mi_val, mf_val, alpha_val)

       
        H_MB = H_MB_0 - m_val * H_MB_t

        C1 = H_MB @ rho - rho @ H_MB
        C2 = H_MB @ C1 - C1 @ H_MB
        rho = rho - dt * 1j * C1 + dt * (-gamma_val * C2)

        # Survival probability
        P[n] = np.real(np.vdot(psi_0, rho @ psi_0))

        # LDOS snapshots
        if n in sample_indices:
            density = np.zeros(A)
            for i in range(A):
                density[i] = sum(np.real(np.trace(rho @ num_ops[2*i + beta]))
                                 for beta in [0, 1])
            density_at_instants.append(density)

    end = time.time()
    total_time = end - start

    # results
    np.savetxt(f"MB_P_mf={mf_val}_alpha={alpha_val}_gamma={gamma_val}.txt", P)
    np.savetxt(f"MB_densities_mf={mf_val}_alpha={alpha_val}_gamma={gamma_val}.txt",
               np.array(density_at_instants))

    # subset of LDOS profiles
    plot_indices = list(range(0, len(density_at_instants), 2))
    fig, axs = plt.subplots(1, len(plot_indices), figsize=(20, 4), sharey=True)
    for ax, idx in zip(axs, plot_indices):
        ax.plot(density_at_instants[idx], 'o-')
        ax.set_title(f"t = {np.round(time_instants[idx], 1)}")
        ax.set_xlabel("Site index")
        ax.set_ylabel("Density")
    plt.tight_layout()
    plt.savefig(f"MB_density_profiles_subset_mf={mf_val}_alpha={alpha_val}_gamma={gamma_val}.png")
    plt.close(fig)


# Cluster compatible main 
if __name__ == '__main__':
    
    # Handle HPC array index or run locally
    if len(sys.argv) > 1:
        array_index = int(sys.argv[1])
    else:
        print("No array index passed. Running with array_index = 0.")
        array_index = 0

    # Parameter combinations
    mf_values = [2.25, -0.25]
    alpha_values = [0.3, 5.0]
    gamma_values = [10, 15, 20, 25]

    
    parameter_combinations = list(itertools.product(mf_values, alpha_values, gamma_values))
    params = parameter_combinations[array_index]
    run_simulation(params)
