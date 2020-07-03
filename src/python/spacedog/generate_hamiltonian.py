import numpy as np
from itertools import product

def generate_hamiltonian(obi, tbi, nuclear_repulsion, tolerance):

    n_qubits = 2 * obi.shape[0]

    # Initialize Hamiltonian coefficients

    obc = np.zeros((n_qubits, n_qubits))
    tbc = np.zeros((n_qubits, n_qubits, n_qubits, n_qubits))

    # Loop through integrals
    for p in range(n_qubits // 2):
        for q in range(n_qubits // 2):

            obc[2*p, 2*q]         = obi[p, q]
            obc[2*p + 1, 2*q + 1] = obi[p, q]

            for r in range(n_qubits // 2): 
                for s in range(n_qubits // 2): 
                    # different spin
                    tbc[2*p,   2*q+1, 2*r+1, 2*s]   = (tbi[p, q, r, s] / 2.0)
                    tbc[2*p+1, 2*q,   2*r,   2*s+1] = (tbi[p, q, r, s] / 2.0) 
                    # same spin
                    tbc[2*p,   2*q,   2*r,   2*s]   = (tbi[p, q, r, s] / 2.0) 
                    tbc[2*p+1, 2*q+1, 2*r+1, 2*s+1] = (tbi[p, q, r, s] / 2.0) 

    obc[np.absolute(obc) < tolerance] = 0.0
    tbc[np.absolute(tbc) < tolerance] = 0.0

    hamiltonian = {}
    hamiltonian['oneBodyCoefficients'] = obc
    hamiltonian['twoBodyCoefficients'] = tbc 
    hamiltonian['nuclearRepulsion'] = nuclear_repulsion

    return hamiltonian

