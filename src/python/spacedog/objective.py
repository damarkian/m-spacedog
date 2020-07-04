from typing import Optional
import numpy as np
from itertools import product
from openfermion.ops import InteractionOperator, InteractionRDM
from openfermion.utils import wedge
from openfermion.transforms import get_fermion_operator



class RhfObjective(): 

    def __init__(self, hamiltonian: InteractionOperator, num_electrons: int):
        self.hamiltonian = hamiltonian
        self.fermion_hamiltonian = get_fermion_operator(self.hamiltonian)
        self.num_qubits = hamiltonian.one_body_tensor.shape[0]
        self.num_orbitals = self.num_qubits // 2
        self.num_electrons = num_electrons
        self.nocc = self.num_electrons // 2
        self.nvirt = self.num_orbitals - self.nocc
        self.occ = list(range(self.nocc))
        self.virt = list(range(self.nocc, self.nocc + self.nvirt))


def generate_hamiltonian(obi: np.ndarray, tbi: np.ndarray, nuclear_repulsion: float, 
                         tolerance=1.0E-12):

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

    # hamiltonian = {}
    # hamiltonian['oneBodyCoefficients'] = obc
    # hamiltonian['twoBodyCoefficients'] = tbc 
    # hamiltonian['nuclearRepulsion'] = nuclear_repulsion

    molHamiltonian = InteractionOperator(nuclear_repulsion, obc, tbc)

    return molHamiltonian

