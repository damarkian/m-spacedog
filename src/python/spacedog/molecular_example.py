import os
from typing import Tuple
from pathlib import Path

import numpy as np
import openfermion as of
import scipy as sp

from .gradient_hf import rhf_minimization
from .objective import (RestrictedHartreeFockObjective, generate_hamiltonian)


def make_h6_2_1() -> Tuple[RestrictedHartreeFockObjective,
                           of.MolecularData,
                           np.ndarray,
                           np.ndarray,
                           np.ndarray]:

    # load the molecule from moelcular data
    cwd = os.path.dirname(os.path.realpath(__file__))
    h6_2_1_path = str(Path(cwd 
                       + "/molecular_data/hydrogen_chains/h_6_sto-3g/bond_distance_2.1/")) 

    molfile = os.path.join(h6_2_1_path, 'H6_sto-3g_singlet_linear_r-2.1.hdf5')
    molecule = of.MolecularData(filename=molfile)
    molecule.load()

    S = np.load(os.path.join(h6_2_1_path, 'overlap.npy'))
    Hcore = np.load(os.path.join(h6_2_1_path, 'h_core.npy'))
    TEI = np.load(os.path.join(h6_2_1_path, 'tei.npy'))

    _, X = sp.linalg.eigh(Hcore, S)
    obi = of.general_basis_change(Hcore, X, (1, 0))
    tbi = np.einsum('psqr', of.general_basis_change(TEI, X, (1, 0, 1, 0)))
    molecular_hamiltonian = generate_hamiltonian(obi, tbi,
                                                 molecule.nuclear_repulsion)

    rhf_objective = RestrictedHartreeFockObjective(molecular_hamiltonian,
                                                   molecule.n_electrons)

    scipy_result = rhf_minimization(rhf_objective)

    return rhf_objective, molecule, scipy_result.x, obi, tbi
