import os
from pathlib import Path
from openfermion.hamiltonians import MolecularData
from openfermionpsi4 import run_psi4

import h5py
import numpy as np

def p4_test(jsonfile):

    description = "CuH200"
    filename = "CuH200.hdf5"
    geometry = [['Cu', [0.0, 0.0, 0.0]], ['H', [0.0, 0.0, 2.0]]]
    basis = '6-31g'
    charge = 0
    multiplicity = 1
    
    molecule = MolecularData(geometry, basis, multiplicity, charge, description, filename=filename)

    molecule = run_psi4(molecule, run_scf=1, run_mp2=1, run_cisd=0, run_ccsd=0, run_fci=0, verbose=1, tolerate_error=1)

    molecule.save()    

    opd = {}
    opd["hfenergy"] = str(molecule.hf_energy)
    opd["mp2energy"] = str(molecule.mp2_energy)
    opd["schema"] = "spacedog-result"

    with open(jsonfile, 'w') as f:
        f.write(json.dumps(opd))