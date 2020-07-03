import os
from pathlib import Path
import numpy as np
import scipy.linalg
import openfermion as of
import json
from .objective import generate_hamiltonian

def outputjson(_teststr, jsonfile):
    _jsondict = {}
    _jsondict["result"] = _teststr
    _jsondict["schema"] = "spacedog-result"
    with open(jsonfile, 'w') as f:
        f.write(json.dumps(_jsondict))


def of_test(jsonfile):
    cwd = os.path.dirname(os.path.realpath(__file__))
    moleculeFile = str(Path(cwd + "/moldata/h3_250.hdf5")) 
    molecule = of.MolecularData(filename=moleculeFile)
    molecule.load()

    # TODO: separate all these steps to produce one dictionary entry per step

    e_nuc = molecule.nuclear_repulsion

    S = np.load(str(Path(cwd + "/moldata/overlap.npy")))
    Hcore = np.load(str(Path(cwd + "/moldata/h_core.npy")))
    TwoERI = np.load(str(Path(cwd + "/moldata/two_eri.npy")))

    _, X = scipy.linalg.eigh(Hcore, S)

    obi = of.general_basis_change(Hcore, X, (1, 0))
    tbi = np.einsum('psqr', of.general_basis_change(TwoERI, X, (1, 0, 1, 0)))

    hamiltonian = generate_hamiltonian(obi, tbi, molecule.nuclear_repulsion, 1.0e-12)

    opd = {}
    opd["eNuc"] = str(e_nuc)
    opd["overlapMatrix"] = np.array2string(S)
    opd["orthoX"] = np.array2string(X)
    opd["schema"] = "spacedog-result"

    with open(jsonfile, 'w') as f:
        f.write(json.dumps(opd))
    #return(opd)


