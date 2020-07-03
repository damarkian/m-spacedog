import os
from pathlib import Path
import numpy as np
import scipy.linalg
import openfermion as of
import json

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
    Hcore = np.load(str(Path(cwd + "/moldata/hcore.npy")))
    TwoERI = np.load(str(Path(cwd + "/moldata/two_eri.npy")))

    _, X = scipy.linalg.eigh(Hcore, S)

    opd = {}
    opd["eNuc"] = str(e_nuc)
    opd["overlapMatrix"] = np.array2string(S)
    opd["schema"] = "spacedog-result"

    with open(jsonfile, 'w') as f:
        f.write(json.dumps(opd))
    #return(opd)


