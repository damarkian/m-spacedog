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

def of_test():
    cwd = os.path.dirname(os.path.realpath(__file__))
    moleculeFile = str(Path(cwd + "/moldata/h3_250.hdf5")) 
    molecule = of.MolecularData(filename=moleculeFile)
    molecule.load()
    e_nuc = molecule.nuclear_repulsion
    return(e_nuc)


