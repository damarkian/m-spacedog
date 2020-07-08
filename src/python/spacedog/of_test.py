import os
from pathlib import Path
from typing import Tuple
import numpy as np
import scipy.linalg
import openfermion as of
import json

from .gradient_hf import rhf_func_generator
from .opdm_functionals import OpdmFunctional
from .analysis import (compute_opdm,
                            mcweeny_purification,
                            resample_opdm,
                            fidelity_witness,
                            fidelity)
from .third_party.higham import fixed_trace_positive_projection


from .gradient_hf import rhf_minimization
from .objective import (RestrictedHartreeFockObjective, generate_hamiltonian)
from .molecular_example_odd_qubits import make_h3_2_5

def outputjson(_teststr, jsonfile):
    _jsondict = {}
    _jsondict["result"] = _teststr
    _jsondict["schema"] = "spacedog-result"
    with open(jsonfile, 'w') as f:
        f.write(json.dumps(_jsondict))

    

def of_test(jsonfile):

    rhf_objective, molecule, parameters, obi, tbi = make_h3_2_5()
    ansatz, energy, gradient = rhf_func_generator(rhf_objective)

    opd = {}
    opd["energy"] = str(energy)
    opd["schema"] = "spacedog-result"

    with open(jsonfile, 'w') as f:
        f.write(json.dumps(opd))
    #return(opd)


