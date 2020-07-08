import os
from pathlib import Path
from typing import Tuple
import numpy as np
import scipy.linalg
import openfermion as of
import json
import cirq

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

    # settings for quantum resources
    qubits = [cirq.GridQubit(0, x) for x in range(molecule.n_orbitals)]
    sampler = cirq.Simulator(dtype=np.complex128)  # this can be a QuantumEngine

    # OpdmFunctional contains an interface for running experiments
    opdm_func = OpdmFunctional(qubits=qubits,
                               sampler=sampler,
                               constant=molecule.nuclear_repulsion,
                               one_body_integrals=obi,
                               two_body_integrals=tbi,
                               num_electrons=molecule.n_electrons // 2,  # only simulate spin-up electrons
                               clean_xxyy=True,
                               purification=True,
                               verbose=True
                               )

    opd = {}
    opd["energy"] = str(rhf_objective)
    opd["schema"] = "spacedog-result"

    with open(jsonfile, 'w') as f:
        f.write(json.dumps(opd))
    #return(opd)


