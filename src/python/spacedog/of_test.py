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
from .molecular_example import make_h6_2_1

from .mfopt import moving_frame_augmented_hessian_optimizer
from .opdm_functionals import RDMGenerator
#import matplotlib.pyplot as plt



def outputjson(_teststr, jsonfile):
    _jsondict = {}
    _jsondict["result"] = _teststr
    _jsondict["schema"] = "spacedog-result"
    with open(jsonfile, 'w') as f:
        f.write(json.dumps(_jsondict))




def of_test(jsonfile):

    rhf_objective, molecule, parameters, obi, tbi = make_h6_2_1()
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

    measurement_data = opdm_func.calculate_data(parameters)

  
    opdm, var_dict = compute_opdm(measurement_data,
                                return_variance=True)
    opdm_pure = mcweeny_purification(opdm)


   
    raw_energies = []
    raw_fidelity_witness = []
    purified_eneriges = []
    purified_fidelity_witness = []
    purified_fidelity = []
    true_unitary = ansatz(parameters)
    nocc = molecule.n_electrons // 2
    nvirt = molecule.n_orbitals - nocc

    initial_fock_state = [1] * nocc + [0] * nvirt
    for _ in range(1000):  # 1000 repetitions of the measurement
        new_opdm = resample_opdm(opdm, var_dict)
        raw_energies.append(opdm_func.energy_from_opdm(new_opdm))
        raw_fidelity_witness.append(
            fidelity_witness(target_unitary=true_unitary,
                            omega=initial_fock_state,
                            measured_opdm=new_opdm)
        )
        # fix positivity and trace of sampled 1-RDM if strictly outside
        # feasible set
        w, v = np.linalg.eigh(new_opdm)
        if len(np.where(w < 0)[0]) > 0:
            new_opdm = fixed_trace_positive_projection(new_opdm, nocc)

        new_opdm_pure = mcweeny_purification(new_opdm)
        purified_eneriges.append(opdm_func.energy_from_opdm(new_opdm_pure))
        purified_fidelity_witness.append(
            fidelity_witness(target_unitary=true_unitary,
                            omega=initial_fock_state,
                            measured_opdm=new_opdm_pure)
        )
        purified_fidelity.append(
            fidelity(target_unitary=true_unitary,
                    measured_opdm=new_opdm_pure)
        )

    rdm_generator = RDMGenerator(opdm_func, purification=True)
    opdm_generator = rdm_generator.opdm_generator

    result = moving_frame_augmented_hessian_optimizer(
        rhf_objective=rhf_objective,
        initial_parameters= parameters + 5.0E-1 ,
        opdm_aa_measurement_func=opdm_generator,
        verbose=True, delta=0.03,
        max_iter=120,
        hessian_update='diagonal',
        rtol=0.050E-2)

    resultstring = np.array2string(np.array(result.func_vals))

    opd = {}
    opd["energy"] = str(molecule.hf_energy)
    opd["trueEnergy"] = str(energy(parameters))
    opd["resultFuncVals"] = resultstring
    opd["schema"] = "spacedog-result"

    with open(jsonfile, 'w') as f:
        f.write(json.dumps(opd))
    #return(opd)


