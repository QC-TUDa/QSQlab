import numpy as np

SINGLE_QUBIT_GATES = {
    # Identity & Pauli
    "id", "x", "y", "z",

    # Rotations
    "rx", "ry", "rz",

    # Clifford gates
    "h", "s", "sdg",

    # T gates
    "t", "tdg",

    # sqrt(X)
    "sx", "sxdg",

    # Phase / general
    "p",
    "u", "u1", "u2", "u3",
}

MULTI_QUBIT_GATES = {
    # Controlled Pauli / Clifford
    "cx", "cy", "cz",
    "ch",

    "cp", "cu",
    "cu1", "cu2", "cu3",

    # Two-qubit rotations
    "rxx", "ryy", "rzz", "rzx",

    # Swap family
    "swap", "iswap", "cswap",

    # Toffoli / multi-control
    "ccx",  # Toffoli
    "mcx",
    "mcy",
    "mcz",

    # General multi-qubit unitary
    "unitary",
}

NON_UNITARY_OPS = {
    "measure",
    "reset",
    "barrier",
    "delay",
}

ALL_QUANTUM_GATES = SINGLE_QUBIT_GATES | MULTI_QUBIT_GATES

SU2 = {
    "X": np.array([[0, 1], [1, 0]]),    
    "Y": np.array([[0, -1j], [1j, 0]]),   
    "Z": np.array([[1, 0], [0, -1]])      
}

