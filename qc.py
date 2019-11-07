import sys
import itertools
import numpy as np


def init(n, value=None):
    if value is None:
        value = [0 for _ in range(n)]
    else:
        assert len(value) == n
    v = np.zeros(2 ** n)
    v = to_cube(v)
    v[tuple(value)] = 1
    return v


def qubits(v):
    return len(v.shape)


def to_cube(v):
    n = v.size.bit_length() - 1
    return v.reshape([2 for _ in range(n)])


# bits[k] = not bits[k]
# view not copy!
def classical_not_gate(v, k):
    return np.flip(v, k)


# p = probability of 1.
def classical_random_bit(v, k, p=0.5):
    proj = np.sum(v, axis=k, keepdims=True) # forgetting
    return np.concatenate((proj * (1 - p), proj * p), axis=k)


def apply_quantum_gate(tensor, v, bits):
    b = len(bits)
    swapped = np.tensordot(tensor, v, (list(range(b)), bits))
    # tensordot puts the results at the beginning instead of in-place. let's move them back by a transpose.
    # based on https://discuss.pennylane.ai/t/batching-inputs-to-quantum-circuit/139/10
    unaffected = [idx for idx in range(qubits(v)) if idx not in bits]
    perm = bits + unaffected # perm is what happened to the coordinates.
    inv_perm = np.argsort(perm) # inv_perm will make it right.
    return np.transpose(swapped, inv_perm)


def quantum_hadamard_gate(v, k):
    hadamard_matrix = np.sqrt(0.5) * np.array([[1, 1], [1, -1]])
    return apply_quantum_gate(hadamard_matrix, v, [k])


# aka quantum_pauli_x_gate
def quantum_not_gate(v, k):
    not_matrix = np.array([[0, 1], [1, 0]])
    return apply_quantum_gate(not_matrix, v, [k])


def quantum_pauli_z_gate(v, k):
    pauli_z_matrix = np.array([[1, 0], [0, -1]])
    return apply_quantum_gate(pauli_z_matrix, v, [k])


# bits[k3] is flipped if (bits[k1] and bits[k2]).
def quantum_toffoli_gate(v, k1, k2, k3):
    toffoli_matrix = np.eye(8)
    toffoli_matrix[-2:, -2:] = np.array([[0, 1], [1, 0]])
    toffoli_tensor = toffoli_matrix.reshape((2, 2, 2, 2, 2, 2))
    return apply_quantum_gate(toffoli_tensor, v, [k1, k2, k3])


# clause is a +1 -1 0 vector on the qubits, controlled is one of the qubits.
# for a basis, if the clause is true, then the effect operator is
# applied to the controlled qubit. effect is a 2x2 unitary matrix.
def quantum_controlled_gate(v, clause, controlled, effect):
    assert qubits(v) == len(clause)
    assert clause[controlled] == 0
    bits = [i for i, l in enumerate(clause) if l != 0] + [controlled]
    controlled_matrix = np.eye(2 ** len(bits))
    controlled_tensor = controlled_matrix.reshape(tuple(2 for _ in range(2 * len(bits))))
    slicer = [1 if literal == +1 else 0 for literal in clause if literal !=0]
    controlled_tensor[tuple(slicer + [slice(0, 2)] + slicer + [slice(0, 2)])] = effect
    return apply_quantum_gate(controlled_tensor, v, bits)


# the quantum version of conditional NOT.
# clause is a +1 -1 0 vector on the qubits, controlled is one of the qubits.
# if the clause is true, then the controlled qubit is flipped,
# otherwise it's unaffected.
def quantum_controlled_pauli_x_gate(v, clause, controlled):
    not_matrix = np.array([[0, 1], [1, 0]])
    return quantum_controlled_gate(v, clause, controlled, not_matrix)


# conditional phase shift by 180 degrees.
# clause is a +1 -1 0 vector on the qubits, controlled is one of the qubits.
# if the clause is true, then the controlled qubit is phase shifted,
# otherwise it's unaffected.
def quantum_controlled_pauli_z_gate(v, clause, controlled):
    pauli_z_matrix = np.array([[1, 0], [0, -1]])
    return quantum_controlled_gate(v, clause, controlled, pauli_z_matrix)


# this is equivalent to picking any of the bits as controlled,
# and using the rest as positive literals in the control.
# see https://algassert.com/post/1706
def quantum_symmetric_controlled_pauli_z_gate(v, bits):
    pauli_z_matrix = np.array([[1, 0], [0, -1]])
    controlled = bits[-1]
    controls = bits[:-1]
    clause = [1 if i in controls else 0 for i in range(qubits(v))]
    return quantum_controlled_gate(v, clause, controlled, pauli_z_matrix)


def deterministic_get(v):
    n = qubits(v)
    assert np.sum(v != 0) == 1
    for bitvector in itertools.product(*[[0, 1] for _ in range(n)]):
        if v[tuple(bitvector)] != 0:
            return bitvector, v[tuple(bitvector)]


# the first n qubits are assumed to be the variables,
# the second c qubits correspond to the clauses,
# and are assumed to be initialized to all zeros.
# the last single qubit is the result of the sat evaluation.
def apply_sat_circuit(v, clauses):
    n = len(clauses[0])
    c = len(clauses)
    dim = n + c + 1
    assert dim == qubits(v) # could be more flexible, but it's enough for our purposes
    pad = [0 for _ in range(c + 1)]
    for i, clause in enumerate(clauses):
        v = quantum_controlled_pauli_x_gate(v, clause + pad, n + i)
    v = quantum_controlled_pauli_x_gate(v, [1 if n <= i < n + c else 0 for i in range(dim)], dim - 1)
    return v


def unapply_sat_circuit(v, clauses):
    n = len(clauses[0])
    c = len(clauses)
    dim = n + c + 1
    assert dim == qubits(v)
    pad = [0 for _ in range(c + 1)]
    v = quantum_controlled_pauli_x_gate(v, [1 if n <= i < n + c else 0 for i in range(dim)], dim - 1)
    for i, clause in list(enumerate(clauses))[::-1]:
        v = quantum_controlled_pauli_x_gate(v, clause + pad, n + i)
    return v


def full_hadamard(v, bits):
    for i in bits:
        v = quantum_hadamard_gate(v, i)
    return v


def grover_diffusion_operator(v, n):
    v = full_hadamard(v, range(n))
    v = quantum_symmetric_controlled_pauli_z_gate(v, range(n))
    v = full_hadamard(v, range(n))
    return v


def grover_iteration(v, clauses):
    n = len(clauses[0])
    c = len(clauses)
    dim = n + c + 1
    assert dim == qubits(v)
    pad = [0 for _ in range(c + 1)]

    v = apply_sat_circuit(v, clauses)
    v = quantum_pauli_z_gate(v, dim - 1)
    v = unapply_sat_circuit(v, clauses)
    v = grover_diffusion_operator(v, n)
    return v


def grover_search(clauses, iterations):
    n = len(clauses[0])
    c = len(clauses)
    dim = n + c + 1
    v = init(dim)
    v = full_hadamard(v, range(n))
    for j in range(iterations):
        v = grover_iteration(v, clauses)
    v = apply_sat_circuit(v, clauses)
    return v


def dump_amplitudes(v, sort=False):
    dim = qubits(v)
    collect = []
    for bitvector in itertools.product(*[[0, 1] for _ in range(dim)]):
        if v[tuple(bitvector)] != 0:
            collect.append((np.absolute(v[tuple(bitvector)]) ** 2, bitvector, v[tuple(bitvector)]))
    if sort:
        collect.sort(reverse=True)
    for prob, bitvector, amplitude in collect:
        if prob > 0:
            print("".join(map(str, bitvector)), prob, amplitude)
