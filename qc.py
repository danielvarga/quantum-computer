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
    assert np.sum(v) == 1
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


# bits[k1] = bits[k2]
def classical_copy_gate(v, k1, k2):
    proj = np.sum(v, axis=k1, keepdims=True)


# bits[k1] = bits[k1] and bits[k2]
def classical_and_gate(v, k1, k2):
    n = qubits(v)


# the quantum version of conditional NOT.
# clause is a +1 -1 0 vector on the qubits, controlled is one of the qubits.
# for a basis, if the clause is true, then the controlled qubit is flipped,
# otherwise it's unaffected.
def quantum_controlled_pauli_x_gate(v, clause, controlled):
    assert qubits(v) == len(clause)
    assert clause[controlled] == 0
    bits = [i for i, l in enumerate(clause) if l != 0] + [controlled]
    controlled_pauli_x_matrix = np.eye(2 ** len(bits))
    controlled_pauli_x_tensor = controlled_pauli_x_matrix.reshape(tuple(2 for _ in range(2 * len(bits))))
    slicer = [1 if literal == +1 else 0 for literal in clause if literal !=0]
    controlled_pauli_x_tensor[tuple(slicer + [slice(0, 2)] + slicer + [slice(0, 2)])] = np.array([[0, 1], [1, 0]])
    return apply_quantum_gate(controlled_pauli_x_tensor, v, bits)


def deterministic_get(v):
    n = qubits(v)
    assert np.sum(v != 0) == 1
    for bitvector in itertools.product(*[[0, 1] for _ in range(n)]):
        if v[tuple(bitvector)] != 0:
            return bitvector


def test_classical_not_gate():
    v = init(3)
    for k in range(3):
        print(k)
        print(classical_not_gate(v, k))
        print("=====")


def test_classical_random_bit():
    v = init(3)
    for i in range(3):
        v = classical_random_bit(v, i)
        print(i)
        print(v)
        print("=====")


def test_classical_and_gate():
    v = init(3)
    v = classical_not_gate(v, 0)
    v = classical_not_gate(v, 1)
    for pair in ((0,1), (1,2), (2,0)):
        k1, k2 = pair
        print(k1, k2)
        print(classical_and_gate(v, k1, k2))
        print("=====")

    v = init(3)
    for i in range(3):
        v = classical_random_bit(v, i)
    for pair in ((0,1), (1,2), (2,0)):
        k1, k2 = pair
        print(k1, k2)
        print(classical_and_gate(v, k1, k2))
        print("=====")


def test_quantum_hadamard_gate():
    v = init(3)
    for i in range(3):
        print(i)
        print(quantum_hadamard_gate(v, i))
        print("=====")


def test_quantum_not_gate():
    v = init(3)
    v = quantum_hadamard_gate(v, 1)
    for i in range(3):
        print(i)
        print(quantum_not_gate(v, i) - classical_not_gate(v, i))
        print("=====")


def test_quantum_toffoli_gate():
    for i in range(2):
        print(i)
        v = init(3)
        v = quantum_not_gate(v, 0)
        if i == 1:
            v = quantum_not_gate(v, 1)
        print("before:")
        print(v)
        print("after:")
        print(quantum_toffoli_gate(v, 0, 1, 2))
        print("=====")


# controlled_pauli_x is a generalization of toffoli,
# so we verify it by comparing it to toffoli in the special case.
# either that, or we verify toffoli by comparing it to pauli, who cares.
def test_quantum_controlled_pauli_x_gate():
    v = init(3)
    for b1 in (0, 1):
        for b2 in (0, 1):
            v = init(3, (b1, b2, 0))
            vnot = quantum_controlled_pauli_x_gate(v, (+1, +1, 0), 2)
            vtof = quantum_toffoli_gate(v, 0, 1, 2)
            assert np.allclose(vnot, vtof)


def test_tensordot():
    a = np.arange(12).reshape((2, 2, 3))
    b = np.arange(4).reshape((2, 2))
    print(a)
    print(b)
    print(np.tensordot(a, b, (1, 0)))


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


def test_sat_circuit():
    n = 4
    # (1, 0, 1, 0) is the only assignment satisfying it:
    clauses = [[+1, -1, +1, 0], [0, 0, +1, -1]]
    c = len(clauses)
    dim = n + c + 1
    pad = [0 for _ in range(c + 1)]
    for bitvector in itertools.product(*[[0, 1] for _ in range(n)]):
        v = init(dim, list(bitvector) + pad)
        v = apply_sat_circuit(v, clauses)
        v = quantum_pauli_z_gate(v, dim - 1)
        v = unapply_sat_circuit(v, clauses)
        print(bitvector, deterministic_get(v))


def test():
    test_sat_circuit()
    # test_quantum_controlled_pauli_x_gate()
    # test_quantum_toffoli_gate()
    # test_quantum_not_gate()
    # test_quantum_hadamard_gate()
    # test_classical_and_gate()
    # test_tensordot()
    # test_classical_random_bit()
    # test_classical_not_gate()


test()
