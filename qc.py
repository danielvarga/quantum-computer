import sys
import numpy as np


def init(n):
    v = np.zeros(2 ** n)
    v[0] = 1
    return to_cube(v)


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


def test_tensordot():
    a = np.arange(12).reshape((2, 2, 3))
    b = np.arange(4).reshape((2, 2))
    print(a)
    print(b)
    print(np.tensordot(a, b, (1, 0)))


def test():
    # test_quantum_toffoli_gate()
    # test_quantum_not_gate()
    test_quantum_hadamard_gate()
    # test_classical_and_gate()
    # test_tensordot()
    # test_classical_random_bit()
    # test_classical_not_gate()


test()
