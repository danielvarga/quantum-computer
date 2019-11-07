from qc import *

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
    print("unimplemented")
    return
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


def test_tensordot():
    a = np.arange(12).reshape((2, 2, 3))
    b = np.arange(4).reshape((2, 2))
    print(a)
    print(b)
    print(np.tensordot(a, b, (1, 0)))


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


def test_quantum_symmetric_controlled_pauli_z_gate():
    n = 4
    v = init(n, [1] * n)
    v = full_hadamard(v, range(n))
    w = quantum_symmetric_controlled_pauli_z_gate(v, range(n))
    for _ in range(100):
        gates = np.random.permutation(4)
        y = quantum_symmetric_controlled_pauli_z_gate(v, gates)
        print(y)
        assert np.allclose(w, y)


def test_sat_circuit():
    n = 2
    # (1, 0) is the only assignment satisfying it:
    clauses = [[+1, 0], [0, -1]]
    c = len(clauses)
    dim = n + c + 1
    pad = [0 for _ in range(c + 1)]
    for bitvector in itertools.product(*[[0, 1] for _ in range(n)]):
        v = init(dim, list(bitvector) + pad)
        v = apply_sat_circuit(v, clauses)
        v = quantum_pauli_z_gate(v, dim - 1)
        v = unapply_sat_circuit(v, clauses)
        print(bitvector, deterministic_get(v))


def test_grover_search():
    # (1, 0) is the only assignment satisfying it:
    clauses = [[+1, 0], [0, -1]]
    iterations = 2
    v = grover_search(clauses, iterations)
    dump_amplitudes(v, sort=True)


def test():
    test_grover_search()
    test_quantum_symmetric_controlled_pauli_z_gate()
    test_sat_circuit()
    test_quantum_controlled_pauli_x_gate()
    test_quantum_toffoli_gate()
    test_quantum_not_gate()
    test_quantum_hadamard_gate()
    test_classical_and_gate()
    test_tensordot()
    test_classical_random_bit()
    test_classical_not_gate()


if __name__ == "__main__":
    test()
