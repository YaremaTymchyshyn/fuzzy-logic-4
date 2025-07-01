import numpy as np


def load_relations():
    R1 = np.array([
        [1.0, 0.8, 0.2, 0.5, 0.0],
        [0.8, 1.0, 0.0, 0.3, 0.5],
        [0.2, 0.0, 1.0, 0.1, 0.1],
        [0.5, 0.3, 0.1, 1.0, 0.0],
        [0.0, 0.5, 0.5, 0.4, 1.0]
    ])
    R2 = np.array([
        [1.0, 0.0, 0.0, 0.0, 0.0],
        [0.8, 1.0, 0.0, 0.0, 0.0],
        [0.2, 0.0, 1.0, 0.0, 0.0],
        [0.5, 0.0, 0.2, 1.0, 0.0],
        [0.0, 0.4, 0.1, 0.4, 1.0]
    ])
    return R1, R2


def fuzzy_union(R, S): return np.maximum(R, S)
def fuzzy_intersection(R, S): return np.minimum(R, S)
def fuzzy_difference(R, S): return np.minimum(R, 1.0 - S)
def fuzzy_sym_diff(R, S): return fuzzy_union(R, S) - fuzzy_intersection(R, S)
def complement(R): return 1.0 - R


def max_min_composition(R, S):
    n = R.shape[0]
    return np.array([[np.max(np.minimum(R[i, :], S[:, j])) for j in range(n)] for i in range(n)])


def max_prod_composition(R, S):
    n = R.shape[0]
    return np.array([[np.max(R[i, :] * S[:, j]) for j in range(n)] for i in range(n)])


def is_reflexive(R): return np.allclose(np.diag(R), 1.0)
def is_irreflexive(R): return np.allclose(np.diag(R), 0.0)
def is_symmetric(R): return np.allclose(R, R.T)
def is_antisymmetric(R): return all(R[i, j] == 0 or R[j, i] == 0 for i in range(5) for j in range(5) if i != j)
def is_asymmetric(R): return is_irreflexive(R) and is_antisymmetric(R)
def is_transitive(R): return np.all(R >= max_min_composition(R, R))


def display(name, M):
    print(f"\n===== {name} =====")
    with np.printoptions(precision=2, suppress=True):
        print(M)


if __name__ == '__main__':
    R1, R2 = load_relations()

    display("R1", R1)
    display("R2", R2)
    display("Union", fuzzy_union(R1, R2))
    display("Intersection", fuzzy_intersection(R1, R2))
    display("Difference", fuzzy_difference(R1, R2))
    display("Symmetric difference", fuzzy_sym_diff(R1, R2))
    display("Complement R1", complement(R1))
    display("Complement R2", complement(R2))
    display("Max-min composition", max_min_composition(R1, R2))
    display("Max-prod composition", max_prod_composition(R1, R2))

    print("\n===== Properties of R1 =====")
    print("Reflexive:", is_reflexive(R1))
    print("Irreflexive:", is_irreflexive(R1))
    print("Symmetric:", is_symmetric(R1))
    print("Antisymmetric:", is_antisymmetric(R1))
    print("Asymmetric:", is_asymmetric(R1))
    print("Transitive:", is_transitive(R1))
