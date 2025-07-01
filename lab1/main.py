import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d, lagrange


def plot_fuzzy_sets(x, A=None, B=None, C=None, title="Fuzzy Set",
                    colors=('bo-', 'ro-', 'mo-'), labels=('μA(x)', 'μB(x)', 'μC(x)')):
    plt.figure(figsize=(8, 5))

    used_colors = []
    used_labels = []

    if A is not None:
        plt.plot(x, A, colors[len(used_colors)], label=labels[len(used_labels)])
        used_colors.append(colors[len(used_colors)])
        used_labels.append(labels[len(used_labels)])

    if B is not None:
        plt.plot(x, B, colors[len(used_colors)], label=labels[len(used_labels)])
        used_colors.append(colors[len(used_colors)])
        used_labels.append(labels[len(used_labels)])

    if C is not None:
        plt.plot(x, C, colors[len(used_colors)], label=labels[len(used_labels)])

    plt.xlabel('x')
    plt.ylabel('Membership')
    plt.title(title)
    plt.legend()
    plt.grid()
    plt.show()


def fuzzy_approximation_and_plot(x, A, B):
    def fuzzy_approximation(x, y, label):
        poly = lagrange(x, y)
        print(f"\nAnalytical representation of {label}:")
        print(np.poly1d(poly))
        return poly

    A_poly = fuzzy_approximation(x, A, "μA(x)")
    B_poly = fuzzy_approximation(x, B, "μB(x)")

    x_dense = np.linspace(1, 8, 100)
    A_approx = A_poly(x_dense)
    B_approx = B_poly(x_dense)

    plt.figure(figsize=(8, 5))
    plt.plot(x, A, 'bo', label='μA(x) (Original)')
    plt.plot(x_dense, A_approx, 'b-', label='μA(x) (Approximation)')
    plt.plot(x, B, 'ro', label='μB(x) (Original)')
    plt.plot(x_dense, B_approx, 'r-', label='μB(x) (Approximation)')

    plt.xlabel('x')
    plt.ylabel('Membership')
    plt.title("Fuzzy Sets A and B with Approximation")
    plt.legend()
    plt.grid()
    plt.show()


def fuzzy_operations(x, A, B):
    A_complement = 1 - np.array(A)
    B_complement = 1 - np.array(B)

    union = np.maximum(A, B)
    intersection = np.minimum(A, B)

    difference = np.maximum(A - B, 0)
    symmetric_difference = np.abs(A - B)

    algebraic_intersection = A * B
    algebraic_union = A + B - A * B

    concentration_A = np.square(A)
    concentration_B = np.square(B)

    dilution_A = np.sqrt(A)
    dilution_B = np.sqrt(B)

    boundary_intersection = np.minimum(A, 1 - B)
    boundary_union = np.maximum(A, 1 - B)

    return {
        "A̅": (A_complement, None, None, 'Fuzzy Set A̅',
               ('bo-',), ('μA̅(x)',)),
        "B̅": (None, B_complement, None, 'Fuzzy Set B̅',
               ('ro-',), ('μB̅(x)',)),
        "A ∪ B": (A, B, union, 'Fuzzy Set A ∪ B',
                  ('bo-', 'ro-', 'mo-'), ('μA(x)', 'μB(x)', 'μC(x)')),
        "A ∩ B": (A, B, intersection, 'Fuzzy Set A ∩ B',
                  ('bo-', 'ro-', 'mo-'), ('μA(x)', 'μB(x)', 'μC(x)')),
        "A - B": (A, B, difference, 'Fuzzy Set A - B',
                  ('bo-', 'ro-', 'mo-'), ('μA(x)', 'μB(x)', 'μC(x)')),
        "A ⊖ B": (A, B, symmetric_difference, 'Fuzzy Set A ⊖ B',
                  ('bo-', 'ro-', 'mo-'), ('μA(x)', 'μB(x)', 'μC(x)')),
        "A ⋂_alg B": (A, B, algebraic_intersection, 'Fuzzy Set A ⋂_alg B',
                      ('bo-', 'ro-', 'mo-'), ('μA(x)', 'μB(x)', 'μC(x)')),
        "A ∪_alg B": (A, B, algebraic_union, 'Fuzzy Set A ∪_alg B',
                      ('bo-', 'ro-', 'mo-'), ('μA(x)', 'μB(x)', 'μC(x)')),
        "Concentration A": (concentration_A, None, None, 'Fuzzy Set A^2',
                            ('bo-',), ('μA^2(x)',)),
        "Concentration B": (None, concentration_B, None, 'Fuzzy Set B^2',
                            ('ro-',), ('μB^2(x)',)),
        "Dilation A": (dilution_A, None, None, 'Fuzzy Set A^0.5',
                       ('bo-',), ('μA^0.5(x)',)),
        "Dilation B": (None, dilution_B, None, 'Fuzzy Set B^0.5',
                       ('ro-',), ('μB^0.5(x)',)),
        "Boundary Intersection": (A, B, boundary_intersection, 'Fuzzy Set Boundary Intersection',
                                  ('bo-', 'ro-', 'mo-'), ('μA(x)', 'μB(x)', 'μC(x)')),
        "Boundary Union": (A, B, boundary_union, 'Fuzzy Set Boundary Union',
                           ('bo-', 'ro-', 'mo-'), ('μA(x)', 'μB(x)', 'μC(x)')),
    }


def characteristics(A, x, alpha):
    support = list(map(int, [x[i] for i in range(len(A)) if A[i] > 0]))
    core = list(map(int, [x[i] for i in range(len(A)) if A[i] == 1]))
    height = float(max(A))
    alpha_cut = list(map(int, [x[i] for i in range(len(A)) if A[i] >= alpha]))

    return support, core, height, alpha_cut


x = np.array([1, 2, 3, 4, 5, 6, 7, 8])
A = np.array([0.2, 0.8, 0.5, 1, 0, 0.9, 0.3, 0.4])
B = np.array([0.7, 0, 0, 0.6, 0.4, 1, 0, 0.4])

plot_fuzzy_sets(x, A, B, title="Fuzzy Sets A and B", colors=('bo-', 'ro-'), labels=('μA(x)', 'μB(x)'))

fuzzy_approximation_and_plot(x, A, B)

operations = fuzzy_operations(x, A, B)

for key, (A_val, B_val, C_val, title, colors, labels) in operations.items():
    plot_fuzzy_sets(x, A_val, B_val, C_val, title=title, colors=colors, labels=labels)

alpha_A = float(input("\nEnter alpha-level for A (0 < α ≤ 1): "))
alpha_B = float(input("Enter alpha-level for B (0 < α ≤ 1): "))

A_support, A_core, A_height, A_alpha_cut = characteristics(A, x, alpha_A)
B_support, B_core, B_height, B_alpha_cut = characteristics(B, x, alpha_B)

print("\n=== Characteristics of Fuzzy Set A ===")
print(f"Support: {A_support}")
print(f"Core: {A_core}")
print(f"Height: {A_height}")
print(f"Alpha-cut (α = {alpha_A}): {A_alpha_cut}")

print("\n=== Characteristics of Fuzzy Set B ===")
print(f"Support: {B_support}")
print(f"Core: {B_core}")
print(f"Height: {B_height}")
print(f"Alpha-cut (α = {alpha_B}): {B_alpha_cut}")
