import numpy as np
from math import factorial
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar


def f(x):
    return x ** 2 - np.sin(10 * x)


def f_deriv(x, order):
    if order == 0:
        return x ** 2 - np.sin(10 * x)
    elif order == 1:
        return 2 * x - 10 * np.cos(10 * x)
    elif order == 2:
        return 2 + 100 * np.sin(10 * x)
    elif order >= 3:
        if order % 2 == 0:  # Even order
            return (-1) ** (order // 2) * (10 ** order) * np.sin(10 * x)
        else:  # Odd order
            return (-1) ** ((order + 1) // 2) * (10 ** order) * np.cos(10 * x)
    else:
        raise ValueError("Недопустимый порядок производной")



def solve_poly_and_error(n, nodes_type='equal'):
    a, b = -1, 1
    if nodes_type == 'equal':
        x_nodes = np.linspace(a, b, n + 1)
    else:
        x_nodes = 0.5 * (a + b) + 0.5 * (b - a) * np.cos((2 * np.arange(n + 1) + 1) * np.pi / (2 * (n + 1)))

    y_nodes = f(x_nodes)
    V = np.vander(x_nodes, increasing=False)
    coeffs = np.linalg.solve(V, y_nodes)

    x_test = np.linspace(a, b, 10000)
    M = np.max(np.abs([f_deriv(x, n + 1) for x in x_test]))

    if nodes_type == 'equal':
        def product(x):
            return np.prod(np.abs(x - x_nodes))

        max_prod = np.max([product(x) for x in x_test])
        error_bound = (M / factorial(n + 1)) * max_prod
    else:
        error_bound = (M * (b - a) ** (n + 1)) / (2 ** (2 * n + 1) * factorial(n + 1))

    return coeffs, x_nodes, error_bound



def newton_poly(x, nodes, values):
    n = len(nodes)
    coeffs = np.copy(values).astype(float)
    for i in range(1, n):
        for j in range(n - 1, i - 1, -1):
            coeffs[j] = (coeffs[j] - coeffs[j - 1]) / (nodes[j] - nodes[j - i])

    result = coeffs[n - 1]
    for i in range(n - 2, -1, -1):
        result = result * (x - nodes[i]) + coeffs[i]
    return result



def quadratic_spline_20(x, nodes, values):
    n = len(nodes)
    for i in range(n - 1):
        if nodes[i] <= x <= nodes[i + 1]:
            A = np.array([
                [nodes[i] ** 2, nodes[i], 1],
                [nodes[i + 1] ** 2, nodes[i + 1], 1],
                [2 * nodes[i], 1, 0]  # Условие S'(x_i) = 0
            ])
            b = np.array([values[i], values[i + 1], 0])
            coeffs = np.linalg.solve(A, b)
            return coeffs[0] * x ** 2 + coeffs[1] * x + coeffs[2]
    raise ValueError("x вне интервала")



def compare_methods(n_values, m=1000):
    results = []
    x_test = np.linspace(-1, 1, m)

    for n in n_values:
        x_eq = np.linspace(-1, 1, n + 1)
        y_eq = f(x_eq)
        x_opt = 0.5 * (-1 + 1) + 0.5 * (1 + 1) * np.cos((2 * np.arange(n + 1) + 1) * np.pi / (2 * (n + 1)))
        y_opt = f(x_opt)

        coeffs, _, _ = solve_poly_and_error(n, 'equal')
        poly_eq = lambda x: np.polyval(coeffs[::-1], x)
        R_poly = np.max(np.abs(f(x_test) - poly_eq(x_test)))

        R_newton = np.max(np.abs(f(x_test) - [newton_poly(x, x_opt, y_opt) for x in x_test]))

        R_spline = np.max(np.abs(f(x_test) - [quadratic_spline_20(x, x_eq, y_eq) for x in x_test]))

        results.append({
            'n': n,
            'm': m,
            'R_poly': f"{R_poly:.6f}",
            'R_newton': f"{R_newton:.6f}",
            'R_spline': f"{R_spline:.6f}"
        })

    return results



n_values = [3, 5, 7, 10]
x_test = np.linspace(-1, 1, 1000)


results = compare_methods(n_values)


import pandas as pd

df = pd.DataFrame(results)
print("\nТаблица сравнения методов:")
print(df.to_markdown(index=False))



def plot_spline_20(n):
    x_nodes = np.linspace(-1, 1, n + 1)
    y_nodes = f(x_nodes)
    x_plot = np.linspace(-1, 1, 1000)
    y_spline = [quadratic_spline_20(x, x_nodes, y_nodes) for x in x_plot]

    plt.figure(figsize=(10, 5))
    plt.plot(x_plot, f(x_plot), 'k-', label='f(x)')
    plt.plot(x_plot, y_spline, 'r--', label=f'S_{{2,0}}, n={n}')
    plt.scatter(x_nodes, y_nodes, c='blue', label='Узлы')
    plt.title(f'Квадратичный сплайн S_{{2,0}} (n={n})')
    plt.legend()
    plt.grid()
    plt.show()


plot_spline_20(5)
plot_spline_20(10)