import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def f(x):
    return x ** 2 + 1 - np.arccos(x)


def linear_spline(x, nodes, values):
    for i in range(len(nodes) - 1):
        if nodes[i] <= x <= nodes[i + 1]:
            return values[i] + (values[i + 1] - values[i]) / (nodes[i + 1] - nodes[i]) * (x - nodes[i])
    raise ValueError(f"x={x} вне интервала [{nodes[0]}, {nodes[-1]}]")


def quadratic_spline(x, nodes, values):
    n = len(nodes)
    h = np.diff(nodes)


    A = np.zeros((3 * (n - 1), 3 * (n - 1)))
    b = np.zeros(3 * (n - 1))


    for i in range(n - 1):
        A[2 * i, 3 * i] = h[i] ** 2
        A[2 * i, 3 * i + 1] = h[i]
        A[2 * i, 3 * i + 2] = 1
        b[2 * i] = values[i + 1]


        A[2 * i + 1, 3 * i + 2] = 1  # c_i
        b[2 * i + 1] = values[i]


    for i in range(n - 2):
        A[2 * (n - 1) + i, 3 * i] = 2 * h[i]
        A[2 * (n - 1) + i, 3 * i + 1] = 1
        A[2 * (n - 1) + i, 3 * (i + 1)] = -2 * h[i + 1]
        A[2 * (n - 1) + i, 3 * (i + 1) + 1] = -1


    A[-1, 1] = 1


    coeffs = np.linalg.solve(A, b)


    for i in range(n - 1):
        if nodes[i] <= x <= nodes[i + 1]:
            a = coeffs[3 * i]
            b = coeffs[3 * i + 1]
            c = coeffs[3 * i + 2]
            return a * (x - nodes[i]) ** 2 + b * (x - nodes[i]) + c
    raise ValueError(f"x={x} вне интервала")


def cubic_spline(x, nodes, values):
    n = len(nodes)
    h = np.diff(nodes)


    A = np.zeros((n, n))
    b = np.zeros(n)


    A[0, 0] = 1  # M_0 = 0
    A[-1, -1] = 1  # M_n = 0


    for i in range(1, n - 1):
        A[i, i - 1] = h[i - 1]
        A[i, i] = 2 * (h[i - 1] + h[i])
        A[i, i + 1] = h[i]
        b[i] = 6 * ((values[i + 1] - values[i]) / h[i] - (values[i] - values[i - 1]) / h[i - 1])

    M = np.linalg.solve(A, b)


    for i in range(n - 1):
        if nodes[i] <= x <= nodes[i + 1]:
            a = (M[i + 1] - M[i]) / (6 * h[i])
            b = M[i] / 2
            c = (values[i + 1] - values[i]) / h[i] - (2 * h[i] * M[i] + h[i] * M[i + 1]) / 6
            d = values[i]
            return a * (x - nodes[i]) ** 3 + b * (x - nodes[i]) ** 2 + c * (x - nodes[i]) + d
    raise ValueError(f"x={x} вне интервала")


def divided_diff(nodes, values):
    n = len(nodes)
    coef = np.zeros((n, n))
    coef[:, 0] = values
    for j in range(1, n):
        for i in range(n - j):
            coef[i][j] = (coef[i + 1][j - 1] - coef[i][j - 1]) / (nodes[i + j] - nodes[i])
    return coef[0]


def newton_poly(x, nodes, coef):
    poly = coef[0]
    for i in range(1, len(nodes)):
        term = coef[i]
        for j in range(i):
            term *= (x - nodes[j])
        poly += term
    return poly



a, b = -0.8, 0.8
n_values = [3, 5, 10, 20]
k = 1000


spline_table = []

for n in n_values:
    x_nodes = np.linspace(a, b, n)
    y_nodes = f(x_nodes)
    x_test = np.linspace(a, b, k)


    y_linear = np.array([linear_spline(x, x_nodes, y_nodes) for x in x_test])
    y_quad = np.array([quadratic_spline(x, x_nodes, y_nodes) for x in x_test])
    y_cubic = np.array([cubic_spline(x, x_nodes, y_nodes) for x in x_test])


    coef = divided_diff(x_nodes, y_nodes)
    y_newton = np.array([newton_poly(x, x_nodes, coef) for x in x_test])


    dev_linear = np.max(np.abs(f(x_test) - y_linear))
    dev_quad = np.max(np.abs(f(x_test) - y_quad))
    dev_cubic = np.max(np.abs(f(x_test) - y_cubic))
    dev_newton = np.max(np.abs(f(x_test) - y_newton))

    spline_table.append({
        'n': n,
        'k': k,
        'RS^n_{1,0}': f"{dev_linear:.6f}",
        'RS^n_{2,1}': f"{dev_quad:.6f}",
        'RS^n_{3,2}': f"{dev_cubic:.6f}",
        'RN^n (Ньютон)': f"{dev_newton:.6f}"
    })

print("Таблица 3. Интерполяционные сплайны и полином Ньютона")
print(pd.DataFrame(spline_table).to_markdown(index=False))


plt.figure(figsize=(15, 10))
for idx, n in enumerate([5, 10]):
    x_nodes = np.linspace(a, b, n)
    y_nodes = f(x_nodes)
    x_plot = np.linspace(a, b, 1000)


    y_linear = [linear_spline(x, x_nodes, y_nodes) for x in x_plot]
    y_quad = [quadratic_spline(x, x_nodes, y_nodes) for x in x_plot]
    y_cubic = [cubic_spline(x, x_nodes, y_nodes) for x in x_plot]


    coef = divided_diff(x_nodes, y_nodes)
    y_newton = [newton_poly(x, x_nodes, coef) for x in x_plot]


    plt.subplot(2, 2, idx * 2 + 1)
    plt.plot(x_plot, f(x_plot), 'k-', label='f(x)')
    plt.plot(x_plot, y_linear, 'b--', label='S_{1,0}(x)')
    plt.plot(x_plot, y_quad, 'g:', label='S_{2,1}(x)')
    plt.plot(x_plot, y_cubic, 'r-.', label='S_{3,2}(x)')
    plt.scatter(x_nodes, y_nodes, c='red', marker='o')
    plt.title(f'Сплайны (n={n})')
    plt.legend()
    plt.grid()


    plt.subplot(2, 2, idx * 2 + 2)
    plt.plot(x_plot, np.abs(f(x_plot) - y_cubic), 'r-', label='Ошибка S_{3,2}')
    plt.plot(x_plot, np.abs(f(x_plot) - y_newton), 'm--', label='Ошибка Ньютона')
    plt.title(f'Ошибки интерполяции (n={n})')
    plt.legend()
    plt.grid()

plt.tight_layout()
plt.show()