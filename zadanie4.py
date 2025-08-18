import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def f(x):
    return x ** 2 + 1 - np.arccos(x)


a, b = -0.8, 0.8
n_values = [3, 5, 10, 20, 50]
m = 1000


def max_deviation(f, poly_func, nodes, values, m_points):
    x_test = np.linspace(min(nodes), max(nodes), m_points)
    y_true = f(x_test)
    y_poly = np.array([poly_func(x, nodes, values) for x in x_test])
    return np.max(np.abs(y_true - y_poly))


def lagrange_poly(x, nodes, values):
    n = len(nodes)
    poly = 0.0
    for i in range(n):
        term = values[i]
        for j in range(n):
            if j != i:
                term *= (x - nodes[j]) / (nodes[i] - nodes[j])
        poly += term
    return poly


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


lagrange_data = []
newton_data = []

for n in n_values:
    x_eq = np.linspace(a, b, n)
    x_cheb = 0.5 * (a + b) + 0.5 * (b - a) * np.cos((2 * np.arange(n) + 1) * np.pi / (2 * n))


    dev_L_eq = max_deviation(f, lagrange_poly, x_eq, f(x_eq), m)
    dev_L_cheb = max_deviation(f, lagrange_poly, x_cheb, f(x_cheb), m)
    lagrange_data.append({
        'n': n,
        'm': m,
        'RLn (равные узлы)': f"{dev_L_eq:.6f}",
        'RLoptn (Чебышёв)': f"{dev_L_cheb:.6f}"
    })


    coef_eq = divided_diff(x_eq, f(x_eq))
    coef_cheb = divided_diff(x_cheb, f(x_cheb))
    dev_N_eq = max_deviation(f, lambda x, nodes, _: newton_poly(x, nodes, coef_eq), x_eq, f(x_eq), m)
    dev_N_cheb = max_deviation(f, lambda x, nodes, _: newton_poly(x, nodes, coef_cheb), x_cheb, f(x_cheb), m)
    newton_data.append({
        'n': n,
        'm': m,
        'RNn (равные узлы)': f"{dev_N_eq:.6f}",
        'RNoptn (Чебышёв)': f"{dev_N_cheb:.6f}"
    })


print("Таблица 1. Полином Лагранжа")
print(pd.DataFrame(lagrange_data).to_markdown(index=False))

print("\nТаблица 2. Полином Ньютона")
print(pd.DataFrame(newton_data).to_markdown(index=False))


plt.figure(figsize=(15, 10))
x_plot = np.linspace(a, b, 1000)


for i, n in enumerate([3, 10]):
    x_eq = np.linspace(a, b, n)
    y_eq = f(x_eq)


    x_cheb = 0.5 * (a + b) + 0.5 * (b - a) * np.cos((2 * np.arange(n) + 1) * np.pi / (2 * n))
    y_cheb = f(x_cheb)


    y_L_eq = [lagrange_poly(x, x_eq, y_eq) for x in x_plot]
    y_L_cheb = [lagrange_poly(x, x_cheb, y_cheb) for x in x_plot]


    coef_eq = divided_diff(x_eq, y_eq)
    coef_cheb = divided_diff(x_cheb, y_cheb)
    y_N_eq = [newton_poly(x, x_eq, coef_eq) for x in x_plot]
    y_N_cheb = [newton_poly(x, x_cheb, coef_cheb) for x in x_plot]


    plt.subplot(2, 2, i + 1)
    plt.plot(x_plot, f(x_plot), 'k-', label='f(x)')
    plt.plot(x_plot, y_L_eq, 'b--', label=f'L_{n}(x) (равные)')
    plt.plot(x_plot, y_L_cheb, 'g--', label=f'L_{n}(x) (Чебышёв)')
    plt.scatter(x_eq, y_eq, c='b', marker='o')
    plt.scatter(x_cheb, y_cheb, c='g', marker='x')
    plt.title(f'Полиномы Лагранжа (n={n})')
    plt.legend()
    plt.grid()

    plt.subplot(2, 2, i + 3)
    plt.plot(x_plot, f(x_plot), 'k-', label='f(x)')
    plt.plot(x_plot, y_N_eq, 'b:', label=f'N_{n}(x) (равные)')
    plt.plot(x_plot, y_N_cheb, 'g:', label=f'N_{n}(x) (Чебышёв)')
    plt.scatter(x_eq, y_eq, c='b', marker='o')
    plt.scatter(x_cheb, y_cheb, c='g', marker='x')
    plt.title(f'Полиномы Ньютона (n={n})')
    plt.legend()
    plt.grid()

plt.tight_layout()
plt.show()