import numpy as np


def f(x):
    return x ** 2 + 1 - np.arccos(x)

def f_prime(x):
    return 2 * x + 1 / np.sqrt(1 - x ** 2)


def localize_root(f, a, b, N):
    h = (b - a) / N
    x_values = np.linspace(a, b, N + 1)
    f_values = [f(x) for x in x_values]

    print(f"\nЛокализация корня на интервале [{a}, {b}] с N = {N}:")
    for i in range(N):
        print(f"x_{i} = {x_values[i]:.6f}, f(x_{i}) = {f_values[i]:.6f}")


    for i in range(N):
        if f_values[i] * f_values[i + 1] < 0:
            a0 = x_values[i]
            b0 = x_values[i + 1]
            print(f"\nСмена знака обнаружена: f({a0:.6f}) = {f_values[i]:.6f}, f({b0:.6f}) = {f_values[i + 1]:.6f}")
            return a0, b0

    return localize_root(f, a, b, 2 * N)


def newton_with_bisection(f, f_prime, a0, b0, x0, tol=1e-4, max_iter=100):
    a, b = a0, b0
    x = x0
    k = 0

    print(f"\nНачальный интервал: [{a:.6f}, {b:.6f}]")
    print(f"Начальное приближение: x0 = {x:.6f}")

    while k < max_iter:
        fx = f(x)
        fpx = f_prime(x)
        x_new = x - fx / fpx

        print(f"\nИтерация {k + 1}:")
        print(f"x_{k} = {x:.6f}, f(x_{k}) = {fx:.6f}")
        print(f"x_{k + 1} (Ньютон) = {x_new:.6f}")


        if x_new < a or x_new > b:
            print(f"x_{k + 1} = {x_new:.6f} вне интервала [{a:.6f}, {b:.6f}]")
            x_new = (a + b) / 2
            print(f"Применяем метод половинного деления: x_{k + 1} = {x_new:.6f}")


        fx_new = f(x_new)
        print(f"f(x_{k + 1}) = {fx_new:.6f}")
        if fx_new < 0:
            a = x_new
        else:
            b = x_new
        print(f"Новый интервал: [{a:.6f}, {b:.6f}]")


        if abs(x_new - x) < tol:
            print(f"\nРешение найдено: x = {x_new:.6f}")
            print(f"Количество итераций: {k + 1}")
            return x_new, k + 1

        x = x_new
        k += 1

    print("Достигнуто максимальное количество итераций.")
    return x, k


a, b = -1, 1
N = 10


a0, b0 = localize_root(f, a, b, N)


x0 = a0
tol = 1e-4


root, iterations = newton_with_bisection(f, f_prime, a0, b0, x0, tol)