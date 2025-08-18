import numpy as np
import matplotlib.pyplot as plt


def F(x, y):
    return np.array([
        np.cos(y) + x - 1.5,
        2 * y - np.sin(x - 0.5) - 1
    ])

def jacobian(x, y):
    return np.array([
        [1, -np.sin(y)],
        [-np.cos(x - 0.5), 2]
    ])


def y1(x):
    arg = 1.5 - x
    arg = np.clip(arg, -1, 1)
    return np.arccos(arg)


def y2(x):
    return (np.sin(x - 0.5) + 1) / 2


def graphical_method():
    x = np.linspace(0.5, 2.5, 1000)
    y1_values = y1(x)
    y2_values = y2(x)

    plt.figure(figsize=(8, 6))
    plt.plot(x, y1_values, label='y = arccos(1.5 - x)', color='blue')
    plt.plot(x, y2_values, label='y = (sin(x - 0.5) + 1) / 2', color='red')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Графический метод: поиск начального приближения')
    plt.grid(True)
    plt.legend()
    plt.show()


    diff = np.abs(y1_values - y2_values)
    idx = np.argmin(diff)
    x_intersect = x[idx]
    y_intersect = y1_values[idx]
    print(f"Примерная точка пересечения (графический метод): (x, y) = ({x_intersect:.3f}, {y_intersect:.3f})")
    return x_intersect, y_intersect


def Phi(lambda_val, x, y):
    return np.array([
        lambda_val * (np.cos(y) + x - 1.5),
        lambda_val * (2 * y - np.sin(x - 0.5) - 1)
    ])


def newton_with_lambda(Phi, jacobian, x0, y0, N=10, tol=1e-4, max_iter=100):
    x, y = x0, y0
    lambda_values = np.linspace(0, 1, N + 1)

    for i, lambda_val in enumerate(lambda_values[1:], 1):
        print(f"\nРешение для lambda = {lambda_val:.2f}:")
        print(f"Начальное приближение: (x, y) = ({x:.6f}, {y:.6f})")

        k = 0
        while k < max_iter:
            phi_val = Phi(lambda_val, x, y)
            J = lambda_val * jacobian(x, y)
            delta = np.linalg.solve(J, -phi_val)
            x_new = x + delta[0]
            y_new = y + delta[1]

            if np.linalg.norm([x_new - x, y_new - y]) < tol:
                print(f"Решение для lambda = {lambda_val:.2f}: (x, y) = ({x_new:.6f}, {y_new:.6f})")
                x, y = x_new, y_new
                break

            x, y = x_new, y_new
            k += 1

    return x, y


def newton_system(F, jacobian, x0, y0, tol=1e-4, max_iter=100):
    x, y = x0, y0
    k = 0

    print(f"\nНачальное приближение: (x0, y0) = ({x0:.6f}, {y0:.6f})")

    while k < max_iter:
        f_val = F(x, y)
        J = jacobian(x, y)
        delta = np.linalg.solve(J, -f_val)
        x_new = x + delta[0]
        y_new = y + delta[1]

        print(f"\nИтерация {k + 1}:")
        print(f"(x_{k}, y_{k}) = ({x:.6f}, {y:.6f})")
        print(f"F(x_{k}, y_{k}) = {f_val}")
        print(f"(x_{k + 1}, y_{k + 1}) = ({x_new:.6f}, {y_new:.6f})")

        if np.linalg.norm([x_new - x, y_new - y]) < tol:
            print(f"\nРешение найдено: (x, y) = ({x_new:.6f}, {y_new:.6f})")
            print(f"Количество итераций: {k + 1}")
            return x_new, y_new, k + 1

        x, y = x_new, y_new
        k += 1

    print("Достигнуто максимальное количество итераций.")
    return x, y, k


print("=== Графический метод ===")
x0_graph, y0_graph = graphical_method()


print("\n=== Метод Ньютона с начальным приближением от графического метода ===")
x_graph, y_graph, iterations_graph = newton_system(F, jacobian, x0_graph, y0_graph, tol=1e-4)


print("\nВспомогательная система Phi(lambda, x, y) ===")
x0_lambda, y0_lambda = 0, 0  # Начальное приближение для lambda = 0
x_final_lambda, y_final_lambda= newton_with_lambda(Phi, jacobian, x0_lambda, y0_lambda, N=10,
                                                                         tol=1e-4)
