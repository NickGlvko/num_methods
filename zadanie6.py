import numpy as np
import matplotlib.pyplot as plt
from scipy.special import eval_legendre
import random


def generate_noisy_data(func, x_values, noise_level=0.1, measurements_per_point=3):
    y_values = []
    for x in x_values:
        true_value = func(x)

        for _ in range(measurements_per_point):
            noise = random.uniform(-noise_level, noise_level)
            y_values.append(true_value + noise)
    x_repeated = np.repeat(x_values, measurements_per_point)
    return x_repeated, np.array(y_values)


def least_squares_normal_equations(x, y, degree):
    X = np.vander(x, degree + 1, increasing=True)
    coefficients = np.linalg.solve(X.T @ X, X.T @ y)
    return coefficients


def evaluate_polynomial(coefficients, x):
    return sum(coef * x ** i for i, coef in enumerate(coefficients))


def least_squares_orthogonal_polynomials(x, y, degree):
    n = len(x)
    q = [np.ones_like(x)]

    if degree >= 1:
        alpha1 = np.mean(x)
        q.append(x - alpha1)


    for j in range(1, degree):
        numerator_alpha = np.sum(x * q[j] ** 2)
        denominator_alpha = np.sum(q[j] ** 2)
        alpha = numerator_alpha / denominator_alpha

        numerator_beta = np.sum(x * q[j] * q[j - 1])
        denominator_beta = np.sum(q[j - 1] ** 2)
        beta = numerator_beta / denominator_beta


        q_next = (x - alpha) * q[j] - beta * q[j - 1]
        q.append(q_next)


    coefficients = []
    for j in range(degree + 1):
        a_j = np.sum(y * q[j]) / np.sum(q[j] ** 2)
        coefficients.append(a_j)


    def poly_func(x_eval):
        result = 0
        q_eval = [1.0]
        if degree >= 1:
            q_eval.append(x_eval - alpha1)

        for j in range(1, degree):
            q_eval.append((x_eval - alpha) * q_eval[j] - beta * q_eval[j - 1])

        for j in range(degree + 1):
            result += coefficients[j] * q_eval[j]

        return result

    return poly_func


def calculate_sse(y_true, y_pred):
    return np.sum((y_true - y_pred)**2)




def target_function(x):
    return x**2 * np.cos(x)


interval = [-1, 1]
num_points = 50
noise_level = 0.1
measurements_per_point = 3
max_degree = 6


x_unique = np.linspace(interval[0], interval[1], num_points)
x, y = generate_noisy_data(target_function, x_unique, noise_level, measurements_per_point)


results_table = []


plt.figure(figsize=(15, 10))

for degree in range(1, max_degree + 1):
    coeff_normal = least_squares_normal_equations(x, y, degree)
    y_pred_normal = np.array([evaluate_polynomial(coeff_normal, xi) for xi in x_unique])
    sse_normal = calculate_sse(target_function(x_unique), y_pred_normal)


    poly_ortho = least_squares_orthogonal_polynomials(x, y, degree)
    y_pred_ortho = np.array([poly_ortho(xi) for xi in x_unique])
    sse_ortho = calculate_sse(target_function(x_unique), y_pred_ortho)


    results_table.append([degree, sse_normal, sse_ortho])


    plt.subplot(2, 3, degree)
    plt.scatter(x, y, s=5, alpha=0.3, label='Экспериментальные данные')
    plt.plot(x_unique, target_function(x_unique), 'k-', label='Истинная функция')
    plt.plot(x_unique, y_pred_normal, 'r-', label=f'МНК (норм. ур.), n={degree}')
    plt.plot(x_unique, y_pred_ortho, 'b--', label=f'МНК (ортог.), n={degree}')
    plt.title(f'Аппроксимация полиномом {degree} степени')
    plt.legend()

plt.tight_layout()
plt.show()


print("\nРезультаты аппроксимации:")
print("| Степень полинома | Сумма квадратов ошибок (норм. ур.) | Сумма квадратов ошибок (ортог.) |")
print("|------------------|------------------------------------|----------------------------------|")
for row in results_table:
    print(f"| {row[0]:16} | {row[1]:34.6f} | {row[2]:32.6f} |")