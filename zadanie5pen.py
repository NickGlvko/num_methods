import numpy as np


def generate_matrices(n):

    Lambda_diag = np.random.uniform(-10, 10, n)
    while len(np.unique(Lambda_diag)) < n:
        Lambda_diag = np.random.uniform(-10, 10, n)
    Lambda = np.diag(Lambda_diag)


    C = np.random.uniform(-10, 10, (n, n))
    while abs(matrix_determinant(C)) < 1e-10:
        C = np.random.uniform(-10, 10, (n, n))


    C_inv = matrix_inverse(C)


    A = C_inv @ Lambda @ C
    return A, Lambda, C


def matrix_determinant(A):

    n = A.shape[0]
    A = A.copy()
    det = 1.0
    for i in range(n):
        max_element = abs(A[i, i])
        max_row = i
        for k in range(i + 1, n):
            if abs(A[k, i]) > max_element:
                max_element = abs(A[k, i])
                max_row = k
        if max_element < 1e-10:
            return 0.0
        if max_row != i:
            A[[i, max_row]] = A[[max_row, i]]
            det *= -1

        det *= A[i, i]

        for k in range(i + 1, n):
            factor = A[k, i] / A[i, i]
            A[k, i:] -= factor * A[i, i:]

    return det


def matrix_inverse(A):
    n = A.shape[0]
    augmented = np.hstack((A, np.eye(n)))

    for i in range(n):
        max_element = abs(augmented[i, i])
        max_row = i
        for k in range(i + 1, n):
            if abs(augmented[k, i]) > max_element:
                max_element = abs(augmented[k, i])
                max_row = k

        if max_element < 1e-10:
            raise ValueError("Matrix is singular")

        if max_row != i:
            augmented[[i, max_row]] = augmented[[max_row, i]]

        augmented[i] /= augmented[i, i]

        for k in range(n):
            if k != i:
                factor = augmented[k, i]
                augmented[k] -= factor * augmented[i]

    return augmented[:, n:]


def faddeev_leverrier(A):
    n = A.shape[0]
    coeffs = np.zeros(n + 1)
    coeffs[n] = 1.0
    B = np.eye(n)
    trace_sum = 0.0

    for k in range(1, n + 1):
        B = A @ B
        trace = np.sum(np.diag(B))
        coeffs[n - k] = -trace / k
        B = B + coeffs[n - k] * np.eye(n)

    return coeffs


def newton_method(poly, x0, max_iter=100, tol=1e-8):

    def poly_eval(coeffs, x):
        result = 0.0
        for i, coef in enumerate(coeffs):
            result += coef * x ** (len(coeffs) - 1 - i)
        return result

    def poly_deriv(coeffs):
        n = len(coeffs) - 1
        deriv = np.zeros(n)
        for i in range(n):
            deriv[i] = (n - i) * coeffs[i]
        return deriv

    x = x0
    for _ in range(max_iter):
        f = poly_eval(poly, x)
        f_prime = poly_eval(poly_deriv(poly), x)
        if abs(f_prime) < 1e-10:
            return None
        x_new = x - f / f_prime
        if abs(x_new - x) < tol:
            return x_new
        x = x_new
    return None


def find_eigenvalues(A, Lambda):
    n = A.shape[0]
    coeffs = faddeev_leverrier(A)

    initial_guesses = np.diag(Lambda)

    eigenvalues = []
    for x0 in initial_guesses:
        root = newton_method(coeffs, x0)
        if root is not None and not any(abs(root - ev) < 1e-6 for ev in eigenvalues):
            eigenvalues.append(root)

    if len(eigenvalues) < n:
        for x0 in np.linspace(-10, 10, 10):
            root = newton_method(coeffs, x0)
            if root is not None and not any(abs(root - ev) < 1e-6 for ev in eigenvalues):
                eigenvalues.append(root)
                if len(eigenvalues) == n:
                    break

    return np.array(eigenvalues)


n = int(input("Введите размерность n: "))


A, Lambda, C = generate_matrices(n)


print("Матрица A:")
print(A)
print("Истинные собственные числа (из Lambda):", np.diag(Lambda))


eigenvalues = find_eigenvalues(A, Lambda)
print("Найденные собственные числа:")
print(eigenvalues)