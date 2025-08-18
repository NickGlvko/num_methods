import numpy as np
from tabulate import tabulate

def lu_decomposition(A):
    n = len(A)
    L = np.eye(n)
    U = A.copy()
    P = np.eye(n)

    for i in range(n):
        pivot = np.argmax(np.abs(U[i:, i])) + i
        if pivot != i:
            U[[i, pivot]] = U[[pivot, i]]
            P[[i, pivot]] = P[[pivot, i]]
            if i > 0:
                L[[i, pivot], :i] = L[[pivot, i], :i]

        for j in range(i + 1, n):
            L[j, i] = U[j, i] / U[i, i]
            U[j, i:] -= L[j, i] * U[i, i:]
    return L, U, P

def solve_lu(L, U, P, b):
    Pb = np.dot(P, b)

    n = len(b)
    y = np.zeros(n)
    for i in range(n):
        y[i] = Pb[i] - np.dot(L[i, :i], y[:i])

    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = (y[i] - np.dot(U[i, i + 1:], x[i + 1:])) / U[i, i]
    return x


def qr_decomposition(A):
    n = len(A)
    Q = np.eye(n)
    R = A.copy()

    for i in range(n - 1):
        pivot = np.argmax(np.abs(R[i:, i])) + i
        if pivot != i:
            R[[i, pivot]] = R[[pivot, i]]
            Q[[i, pivot]] = Q[[pivot, i]]

        x = R[i:, i]
        e = np.zeros_like(x)
        e[0] = np.sign(x[0]) * np.linalg.norm(x)
        u = x - e
        u /= np.linalg.norm(u)
        H = np.eye(n)
        H[i:, i:] -= 2.0 * np.outer(u, u)
        R = H @ R
        Q = H @ Q
    return Q.T, R

def solve_qr(Q, R, b):
    y = Q.T @ b

    n = len(y)
    x = np.zeros(n)

    for i in range(n - 1, -1, -1):
        x[i] = (y[i] - np.dot(R[i, i + 1:], x[i + 1:])) / R[i, i]
    return x


def is_diagonally_dominant(A):
    n = len(A)
    for i in range(n):
        row_sum = sum(abs(A[i, j]) for j in range(n) if j != i)
        if abs(A[i, i]) <= row_sum:
            return False
    return True

def is_positive_definite(A):
    A_sym = (A + A.T) / 2
    eigenvalues = np.linalg.eigvals(A_sym)
    return np.all(eigenvalues > 0)


def simple_iteration(A, b, tol=1e-2, max_iter=1000000):
    n = len(A)
    x = np.zeros(n)

    if not is_positive_definite(A):
        A = np.dot(A.T, A)
        b = np.dot(A.T, b)

    eigenvalues = np.linalg.eigvals(A)
    lambda_min = np.min(np.abs(eigenvalues[eigenvalues > 0]))
    lambda_max = np.max(np.abs(eigenvalues))
    mu = 2 / (lambda_min + lambda_max)

    B = np.eye(n) - mu * A
    c = mu * b

    iter_count = 0

    for _ in range(max_iter):
        x_new = np.dot(B, x) + c
        iter_count += 1

        if np.linalg.norm(x_new - x, ord=np.inf) < tol:
            return x_new, iter_count

        x = x_new

    return x, iter_count


def seidel_method(A, b, tol=1e-2, max_iter=1000):
    if np.any(np.diag(A) == 0):
        return "zero diagonal elements", 0

    n = len(A)
    C = np.zeros_like(A)
    d = np.zeros_like(b)
    for i in range(n):
        d[i] = b[i] / A[i, i]
        for j in range(n):
            if j != i:
                C[i, j] = -A[i, j] / A[i, i]
    x = d.copy()
    iter_count = 0

    for _ in range(max_iter):
        x_new = x.copy()
        for i in range(n):
            sum1 = sum(C[i, j] * x_new[j] for j in range(i))
            sum2 = sum(C[i, j] * x[j] for j in range(i + 1, n))
            x_new[i] = sum1 + sum2 + d[i]

        residual = np.linalg.norm(A @ x_new - b)
        iter_count += 1

        if residual < tol:
            return x_new, iter_count

        x = x_new

    return x, iter_count


def get_test5_matrix(N, n, epsilon):
    A1 = np.eye(n)
    for i in range(n):
        for j in range(i + 1, n):
            A1[i, j] = -1

    A2 = np.ones((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            A2[i, j] = -1

    A = A1 + epsilon * N * A2


    b = np.full(n, -1.0)
    b[-1] = 1.0
    return A, b


N = 4
n_values = [3, 5, 10]
epsilon_values = [1e-3, 1e-4, 1e-5]
tol_values = [1e-2, 1e-3, 1e-4]


table_data_test5 = []


for n in n_values:
    for epsilon in epsilon_values:
        for tol in tol_values:

            A, b = get_test5_matrix(N, n, epsilon)
            try:
                exact_solution = np.linalg.solve(A, b)
            except np.linalg.LinAlgError:
                exact_solution = "singular matrix"


            try:
                L, U, P = lu_decomposition(A.copy())
                x_lu = solve_lu(L, U, P, b)
                error_lu = (
                    np.linalg.norm(x_lu - exact_solution, ord=np.inf)
                    if isinstance(exact_solution, np.ndarray)
                    else "-"
                )
            except:
                x_lu, error_lu = "error", "-"

            try:
                Q, R = qr_decomposition(A.copy())
                x_qr = solve_qr(Q, R, b)
                error_qr = (
                    np.linalg.norm(x_qr - exact_solution, ord=np.inf)
                    if isinstance(exact_solution, np.ndarray)
                    else "-"
                )
            except:
                x_qr, error_qr = "error", "-"


            try:
                x_simple, k_simple = simple_iteration(A, b, tol)
                error_simple = (
                    np.linalg.norm(x_simple - exact_solution, ord=np.inf)
                    if isinstance(exact_solution, np.ndarray)
                    else "-"
                )
            except:
                x_simple, error_simple, k_simple = "error", "-", 0

            if not (is_diagonally_dominant(A) or is_positive_definite(A)):
                x_seidel, k_seidel = "inappropriate matrix", 0
                error_seidel = "-"
            else:
                try:
                    x_seidel, k_seidel = seidel_method(A, b, tol)
                    error_seidel = (
                        np.linalg.norm(x_seidel - exact_solution, ord=np.inf)
                        if isinstance(exact_solution, np.ndarray)
                        else "-"
                    )
                except:
                    x_seidel, error_seidel, k_seidel = "error", "-", 0


            table_data_test5.append([
                5, n, epsilon, exact_solution, tol,
                x_simple, error_simple, k_simple,
                x_seidel, error_seidel, k_seidel,
                x_lu, error_lu, x_qr, error_qr])

headers_test5 = ["Test #", "n", "eps", "Exact Solution", "e",
                 "Simple Iteration (x)", "Δ (Simple)", "k (Simple)",
                 "Seidel (x)", "Δ (Seidel)", "k (Seidel)",
                 "LU (x)", "Δ (LU)", "QR (x)", "Δ (QR)"]
print(tabulate(table_data_test5, headers=headers_test5, tablefmt="grid"))