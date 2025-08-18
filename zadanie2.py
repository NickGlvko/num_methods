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


def is_positive_definite(A):
    eigenvalues = np.linalg.eigvals(A)
    return np.all(eigenvalues > 0)

def is_diagonally_dominant(A):
    n = len(A)
    for i in range(n):
        row_sum = sum(abs(A[i, j]) for j in range(n) if j != i)
        if abs(A[i, i]) < row_sum:
            return False
    return True


def simple_iteration(A, b, tol=1e-2, max_iter=10000):
    n = len(A)
    x = np.zeros(n)

    if not is_positive_definite(A):
        A = np.dot(A.T, A)
        b = np.dot(A.T, b)

    eigenvalues = np.linalg.eigvals(A)
    lambda_min = np.min(np.abs(eigenvalues))
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


def seidel_method(A, b, tol=1e-2, max_iter=100):
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


def get_test_matrices(N, test_num):
    if test_num == 0:
        A = np.array([[0, 2, 3],
                      [1, 2, 4],
                      [4, 5, 6]], dtype=float)
        b = np.array([13, 17, 32], dtype=float)
    elif test_num == 1:
        A = np.array([[N + 2, 1, 1],
                      [1, N + 4, 1],
                      [1, 1, N + 6]], dtype=float)
        b = np.array([N + 4, N + 6, N + 8], dtype=float)
    elif test_num == 2:
        A = np.array([[-(N + 2), 1, 1],
                      [1, -(N + 4), 1],
                      [1, 1, -(N + 6)]], dtype=float)
        b = np.array([-(N + 4), -(N + 6), -(N + 8)], dtype=float)
    elif test_num == 3:
        A = np.array([[-(N + 2), N + 3, N + 4],
                      [N + 5, -(N + 4), N + 1],
                      [N + 4, N + 5, -(N + 6)]], dtype=float)
        b = np.array([N + 4, N + 6, N + 8], dtype=float)
    elif test_num == 4:
        A = np.array([[N + 2, N + 1, N + 1],
                      [N + 1, N + 4, N + 1],
                      [N + 1, N + 1, N + 6]], dtype=float)
        b = np.array([N + 4, N + 6, N + 8], dtype=float)
    return A, b

N = 4
tests = range(0, 5)
tol_values=[1e-2, 1e-3, 1e-4]

table_data = []

for test_num in tests:
    for tol in tol_values:
        A, b = get_test_matrices(N, test_num)

        L, U, P = lu_decomposition(A.copy())
        x_lu = solve_lu(L, U, P, b)
        Q, R = qr_decomposition(A.copy())
        x_qr = solve_qr(Q, R, b)
        x_simple, k_simple = simple_iteration(A, b, tol)

        if not (is_diagonally_dominant(A) or is_positive_definite(A)):
            x_seidel, k_seidel = "inappropriate matrix", 0
        else:
            x_seidel, k_seidel = seidel_method(A, b, tol)

        exact_solution = np.linalg.solve(A, b)
        error_simple = (
            np.linalg.norm(x_simple - exact_solution, ord=np.inf)
            if isinstance(x_simple, np.ndarray)
            else "-"
        )
        error_seidel = (
            np.linalg.norm(x_seidel - exact_solution, ord=np.inf)
            if isinstance(x_seidel, np.ndarray)
            else "-"
        )
        error_lu = np.linalg.norm(x_lu - exact_solution, ord=np.inf)
        error_qr = np.linalg.norm(x_qr - exact_solution, ord=np.inf)

        table_data.append([test_num, tol,
                           f"{exact_solution}",
                           f"{x_simple}", error_simple, k_simple,
                           f"{x_seidel}", error_seidel, k_seidel,
                           f"{x_lu}", error_lu,
                           f"{x_qr}", error_qr])

headers = ["Test #", "e", "Exact Solution",
           "Simple Iteration (x)", "Δ (Simple)", "k (Simple)",
           "Seidel (x)", "Δ (Seidel)", "k (Seidel)",
           "LU (x)", "Δ (LU)", "QR (x)", "Δ (QR)"]
print(tabulate(table_data, headers=headers, tablefmt="grid"))
