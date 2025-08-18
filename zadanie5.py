import numpy as np
from scipy.linalg import inv, norm, qr, hessenberg


def generate_matrices(n):
    Lambda = np.diag(np.random.randn(n))
    C = np.random.randn(n, n)
    C_inv = inv(C)
    A = C_inv @ Lambda @ C
    return A, Lambda, C


def power_method(A, delta=1e-8, rtol=1e-6, max_iter=1000):
    n = A.shape[0]
    y = np.random.randn(n)
    z = y / norm(y)
    lambda_avg_prev = 0
    for k in range(1, max_iter + 1):
        y = A @ z
        I = np.where(np.abs(z) > delta)[0]
        if len(I) == 0:
            raise ValueError("All components are smaller than delta")
        lambda_k = y[I] / z[I]
        lambda_avg = np.mean(lambda_k)
        if k > 1 and np.abs(lambda_avg - lambda_avg_prev) <= rtol * np.abs(lambda_avg):
            return lambda_avg, z
        lambda_avg_prev = lambda_avg
        z = y / norm(y)
    raise ValueError("Power method did not converge")


def inverse_power_method(A, sigma0, delta=1e-8, rtol=1e-6, max_iter=1000):

    n = A.shape[0]
    y = np.random.randn(n)
    z = y / norm(y)
    sigma = sigma0
    for k in range(1, max_iter + 1):
        I = np.eye(n)
        try:
            y = np.linalg.solve(A - sigma * I, z)
        except np.linalg.LinAlgError:
            print(f"Matrix is singular for sigma = {sigma}")
            break
        I_indices = np.where(np.abs(y) > delta)[0]
        if len(I_indices) == 0:
            raise ValueError("All components of y are smaller than delta")
        mu_k = z[I_indices] / y[I_indices]
        mu_avg = np.mean(mu_k)
        sigma_new = sigma + mu_avg
        if k > 1 and np.abs(sigma_new - sigma) <= rtol * np.abs(sigma_new):
            z = y / norm(y)
            return sigma_new, z
        sigma = sigma_new
        z = y / norm(y)
    raise ValueError("Inverse power method did not converge")


def qr_algorithm(B, epsilon=1e-8, max_iter=1000):

    n = B.shape[0]
    eigenvalues = []
    m = n
    while m > 0:
        if m == 1:
            eigenvalues.append(B[0, 0])
            break
        iter_count = 0
        while m > 1 and np.abs(B[m - 1, m - 2]) > epsilon and iter_count < max_iter:
            s = B[m - 1, m - 1]
            Q, R = qr(B[:m, :m] - s * np.eye(m))
            B[:m, :m] = R @ Q + s * np.eye(m)
            iter_count += 1
        if m > 1 and np.abs(B[m - 1, m - 2]) <= epsilon:
            eigenvalues.append(B[m - 1, m - 1])
            m -= 1
        else:
            raise ValueError(f"QR algorithm did not converge for submatrix of size {m}")
    return np.array(eigenvalues[::-1])



n = int(input("Enter the dimension n: "))

# Generate matrices
A, Lambda, C = generate_matrices(n)
print("Matrix A:")
print(A)
print("True eigenvalues:", np.diag(Lambda))

# Part 1: Power method
lambda_max, x_max = power_method(A)
print("\nPower method:")
print("Largest magnitude eigenvalue:", lambda_max)
print("Corresponding eigenvector:", x_max)

# Part 2: Inverse power method for all eigenvalues
eigenvalues_true = np.diag(Lambda)
initial_shifts = eigenvalues_true + 0.1 * np.random.randn(n)
eigenvalues_ipm = []
eigenvectors_ipm = []
for shift in initial_shifts:
    lambda_ipm, x_ipm = inverse_power_method(A, shift)
    eigenvalues_ipm.append(lambda_ipm)
    eigenvectors_ipm.append(x_ipm)
print("\nInverse power method:")
print("Eigenvalues:", eigenvalues_ipm)
print("Eigenvectors:")
for evec in eigenvectors_ipm:
    print(evec)

# Part 3: QR algorithm
H = hessenberg(A)
eigenvalues_qr = qr_algorithm(H.copy())
print("\nQR algorithm:")
print("Eigenvalues:", eigenvalues_qr)