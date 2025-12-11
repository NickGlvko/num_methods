import numpy as np
import math
from scipy.integrate import quad

def f(x):
    return 3 * np.cos(3.5 * x) * np.exp(4 * x / 3) + 2 * np.sin(3.5 * x) * np.exp(-2 * x / 3) + 4 * x

def F(x, a=1, b=3, alpha=0, beta=1/6):
    x = np.asarray(x)
    x_safe = np.where(x == b, b - 1e-15, x)
    return f(x_safe) / ((b - x_safe) ** beta)

def compute_exact_integral(a, b, func, tolerance=1e-12):
    result, _ = quad(func, a, b, epsabs=tolerance, epsrel=tolerance)
    return result


def composite_newton_cotes(a, b, n, func_f, alpha=0, beta=1/6):
    if n % 2 != 0:
        n += 1
    h = (b - a) / n
    total = 0.0

    for i in range(n):
        z0 = a + i * h
        z1 = a + (i + 1) * h
        x1, x2, x3 = z0, (z0 + z1) / 2.0, z1


        def moment(s):
            if s == 0:
                return ((b - z0) ** (1 - beta) - (b - z1) ** (1 - beta)) / (1 - beta)
            elif s == 1:
                return b * moment(0) - ((b - z0) ** (2 - beta) - (b - z1) ** (2 - beta)) / (2 - beta)
            elif s == 2:
                return (b ** 2) * moment(0) \
                       - 2 * b * ((b - z0) ** (2 - beta) - (b - z1) ** (2 - beta)) / (2 - beta) \
                       + ((b - z0) ** (3 - beta) - (b - z1) ** (3 - beta)) / (3 - beta)

        mu0, mu1, mu2 = moment(0), moment(1), moment(2)
        X = np.array([[1, 1, 1],
                      [x1, x2, x3],
                      [x1**2, x2**2, x3**2]])
        mu = np.array([mu0, mu1, mu2])
        try:
            A1, A2, A3 = np.linalg.solve(X, mu)
        except np.linalg.LinAlgError:
            A1, A2, A3 = h/6, 4*h/6, h/6
        total += A1 * func_f(x1) + A2 * func_f(x2) + A3 * func_f(x3)
    return total


def composite_gauss(a, b, n, func_f, alpha=0, beta=1/6):
    h = (b - a) / n
    total = 0.0

    for i in range(n):
        z0 = a + i * h
        z1 = a + (i + 1) * h

        def moment(s):
            total_m = 0.0
            for k in range(s + 1):
                binom = math.comb(s, k)
                coeff = binom * (b ** (s - k)) * ((-1) ** k)
                denom = k + 1 - beta
                term = ((b - z0) ** (k + 1 - beta) - (b - z1) ** (k + 1 - beta)) / denom
                total_m += coeff * term
            return total_m

        try:
            mu = [moment(s) for s in range(6)]
        except:
            continue

        M = np.array([[mu[0], mu[1], mu[2]],
                      [mu[1], mu[2], mu[3]],
                      [mu[2], mu[3], mu[4]]])
        rhs = -np.array([mu[3], mu[4], mu[5]])
        try:
            a0, a1, a2 = np.linalg.lstsq(M, rhs, rcond=None)[0]
        except np.linalg.LinAlgError:
            continue


        p_val = (a1 - a2**2 / 3) / 3
        q_val = (2 * a2**3 / 27 - a1 * a2 / 3 + a0) / 2
        D = q_val**2 + p_val**3

        if D >= 0 or abs(p_val) < 1e-15:
            roots = np.roots([1, a2, a1, a0])
            x_nodes = np.real(roots)
        else:
            r = np.sqrt(-p_val)
            cos_phi = np.clip(q_val / (r**3), -1.0, 1.0)
            phi = np.arccos(cos_phi)
            y1 = -2 * r * np.cos(phi / 3)
            y2 = 2 * r * np.cos(np.pi/3 - phi / 3)
            y3 = 2 * r * np.cos(np.pi/3 + phi / 3)
            x1 = y1 - a2 / 3
            x2 = y2 - a2 / 3
            x3 = y3 - a2 / 3
            x_nodes = np.array([x1, x2, x3])

        x_nodes = np.sort(x_nodes)
        if not (np.all(x_nodes >= z0 - 1e-10) and np.all(x_nodes <= z1 + 1e-10)):
            continue


        V = np.vander(x_nodes, 3, increasing=True).T
        try:
            A = np.linalg.lstsq(V, mu[:3], rcond=None)[0]
        except np.linalg.LinAlgError:
            continue

        if np.any(np.isnan(A)) or np.any(np.isinf(A)):
            continue

        total += np.dot(A, func_f(x_nodes))

    return total

def richardson_extrapolation(a, b, f, formula_func, alpha, beta, h0, r):
    h_vals = [h0 / (2**i) for i in range(r + 1)]
    n_vals = [max(1, int(round((b - a) / h))) for h in h_vals]
    

    if formula_func == composite_newton_cotes:
        n_vals = [n + 1 if n % 2 != 0 else n for n in n_vals]
    
    S_vals = [formula_func(a, b, n, f, alpha, beta) for n in n_vals]


    if len(S_vals) >= 3:
        d1, d2 = S_vals[1] - S_vals[0], S_vals[2] - S_vals[1]
        if abs(d1) > 1e-15 and abs(d2) > 1e-15:
            m_est = -np.log(abs(d2 / d1)) / np.log(2)
        else:
            m_est = 6.0 if formula_func == composite_gauss else 4.0
    else:
        m_est = 6.0 if formula_func == composite_gauss else 4.0


    if r == 1:
        S1, S2 = S_vals[0], S_vals[1]
        R = (S2 - S1) / (2**m_est - 1) if abs(2**m_est - 1) > 1e-15 else 0.0
        J = S2 + R
    else:
        A = np.zeros((r + 1, r + 1))
        b_vec = np.array(S_vals)
        for i in range(r + 1):
            A[i, 0] = 1.0
            for k in range(1, r + 1):
                A[i, k] = h_vals[i] ** (m_est + k - 1)
        try:
            sol = np.linalg.solve(A, b_vec)
            J = sol[0]

            R = sum(sol[k] * h_vals[-1] ** (m_est + k - 1) for k in range(1, r + 1))
        except np.linalg.LinAlgError:

            S1, S2 = S_vals[-2], S_vals[-1]
            R = (S2 - S1) / (2**m_est - 1) if abs(2**m_est - 1) > 1e-15 else 0.0
            J = S2 + R

    return J, R, m_est, S_vals, h_vals


def find_optimal_n_richardson(a, b, f, formula_func, alpha=0, beta=1/6,
                              epsilon=1e-6, r=1):
    n = 2
    h0 = (b - a) / n

    while True:
        try:
            J, R, m, _, h_vals = richardson_extrapolation(
                a, b, f, formula_func, alpha, beta, h0, r
            )
            if abs(R) < epsilon:
                n_opt = max(1, int(round((b - a) / h_vals[-1])))
                return n_opt, h_vals[-1], J, m, abs(R)
        except Exception:
            pass
        h0 /= 2


def main():
    a, b = 1.0, 3.0
    alpha, beta = 0.0, 1/6
    eps = 1e-6


    exact_F = compute_exact_integral(a, b, lambda x: F(x, a, b, alpha, beta))
    print(f"F(x) exact integral value: {exact_F:.10f}\n")


    for name, formula in [("Newton-Cotes", composite_newton_cotes),
                          ("Gauss", composite_gauss)]:
        print(f"=== {name} (eps = {eps}) ===")
        for r in [1, 2, 3]:
            try:
                n_opt, h_opt, J_est, m_est, R_est = find_optimal_n_richardson(
                    a, b, f, formula, alpha, beta, eps, r
                )
                error = abs(J_est - exact_F)
                print(f"r = {r}: n = {n_opt:3d}, h = {h_opt:.6f}, "
                      f"J = {J_est:.10f}, m = {m_est:.3f}, |R| = {R_est:.2e}, "
                      f"|error| = {error:.2e}")
            except Exception as e:
                print(f"r = {r}: failed — {e}")
        print()

if __name__ == "__main__":
    main()