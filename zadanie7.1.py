import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
import math


def f(x):
    return 3 * np.cos(3.5 * x) * np.exp(4 * x / 3) + 2 * np.sin(3.5 * x) * np.exp(-2 * x / 3) + 4 * x


def F(x, a=1, b=3, alpha=0, beta=1/6):
    x = np.asarray(x)
    x_safe = np.where(x == b, b - 1e-15, x)
    return f(x_safe) / ((b - x_safe) ** beta)


def composite_left_rectangles(a, b, n, func):
    h = (b - a) / n
    x = np.linspace(a, b - h, n)
    return h * np.sum(func(x))


def composite_midpoint_rectangles(a, b, n, func):
    h = (b - a) / n
    x = a + (np.arange(n) + 0.5) * h
    return h * np.sum(func(x))


def composite_trapezoid(a, b, n, func):
    h = (b - a) / n
    x = np.linspace(a, b, n + 1)
    y = func(x)
    return h * (0.5 * y[0] + np.sum(y[1:n]) + 0.5 * y[n])


def composite_simpson(a, b, n, func):
    if n % 2 != 0:
        n += 1
    
    h = (b - a) / n
    x = np.linspace(a, b, n + 1)
    y = func(x)
    

    odd_sum = np.sum(y[1:n:2])
    even_sum = np.sum(y[2:n-1:2])
    
    return h/3 * (y[0] + 4*odd_sum + 2*even_sum + y[n])


def composite_newton_cotes(a, b, n, func_f, alpha=0, beta=1/6):

    if n % 2 != 0:
        n += 1
    
    h = (b - a) / n
    total = 0.0
    
    for i in range(n):
        z0 = a + i * h
        z1 = a + (i + 1) * h

        x1 = z0
        x2 = (z0 + z1) / 2.0
        x3 = z1


        def moment(s):
            if s == 0:
                return ((b - z0)**(1 - beta) - (b - z1)**(1 - beta)) / (1 - beta)
            elif s == 1:
                term1 = b * moment(0)
                term2 = ((b - z0)**(2 - beta) - (b - z1)**(2 - beta)) / (2 - beta)
                return term1 - term2
            elif s == 2:
                term1 = b**2 * moment(0)
                term2 = 2 * b * ((b - z0)**(2 - beta) - (b - z1)**(2 - beta)) / (2 - beta)
                term3 = ((b - z0)**(3 - beta) - (b - z1)**(3 - beta)) / (3 - beta)
                return term1 - term2 + term3
        
        mu0 = moment(0)
        mu1 = moment(1)
        mu2 = moment(2)
        

        X = np.array([
            [1, 1, 1],
            [x1, x2, x3],
            [x1**2, x2**2, x3**2]
        ])
        mu = np.array([mu0, mu1, mu2])
        A1, A2, A3 = np.linalg.solve(X, mu)
        

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
        except Exception as e:
            continue


        M = np.array([
            [mu[0], mu[1], mu[2]],
            [mu[1], mu[2], mu[3]],
            [mu[2], mu[3], mu[4]]
        ])
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
            cos_phi = q_val / (r**3)
            cos_phi = np.clip(cos_phi, -1.0, 1.0)
            phi = np.arccos(cos_phi)

            y1 = -2 * r * np.cos(phi / 3)
            y2 = 2 * r * np.cos(np.pi/3 - phi / 3)
            y3 = 2 * r * np.cos(np.pi/3 + phi / 3)

            x1 = y1 - a2 / 3
            x2 = y2 - a2 / 3
            x3 = y3 - a2 / 3
            x_nodes = np.array([x1, x2, x3])

        x_nodes = np.sort(x_nodes)


        if not (np.all(x_nodes >= z0 - 1e-12) and np.all(x_nodes <= z1 + 1e-12)):
            continue

        V = np.vander(x_nodes, 3, increasing=True).T  # [[1,1,1], [x1,x2,x3], [x1^2,x2^2,x3^2]]
        try:
            A = np.linalg.lstsq(V, mu[:3], rcond=None)[0]
        except np.linalg.LinAlgError:
            continue

        if np.any(np.isinf(A)) or np.any(np.isnan(A)):
            continue

        total += np.dot(A, func_f(x_nodes))

    return total


def compute_exact_integral(a, b, func, tolerance=1e-10):
    result, _ = quad(func, a, b, epsabs=tolerance, epsrel=tolerance)
    return result


def main():
    a = 1
    b = 3
    alpha = 0
    beta = 1/6
    

    exact_integral_f = compute_exact_integral(a, b, f)
    print(f"f(x) exact integral value: {exact_integral_f:.10f}")
    

    exact_integral_F = compute_exact_integral(a, b, lambda x: F(x, a, b, alpha, beta))
    print(f"F(x) exact integral value: {exact_integral_F:.10f}")
    

    n_values = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
    

    results = {
        'left_rect': [],
        'midpoint_rect': [],
        'trapezoid': [],
        'simpson': [],
        'newton_cotes': [],
        'gauss': []
    }
    

    for n in n_values:
        results['left_rect'].append(composite_left_rectangles(a, b, n, f))
        results['midpoint_rect'].append(composite_midpoint_rectangles(a, b, n, f))
        results['trapezoid'].append(composite_trapezoid(a, b, n, f))
        results['simpson'].append(composite_simpson(a, b, n, f))
        

        results['newton_cotes'].append(composite_newton_cotes(a, b, n, f, alpha=alpha, beta=beta))
        results['gauss'].append(composite_gauss(a, b, n, f, alpha=alpha, beta=beta)) if n<=64 else results['gauss'].append(0)
    
 
    errors = {
        'left_rect': [abs(val - exact_integral_f) for val in results['left_rect']],
        'midpoint_rect': [abs(val - exact_integral_f) for val in results['midpoint_rect']],
        'trapezoid': [abs(val - exact_integral_f) for val in results['trapezoid']],
        'simpson': [abs(val - exact_integral_f) for val in results['simpson']],
        'newton_cotes': [abs(val - exact_integral_F) for val in results['newton_cotes']],
        'gauss': [abs(val - exact_integral_F) for val in results['gauss']]
    }
    

    plt.figure(figsize=(12, 8))
    

    plt.subplot(2, 1, 1)
    plt.loglog(n_values, errors['left_rect'], 'o-', label='Левые прямоугольники')
    plt.loglog(n_values, errors['midpoint_rect'], 's-', label='Средние прямоугольники')
    plt.loglog(n_values, errors['trapezoid'], 'd-', label='Трапеции')
    plt.loglog(n_values, errors['simpson'], '^-', label='Симпсон')
    plt.xlabel('Количество разбиений n')
    plt.ylabel('Абсолютная погрешность')
    plt.title('Зависимость погрешности от количества разбиений (f(x))')
    plt.grid(True, which="both", ls="-")
    plt.legend()
    
 
    plt.subplot(2, 1, 2)
    plt.loglog(n_values, errors['newton_cotes'], 'o-', label='Ньютон-Котес (3-точечная)')
    plt.loglog(n_values, errors['gauss'], 's-', label='Гаусс (3-точечная)')
    plt.xlabel('Количество разбиений n')
    plt.ylabel('Абсолютная погрешность')
    plt.title('Зависимость погрешности от количества разбиений (F(x))')
    plt.grid(True, which="both", ls="-")
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    
 
    print("\nCalculation results:")
    print(f"{'n':<5} | {'Left':>12} | {'Midpoint':>12} | {'Trapezoid':>12} | {'Simpson':>12} | {'Newton-Cotes':>15} | {'Gauss':>12}")
    print("-" * 85)
    for i, n in enumerate(n_values):
        print(f"{n:<5} | {results['left_rect'][i]:>12.8f} | {results['midpoint_rect'][i]:>12.8f} | "
              f"{results['trapezoid'][i]:>12.8f} | {results['simpson'][i]:>12.8f} | "
              f"{results['newton_cotes'][i]:>15.8f} | {results['gauss'][i]:>12.8f}")

if __name__ == "__main__":
    main()