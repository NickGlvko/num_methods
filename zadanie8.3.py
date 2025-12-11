import numpy as np
import math
import matplotlib.pyplot as plt


A = 1/35
B = 1/10
x0 = 0.0
xi = 1/17
x_final = math.pi
y0 = np.array([B * math.pi, A * math.pi])


def f(x, y):
    return np.array([
        A * y[1],
        -B * y[0]
    ])


def runge_kutta_4step(x, y, h):
    k1 = h * f(x, y)
    k2 = h * f(x + 0.5 * h, y + 0.5 * k1)
    k3 = h * f(x + 0.5 * h, y + 0.5 * k2)
    k4 = h * f(x + h, y + k3)
    
    y_new = y + (1/6) * (k1 + 2*k2 + 2*k3 + k4)
    return y_new


def runge_kutta_2step(x, y, h):
    c2 = xi
    a21 = c2
    b1 = 1 - 1/(2*c2)
    b2 = 1/(2*c2)
    

    k1 = h * f(x, y)
    x2 = x + c2 * h
    y2 = y + a21 * k1
    k2 = h * f(x2, y2)
    

    y_new = y + b1 * k1 + b2 * k2
    return y_new


def exact_solution(x):
    omega = math.sqrt(A * B)
    C1 = B * math.pi
    C2 = (A * A * math.pi) / omega
    
    y1 = C1 * math.cos(omega * x) + C2 * math.sin(omega * x)
    y2 = (-C1 * omega * math.sin(omega * x) + C2 * omega * math.cos(omega * x)) / A
    return np.array([y1, y2])


def solve_with_fixed_step(method_func, h):
    x = x0
    y = y0.copy()
    
    x_values = [x]
    y_values = [y.copy()]
    exact_values = [exact_solution(x)]
    
    while x < x_final:
        if x + h > x_final:
            h = x_final - x
        
        y = method_func(x, y, h)
        x += h
        
        x_values.append(x)
        y_values.append(y.copy())
        exact_values.append(exact_solution(x))
    
    return np.array(x_values), np.array(y_values), np.array(exact_values)

def initial_step_size(s, epsilon):
    f0 = f(x0, y0)
    norm_f0 = np.linalg.norm(f0)
    
    x_max = max(abs(x0), abs(x_final))
    delta = (1/x_max)**(s+1) + norm_f0**(s+1)
    
    h = (epsilon / delta)**(1/(s+1))
    return h

def estimate_local_error(x, y, h, method_func, s):

    y_h = method_func(x, y, h)
    y_h2 = method_func(x, y, h/2)
    y_h2 = method_func(x + h/2, y_h2, h/2)

    error_est = np.linalg.norm(y_h2 - y_h) / (1 - 2**(-s))
    return error_est, y_h, y_h2


def solve_with_adaptive_step(method_func, s, epsilon):
    x = x0
    y = y0.copy()
    
    x_values = [x]
    y_values = [y.copy()]
    h_values = []
    local_errors = []
    estimated_errors = []
    f_eval_count = 0
    

    h = initial_step_size(s, epsilon)
    
    while x < x_final:
        if x + h > x_final:
            h = x_final - x
        
        error_est, y_h, y_h2 = estimate_local_error(x, y, h, method_func, s)
        f_eval_count += (3 * (4 if s == 4 else 2) - 1)
        exact_next = exact_solution(x + h)
        local_error = np.linalg.norm(y_h2 - exact_next)
        

        h_values.append(h)
        local_errors.append(local_error)
        estimated_errors.append(error_est)
        

        if error_est > epsilon * 2**s:

            h = h * 0.5
        elif epsilon < error_est <= epsilon * 2**s:

            x += h
            y = y_h2
            x_values.append(x)
            y_values.append(y.copy())
            h = h * 0.5
        elif epsilon / 2**(s+1) <= error_est <= epsilon:

            x += h
            y = y_h
            x_values.append(x)
            y_values.append(y.copy())

        else:
            x += h
            y = y_h
            x_values.append(x)
            y_values.append(y.copy())
            h = h * 2
    
    return (np.array(x_values), np.array(y_values), np.array(h_values), 
            np.array(local_errors), np.array(estimated_errors), f_eval_count)

def find_optimal_fixed_step(method_func, s, epsilon):
    h_low = 1e-6
    h_high = (x_final - x0) / 2
    
    while h_high - h_low > 1e-8:
        h_mid = (h_low + h_high) / 2
        
        x_vals, y_vals, exact_vals = solve_with_fixed_step(method_func, h_mid)
        
        errors = np.linalg.norm(y_vals - exact_vals, axis=1)
        max_error = np.max(errors)
        
        if max_error > epsilon:
            h_high = h_mid
        else:
            h_low = h_mid
    
    return (h_low + h_high) / 2

def analyze_methods():
    epsilon = 1e-4

    print("\nRunge-Kutta 4step:")
    h_opt_rk4 = find_optimal_fixed_step(runge_kutta_4step, 4, epsilon)
    print(f"Optimal step: h = {h_opt_rk4:.6f}")
    
    x_rk4, y_rk4, exact_rk4 = solve_with_fixed_step(runge_kutta_4step, h_opt_rk4)
    rk4_errors = np.linalg.norm(y_rk4 - exact_rk4, axis=1)
    
    print("\nRunge-kutta 2step:")
    h_opt_rk2 = find_optimal_fixed_step(runge_kutta_2step, 2, epsilon)
    print(f"Optimal step: h = {h_opt_rk2:.6f}")
    
    x_rk2, y_rk2, exact_rk2 = solve_with_fixed_step(runge_kutta_2step, h_opt_rk2)
    rk2_errors = np.linalg.norm(y_rk2 - exact_rk2, axis=1)
    

    plt.figure(figsize=(10, 6))
    plt.plot(x_rk4, rk4_errors, 'b-', linewidth=2, label='RK4 (4-й порядок)')
    plt.plot(x_rk2, rk2_errors, 'r--', linewidth=2, label='RK2 (2-й порядок)')
    plt.axhline(y=epsilon, color='k', linestyle=':', label='Заданная точность')
    plt.xlabel('x')
    plt.ylabel('Полная погрешность')
    plt.title('Зависимость полной погрешности от x при постоянном шаге')
    plt.legend()
    plt.grid(True)
    plt.show()
    

    x_rk4_ad, y_rk4_ad, h_rk4, local_err_rk4, est_err_rk4, f_count_rk4 = solve_with_adaptive_step(
        runge_kutta_4step, 4, epsilon)
    

    x_rk2_ad, y_rk2_ad, h_rk2, local_err_rk2, est_err_rk2, f_count_rk2 = solve_with_adaptive_step(
        runge_kutta_2step, 2, epsilon)
    
    plt.figure(figsize=(10, 6))
    plt.step(x_rk4_ad[:-1], h_rk4, 'b-', where='post', linewidth=2, label='RK4 (4-й порядок)')
    plt.step(x_rk2_ad[:-1], h_rk2, 'r--', where='post', linewidth=2, label='RK2 (2-й порядок)')
    plt.xlabel('x')
    plt.ylabel('Шаг интегрирования h')
    plt.title('Зависимость шага интегрирования от x')
    plt.legend()
    plt.grid(True)
    plt.show()
    

    xi_rk4 = local_err_rk4 / est_err_rk4
    xi_rk2 = local_err_rk2 / est_err_rk2
    
    plt.figure(figsize=(10, 6))
    plt.plot(x_rk4_ad[:-1], xi_rk4, 'b-', linewidth=2, label='RK4 (4-й порядок)')
    plt.plot(x_rk2_ad[:-1], xi_rk2, 'r--', linewidth=2, label='RK2 (2-й порядок)')
    plt.axhline(y=1, color='k', linestyle=':', label='ξ = 1')
    plt.xlabel('x')
    plt.ylabel('Отношение ξ = ||l|| / ||ρ||')
    plt.title('Отношение истинной локальной погрешности к оценке')
    plt.legend()
    plt.grid(True)
    plt.ylim(0, max(2, np.max([np.max(xi_rk4), np.max(xi_rk2)])))
    plt.show()
    

    epsilons = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5]
    f_counts_rk4 = []
    f_counts_rk2 = []
    
    for eps in epsilons:
        print(f"\nCalculation for eps = {eps:.0e}")
        
        _, _, _, _, _, f_count = solve_with_adaptive_step(runge_kutta_4step, 4, eps)
        f_counts_rk4.append(f_count)
        print(f"RK4: number of calculations = {f_count}")
        
        # Для RK2
        _, _, _, _, _, f_count = solve_with_adaptive_step(runge_kutta_2step, 2, eps)
        f_counts_rk2.append(f_count)
        print(f"RK2: number of calculations = {f_count}")
    
    plt.figure(figsize=(10, 6))
    plt.loglog(epsilons, f_counts_rk4, 'bo-', linewidth=2, markersize=8, label='RK4 (4-й порядок)')
    plt.loglog(epsilons, f_counts_rk2, 'rs--', linewidth=2, markersize=8, label='RK2 (2-й порядок)')
    plt.xlabel('Заданная точность eps')
    plt.ylabel('Количество вычислений правой части')
    plt.title('Зависимость количества вычислений правой части от точности')
    plt.legend()
    plt.grid(True, which="both", ls="-")
    plt.show()


if __name__ == "__main__":
    analyze_methods()