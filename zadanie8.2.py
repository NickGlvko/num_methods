import numpy as np
import math


A = 1/35
B = 1/10
xi = 1/17
x0 = 0.0
y0 = np.array([B * math.pi, A * math.pi])
x_final = math.pi


def f(x, y):
    return np.array([
        A * y[1],
        -B * y[0]
    ])


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


def estimate_local_error(x, y, h, method_func, s):
    y_h = method_func(x, y, h)
    

    y_h2 = method_func(x, y, h/2)
    y_h2 = method_func(x + h/2, y_h2, h/2)
    

    error_est = np.linalg.norm(y_h2 - y_h) / (1 - 2**(-s))
    return error_est, y_h, y_h2


def initial_step_size(s, epsilon):
    f0 = f(x0, y0)
    norm_f0 = np.linalg.norm(f0)
    
    x_max = max(abs(x0), abs(x_final))
    delta = (1/x_max)**(s+1) + norm_f0**(s+1)
    
    h = (epsilon / delta)**(1/(s+1))
    return h


def solve_ivp_with_adaptive_step(epsilon):
    x = x0
    y = y0.copy()
    x_values = [x]
    y_values = [y.copy()]
    
    h = initial_step_size(2, 1e-5)
    
    while x < x_final:
        if x + h > x_final:
            h = x_final - x
        

        error_est, y_h, y_h2 = estimate_local_error(x, y, h, runge_kutta_2step, 2)
        

        if error_est > epsilon * (2**2):
            h = h * 0.5
            continue
            
        elif epsilon < error_est <= epsilon * (2**2):
            x += h
            y = y_h2
            x_values.append(x)
            y_values.append(y.copy())
            
            h = h * 0.5
            
        elif epsilon / (2**(2+1)) <= error_est <= epsilon:
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
    
    return np.array(x_values), np.array(y_values)


def exact_solution(x):
    omega = math.sqrt(A * B)
    C1 = B * math.pi
    C2 = (A * A * math.pi) / omega
    
    y1 = C1 * math.cos(omega * x) + C2 * math.sin(omega * x)
    y2 = (-C1 * omega * math.sin(omega * x) + C2 * omega * math.cos(omega * x)) / A
    return np.array([y1, y2])


def main():

    x_values, y_values = solve_ivp_with_adaptive_step(1e-5)
    
    y_final = y_values[-1]
    exact_final = exact_solution(x_final)
    error_final = np.linalg.norm(y_final - exact_final)
    

    print(f"initial condition: y1(0) = {y0[0]:.10f}, y2(0) = {y0[1]:.10f}")
    print(f"Approximate solution: y1(pi) = {y_final[0]:.10f}, y2(pi) = {y_final[1]:.10f}")
    print(f"Exact solution:     y1(pi) = {exact_final[0]:.10f}, y2(pi) = {exact_final[1]:.10f}")
    print(f"Error: {error_final:.10f}")
    print(f"Number of steps: {len(x_values) - 1}")

if __name__ == "__main__":
    main()