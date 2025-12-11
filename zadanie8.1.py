import numpy as np
import math


A = 1/35
B = 1/10
xi = 1/17
epsilon = 1e-4
s = 2


x0 = 0.0
x_final = math.pi
y0 = np.array([B * math.pi, A * math.pi])


def f(x, y):
    return np.array([A * y[1], -B * y[0]])


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

def initial_step_size(s, epsilon):
    f0 = f(x0, y0)
    norm_f0 = np.linalg.norm(f0)
    
    x_max = max(abs(x0), abs(x_final))
    delta = (1/x_max)**(s+1) + norm_f0**(s+1)
    
    h = (epsilon / delta)**(1/(s+1))
    return h


def solve_with_fixed_step():
    h = initial_step_size(2, 1e-4)
    x = x0
    y = y0.copy()
    
    x_values = [x]
    y_values = [y.copy()]
    
    while x < x_final:
        if x + h > x_final:
            h = x_final - x
        
        y = runge_kutta_2step(x, y, h)
        x += h
        
        x_values.append(x)
        y_values.append(y.copy())
    
    return np.array(x_values), np.array(y_values)


def exact_solution(x):
    omega = math.sqrt(A * B)
    C1 = B * math.pi
    C2 = (A * A * math.pi) / omega
    
    y1 = C1 * math.cos(omega * x) + C2 * math.sin(omega * x)
    y2 = (-C1 * omega * math.sin(omega * x) + C2 * omega * math.cos(omega * x)) / A
    return np.array([y1, y2])


if __name__ == "__main__":

    x_values, y_values = solve_with_fixed_step()
    

    y_final = y_values[-1]
    exact_final = exact_solution(x_final)
    error = np.linalg.norm(y_final - exact_final)
    

    print(f"Initial condition: y1(0) = {y0[0]:.10f}, y2(0) = {y0[1]:.10f}")
    print(f"Exact solution:    y1(pi) = {exact_final[0]:.10f}, y2(pi) = {exact_final[1]:.10f}")
    print(f"Approximate solution: y1(pi) = {y_final[0]:.10f}, y2(pi) = {y_final[1]:.10f}")
    print(f"Error: {error:.10f}")
    print(f"Length of the step: h = {initial_step_size(2, 1e-4):.6f}")