import math

def sqrt_g(c, eps=1e-6):
    r = c
    while True:
        r_next = 0.5 * (r + c / r)
        if abs(r_next - r) < eps:
            break
        r = r_next
    return r_next

def cos_t(x, eps=1e-6):
    result = 0
    term = 1
    k = 0
    while abs(term) > eps:
        result += term
        term *= -x * x / ((2 * k + 1) * (2 * k + 2))
        k += 1
    return result

def arctg_t(x, eps=1e-6):
    result = 0
    term = x
    k = 0
    while abs(term) > eps:
        result += term
        term *= -x * x * (2 * k + 1) / (2 * k + 3)
        k += 1
    return result

def z(x, eps=1e-6):
    u = sqrt_g(2 * x + 0.4, eps)
    phi = cos_t(3 * x + 1, eps)
    v = arctg_t(phi, eps)
    z_val = u * v
    return z_val, u, v, phi

a = 0.01
b = 0.06
h = 0.005

x_values = [a + i * h for i in range(int((b - a) / h) + 2)]
results = [z(x) for x in x_values]

delta_phi = 10**(-6)/(2 * 0.721 * 0.873)
delta_v = 10**(-6)/(4 * 0.721)
delta_u = 10**(-6)/(2 * 0.475)
delta_z = 10**(-6)

print(f"{'x':<10} {'phi(x)':<15} {'Δphi':<15} {'py_phi(x)':<15} {'Δpy_phi':<15} {'v(x)':<15} {'Δv':<15} {'py_v(x)':<15} {'Δpy_v':<15} {'u(x)':<15} {'Δu':<15} {'py_u(x)':<15} {'Δpy_u':<15} {'z(x)':<15} {'Δz=10^(-6)':<15} {'py_z(x)':<15} {'Δpy_z':<15}")
print("-" * 270)

for x, (z_val, u, v, phi) in zip(x_values, results):
    py_u = math.sqrt(2 * x + 0.4)
    py_phi = math.cos(3 * x + 1)
    py_v = math.atan(py_phi)
    py_z = py_u * py_v

    delta_py_phi = abs(phi - py_phi)
    delta_py_v = abs(v - py_v)
    delta_py_u = abs(u - py_u)
    delta_py_z = abs(z_val - py_z)

    print(f"{x:<10.4f} {phi:<15.8f} {delta_phi:<15.8e} {py_phi:<15.8f} {delta_py_phi:<15.8e} {v:<15.8f} {delta_v:<15.8e} {py_v:<15.8f} {delta_py_v:<15.8e} {u:<15.8f} {delta_u:<15.8e} {py_u:<15.8f} {delta_py_u:<15.8e} {z_val:<15.8f} {delta_z:<15.8e} {py_z:<15.8f} {delta_py_z:<15.8e}")