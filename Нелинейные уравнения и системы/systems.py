import numpy as np

def derivative(f, x, h):
    term1 = (3/2) * (f(x + h) - f(x - h)) / (2 * h)
    term2 = (3/5) * (f(x + 2*h) - f(x - 2*h)) / (4 * h)
    term3 = (1/10) * (f(x + 3*h) - f(x - 3*h)) / (6 * h)
    return term1 - term2 + term3

# eps = 1e-3
def f1(x, y):
    return np.array([np.cos(x - 1) + y - 0.5,
                     x - np.cos(y) - 3])

def phi1(x, y):
    return np.array([3 + np.cos(y),
                     0.5 - np.cos(x-1)])

# eps = 1e-5
def f2(x, y):
    return np.array([2*x**2 - x*y - 5*x + 1,
                     x + 3*np.log10(x) - y**2])

def phi2(x, y):
    return np.array([(y + 5 - 1/x)/2,
                     np.sign(y) * np.sqrt(3*np.log10(x) + x)])

def simple_iter(phi, x0, eps):
    x_prev = x0

    for i in range(100):
        x_next = phi(x_prev[0], x_prev[1])

        if np.linalg.norm(x_next - x_prev) < eps:
            return x_next
        
        x_prev = x_next

    return x_prev

def solve(A, b):
    n = len(b)
    x = np.zeros(n)

    # Точность
    eps = 1e-6 
    
    r = b - A @ x
    r_wave = r
    ro, alpha, omega = 1, 1, 1
    v = np.zeros(n)
    p = np.zeros(n)

    for i in range(100):
        if ro == 0 or omega == 0:
            continue
        old_ro = ro # ro_k-1
        ro = np.dot(r_wave, r)
        beta = ro / old_ro * alpha / omega
        p = r + beta * (p - omega * v)
        v = A @ p
        alpha = ro / np.dot(r_wave, v)
        s = r - alpha * v
        t = A @ s
        omega = np.dot(t, s) / np.dot(t, t)
        x = x + omega * s + alpha * p
        r = s - omega * t

        if np.linalg.norm(b - A @ x) < eps:
            break

    return x

def jacobian(J, func, x, h):
    n = len(x)
    m = len(func(x[0], x[1]))
    for i in range(m):
        for j in range(n):
            def f_component(var):
                x_perturbed = x.copy()
                x_perturbed[j] = var
                return func(x_perturbed[0], x_perturbed[1])[i]
            
            J[i, j] = derivative(f_component, x[j], h)

    return J

def newton(func, x0, eps, mod_flag):
    x_prev = x0
    h = 2.0 / (2.0**9)

    # Было: x_next = x_prev - func(x_prev) / derivative(func, x_prev, h)
    # Стало: J * x_next = J * x_prev - func(x_prev)

    J = np.zeros((2, 2))
    if mod_flag == 1:
        J = jacobian(J, func, x0, h)
    for i in range(100):
        if mod_flag == 0:
            J = jacobian(J, func, x_prev, h)
        b = J @ x_prev - func(x_prev[0], x_prev[1])

        x_next = solve(J, b)

        if np.linalg.norm(x_next - x_prev) < eps:
            return x_next
        
        x_prev = x_next

    return x_prev


eps1 = 1e-3
flag = 1
x1, y1 = simple_iter(phi1, [3, 2], eps1), newton(f1, [3, 2], eps1, flag)

eps2 = 1e-5
x2, y2 = simple_iter(phi2, [3, 2], eps2), newton(f2, [3, 2], eps2, flag)
x3, y3 = simple_iter(phi2, [1, -1], eps2), newton(f2, [1, -1], eps2, flag)


print("Метод простой итерации: ", x1, x2, x3)
print("Метод Ньютона:          ", y1, y2, y3)