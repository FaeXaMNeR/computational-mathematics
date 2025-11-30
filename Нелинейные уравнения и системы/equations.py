import numpy as np

def derivative(f, x, h):
    term1 = (3/2) * (f(x + h) - f(x - h)) / (2 * h)
    term2 = (3/5) * (f(x + 2*h) - f(x - 2*h)) / (4 * h)
    term3 = (1/10) * (f(x + 3*h) - f(x - 3*h)) / (6 * h)
    return term1 - term2 + term3

def f1(x):
    return x**2 - np.exp(x) / 5

def phi1(x):
    if x >= 0 and x < 4:
        return np.sqrt(np.exp(x) / 5)
    elif x >= 4:
        return np.log(5 * x**2)
    else:
        return -np.sqrt(np.exp(x) / 5)


def f2(x):
    return x**2 - 20*np.sin(x) + np.pi

# x = x + a(x² - 20sinx), a = -0.05
def phi2(x):
    return x - 0.05*(x**2 - 20 * np.sin(x))

def bisection(func, a, b, eps):
    if (func(a) * func(b) > 0):
        print("Неверные границы отрезка")
        return
    
    for i in range(100):
        c = (a + b) / 2.0
        if  b - a < eps:
            return c
        
        if func(a) * func(c) <= 0:
            b = c
        else:
            a = c


    return (a + b) / 2.0

def simple_iter(phi, x0, eps):
    x_prev = x0
    
    for i in range(100):
        x_next = phi(x_prev)

        if np.abs(x_next - x_prev) < eps:
            return x_next
        
        x_prev = x_next

    return x_prev

def newton(func, x0, eps, mod_flag):
    x_prev = x0
    h = 2.0 / (2.0**9) # наиболее подходящий шаг, исходя из 1 ЛР

    if mod_flag == 1:
        deriv = derivative(func, x0, h)

    for i in range(100):
        if mod_flag == 0:
            deriv = derivative(func, x_prev, h)
            
        x_next = x_prev - func(x_prev) / deriv

        if np.abs(x_next - x_prev < eps):
            return x_next
        
        x_prev = x_next

    return x_prev


eps = 1e-5
x1 = bisection(f1, -1, 0.5, eps)
x2 = bisection(f1, 0, 1, eps)
x3 = bisection(f1, 4, 5, eps)

x4 = bisection(f2, 2, 3, eps)
x5 = bisection(f2, -0.5, 0.5, eps)

y1 = simple_iter(phi1, -1.0, eps)
y2 = simple_iter(phi1, 1.0, eps)
y3 = simple_iter(phi1, 4.0, eps)

y4 = simple_iter(phi2, 1.0, eps)
y5 = simple_iter(phi2, 0.0, eps)

flag = 0

z1 = newton(f1, -1.0, eps, flag)
z2 = newton(f1, 1.0, eps, flag)
z3 = newton(f1, 5, eps, flag)

z4 = newton(f2, 3, eps, flag)
z5 = newton(f2, 0, eps, flag)

true_x1, true_x2, true_x3 = -0.37142, 0.60527, 4.70794
true_x4, true_x5 = 2.75295, 0

print("Истинные корни:            ", true_x1, true_x2, true_x3, true_x4, true_x5)
print(f"Метод половинного деления:  {x1:.5f}, {x2:.5f}, {x3:.5f}, {x4:.5f}, {x5:.5f}")
print(f"Метод простой итерации:     {y1:.5f}, {y2:.5f}, {y3:.5f}, {y4:.5f}, {y5:.5f}")
print(f"Метод Ньютона:              {z1:.5f}, {z2:.5f}, {z3:.5f}, {z4:.5f}, {z5:.5f}")