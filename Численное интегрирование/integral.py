import numpy as np
import matplotlib.pyplot as plt
import random
import spline as sp

x = np.array([0, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2])
y = np.array([0, 0.004, 0.015, 0.034, 0.059, 0.089, 0.123, 0.3, 0.2])

# Функция для несобственного интеграла (от 0 до 1)
def f(x):
    # Замена x -> x²: cos(x) / √x -> 2cos(x²), отрезок 0 - 1
    return 2*np.cos(x**2)

# Прямоугольник
def rectangles(x, y):
    """Левые прямоугольники для табличной функции"""
    h = np.diff(x)
    return np.sum(h * y[:-1])

# Трапеция
def trapezoidal(x, y):
    h = np.diff(x)
    return np.sum(h * (y[:-1] + y[1:]) / 2)

# Метод Симпсона
def simpson(x, y):
    h = np.diff(x)
    n = len(x) - 1

    if n % 2:
        h, x, y = h[:-1], x[:-1], y[:-1]
        n -= 1

    idx_even = np.arange(0, n, 2)        # 0, 2, 4, ...
    idx_odd  = np.arange(1, n, 2)        # 1, 3, 5, ...

    h_pair = h[idx_even] + h[idx_odd]

    return np.sum(h_pair / 6 *
                  (y[idx_even] + 4*y[idx_odd] + y[idx_even+2]))

# Метод Гаусса (4 точки)
def gauss4(x, y):
    # Стандартные узлы и веса на [-1,1]
    t = np.array([-np.sqrt((3 + 2*np.sqrt(6/5))/7),
                  -np.sqrt((3 - 2*np.sqrt(6/5))/7),
                   np.sqrt((3 - 2*np.sqrt(6/5))/7),
                   np.sqrt((3 + 2*np.sqrt(6/5))/7)])
    w = np.array([(18 - np.sqrt(30))/36,
                  (18 + np.sqrt(30))/36,
                  (18 + np.sqrt(30))/36,
                  (18 - np.sqrt(30))/36])
    
    a, b = x[0], x[-1]

    # Преобразование узлов Гаусса к интервалу [a, b]
    x_points = 0.5*(b - a)*t + 0.5*(a + b)
    
    # Вычисление значений функции через сплайн в точках Гаусса
    # natural_cubic_spline уже поддерживает векторный ввод x_eval
    y_points = sp.natural_cubic_spline(x, y, x_points)
    
    # Вычисление интеграла
    return 0.5*(b - a) * np.dot(w, y_points)

# Метод Монте-Карло
def monte_carlo(x, y, n_samples=100000):
    a, b = x[0], x[-1]
    y_max = np.max(y)

    rand_x = np.random.uniform(a, b, n_samples)
    rand_y = np.random.uniform(0, y_max, n_samples)
    
    interp_y = sp.natural_cubic_spline(x, y, rand_x)
    
    # Подсчет точек под графиком
    count_under = np.sum(rand_y <= interp_y)
    
    rect_area = (b - a) * y_max
    integral = (count_under / n_samples) * rect_area
    
    return integral

# Несобственный интеграл
def analytic_integral(func, a, b, n = 100):
    if n % 2 != 0:
        n += 1
    
    x = np.linspace(a, b, n + 1)
    y = func(x)
    
    return simpson(x, y)

print("Численные интегралы для табличной функции y(x) на [0,2]:")
print("Прямоугольники :", rectangles(x,y))
print("Трапеции       :", trapezoidal(x,y))
print("Симпсон        :", simpson(x,y))
print("Гаусс 4 точки  :", gauss4(x,y))
print("Монте-Карло    :", monte_carlo(x,y))
print("\nНесобственный ∫₀¹ cos x / √x dx =", analytic_integral(f, 0, 1))

x_fine = np.linspace(0,2,500)
y_spline = sp.natural_cubic_spline(x,y,x_fine)

plt.figure(figsize=(6,4))
plt.plot(x,y,'o',label='табличные узлы')
plt.plot(x_fine,y_spline,label='натуральный кубический сплайн')
plt.legend(); plt.xlabel('x'); plt.ylabel('y')
plt.title('Интерполяция сплайном'); plt.grid(True); plt.show()