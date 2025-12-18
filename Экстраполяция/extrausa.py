import numpy as np
import matplotlib.pyplot as plt

years = np.array([1910, 1920, 1930, 1940, 1950, 1960, 1970, 1980, 1990, 2000])
population = np.array([92228496, 106021537, 123202624, 132164569, 
                       151325798, 179323175, 203211926, 226545805, 
                       248709873, 281421906])

def solve(A, b):
    n = len(b)
    x = np.zeros(n)

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

def newton_interpolation(x, y, x_eval):
    n = len(x)

    coef = np.zeros([n, n])
    coef[:,0] = y
    
    for j in range(1, n):
        for i in range(n - j):
            coef[i, j] = (coef[i+1, j-1] - coef[i, j-1]) / (x[i+j] - x[i])
    

    result = np.zeros_like(x_eval, dtype=float)
    for i in range(n):
        term = coef[0, i] * np.ones_like(x_eval)
        for j in range(i):
            term *= (x_eval - x[j])
        result += term
    
    return result

def tridiagonal_solve(a, b, c, d):
    """
    Решение трехдиагональной системы методом прогонки
    a - нижняя диагональ (a[0] не используется)
    b - главная диагональ
    c - верхняя диагональ
    d - правая часть
    """
    n = len(d)
    
    # Прямой ход прогонки
    alpha = np.zeros(n)
    beta = np.zeros(n)
    
    # Первое уравнение
    alpha[0] = -c[0] / b[0]
    beta[0] = d[0] / b[0]
    
    for i in range(1, n):
        denominator = b[i] + a[i] * alpha[i-1]
        alpha[i] = -c[i] / denominator
        beta[i] = (d[i] - a[i] * beta[i-1]) / denominator
    
    # Обратный ход прогонки
    x = np.zeros(n)
    x[n-1] = beta[n-1]
    
    for i in range(n-2, -1, -1):
        x[i] = alpha[i] * x[i+1] + beta[i]
    
    return x

def natural_cubic_spline(x, y, x_eval):
    n = len(x)
    h = np.diff(x)
    
    # Правая часть системы уравнений для моментов
    alpha = np.zeros(n)
    for i in range(1, n-1):
        alpha[i] = 3/h[i]*(y[i+1]-y[i]) - 3/h[i-1]*(y[i]-y[i-1])
    
    # Создание трехдиагональной матрицы
    # Для естественного сплайна: m0 = 0, mn-1 = 0
    a = np.zeros(n)  # нижняя диагональ
    b = np.zeros(n)  # главная диагональ  
    c = np.zeros(n)  # верхняя диагональ
    
    b[0] = 1.0
    c[0] = 0.0
    
    b[n-1] = 1.0
    a[n-1] = 0.0
    
    for i in range(1, n-1):
        a[i] = h[i-1]
        b[i] = 2*(h[i-1] + h[i])
        c[i] = h[i]
    
    m = tridiagonal_solve(a, b, c, alpha)
    
    a_coef = y[:-1]
    b_coef = np.zeros(n-1)
    c_coef = m[:-1]
    d_coef = np.zeros(n-1)
    
    for i in range(n-1):
        b_coef[i] = (y[i+1] - y[i])/h[i] - h[i]*(2*m[i] + m[i+1])/3
        d_coef[i] = (m[i+1] - m[i])/(3*h[i])
    
    result = np.zeros_like(x_eval)
    for idx, xi in enumerate(x_eval):
        i = np.searchsorted(x, xi) - 1
        if i < 0:
            i = 0
        elif i >= n-1:
            i = n-2
        
        dx = xi - x[i]
        result[idx] = a_coef[i] + b_coef[i]*dx + c_coef[i]*dx**2 + d_coef[i]*dx**3
    
    return result

def least_squares_poly(x, y, degree, x_eval):
    # Масштабируем годы к диапазону [0, 1] или [-1, 1]
    x_mean = np.mean(x)
    x_std = np.std(x)
    
    # Если стандартное отклонение близко к 0, используем просто смещение
    if x_std < 1e-10:
        x_scaled = x - x_mean
    else:
        x_scaled = (x - x_mean) / x_std
    
    if x_std < 1e-10:
        x_eval_scaled = x_eval - x_mean
    else:
        x_eval_scaled = (x_eval - x_mean) / x_std
    
    A = np.vander(x_scaled, degree + 1)
    
    # Нормальные уравнения: (A^T * A) * coeffs = A^T * y
    ATA = A.T @ A
    ATy = A.T @ y
    
    coeffs_scaled = solve(ATA, ATy)
    
    result_scaled = np.polyval(coeffs_scaled, x_eval_scaled)
    
    return result_scaled

x_interp = np.linspace(1910, 2010, 101)

newton_values = newton_interpolation(years, population, x_interp)
spline_values = natural_cubic_spline(years, population, x_interp)
lsq_values = least_squares_poly(years, population, 3, x_interp)

plt.figure(figsize=(12, 8))

plt.scatter(years, population, color='black', s=100, zorder=5, label='Исходные данные')

plt.plot(x_interp, newton_values, 'b-', linewidth=2, label='Полином Ньютона')
plt.plot(x_interp, spline_values, 'r--', linewidth=2, label='Кубический сплайн')
plt.plot(x_interp, lsq_values, 'g-.', linewidth=2, label='МНК (полином 3-й степени)')

plt.axvline(x=2000, color='gray', linestyle=':', linewidth=1)

plt.axvspan(2000, 2010, alpha=0.1, color='yellow', label='Область экстраполяции')

plt.xlabel('Год', fontsize=12)
plt.ylabel('Население', fontsize=12)
plt.title('Экстраполяция населения США (1910-2010)\nСравнение трех методов', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.legend(loc='upper left', fontsize=11)
plt.xlim(1910, 2010)


plt.text(2005, newton_values[-1], f'{int(newton_values[-1]):,}', 
    ha='center', va='bottom', color='blue', fontsize=9)
plt.text(2005, spline_values[-1], f'{int(spline_values[-1]):,}', 
    ha='center', va='top', color='red', fontsize=9)
plt.text(2005, lsq_values[-1], f'{int(lsq_values[-1]):,}', 
    ha='center', va='bottom', color='green', fontsize=9)

plt.tight_layout()
plt.show()

print("Экстраполяция населения на 2010 год:")
print(f"Метод Ньютона: {int(newton_values[-1]):,} чел.")
print(f"Кубический сплайн: {int(spline_values[-1]):,} чел.")
print(f"МНК (полином 3-й степени): {int(lsq_values[-1]):,} чел.")
print(f"\nДля сравнения - реальное население США в 2010 году: 308,745,538 чел.")