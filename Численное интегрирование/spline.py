import numpy as np

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