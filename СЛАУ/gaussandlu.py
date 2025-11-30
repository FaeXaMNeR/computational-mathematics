import numpy as np

def create_system():    
    n = 100
    A = np.zeros((n, n))
    b = np.zeros(n)
    
    A[0, :] = 1
    b[0] = 100
    
    for i in range(1, n-1):
        A[i, i - 1] = 1
        A[i, i] = 10
        A[i, i + 1] = 1
        b[i] = 101 - i
    
    A[n - 1, n - 2] = 1
    A[n - 1, n - 1] = 1
    b[n - 1] = 1
    
    return A, b

def gauss(A, b):
    n = len(b)
    Ab = np.hstack([A.astype(float), b.reshape(-1, 1).astype(float)])
    
    for i in range(n):
        max_row = np.argmax(np.abs(Ab[i:, i])) + i
        Ab[[i, max_row]] = Ab[[max_row, i]]
    
        # Прямой ход
        for j in range(i + 1, n):
            factor = Ab[j, i] / Ab[i, i]
            Ab[j, i:] -= factor * Ab[i, i:]
    
    # Обратный ход
    x = np.zeros(n)
    for i in range(n-1, -1, -1):
        x[i] = (Ab[i, -1] - np.dot(Ab[i, i+1:n], x[i+1:n])) / Ab[i, i]
    
    return x

def lu(A, b):
    n = len(b)
    L = np.eye(n)
    U = A.astype(float).copy()
    
    for i in range(n):
        for j in range(i + 1, n):
            L[j, i] = U[j, i] / U[i, i]
            U[j, i:] -= L[j, i] * U[i, i:]
    
    # Решение Ly = b
    y = np.zeros(n)
    for i in range(n):
        y[i] = b[i] - np.dot(L[i, :i], y[:i])
    
    # Решение Ux = y
    x = np.zeros(n)
    for i in range(n-1, -1, -1):
        x[i] = (y[i] - np.dot(U[i, i+1:], x[i+1:])) / U[i, i]
    
    return x






A, b = create_system()

# Метод Гаусса
x_gauss = gauss(A, b)

# LU-разложение
x_lu = lu(A, b)

# Эталонное решение
x_numpy = np.linalg.solve(A, b)

print(x_gauss)
print(x_lu)
print(x_numpy)