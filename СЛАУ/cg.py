import numpy as np
import matplotlib.pyplot as plt

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

# 8. Метод сопряженных градиентов
def solve(A, b):
    n = len(b)
    x = np.zeros(n)

    # Условия применимости: симметричная матрица

    #x_k+1 = x_k + a_k * p_k
    #a_k = (r_k, r_k) / (Ap_k, p_k)
    #r_k+1 = r_k - a_k * A * p_k
    #p_k+1 = r_k+1 + b_k+1 * p_k
    #b_k+1 = (r_k+1, r_k+1) / (r_k, r_k)

    nevaz = [] # невязка
    iterations = []
    
    r = b - A @ x
    p = r

    for i in range(100):
        rk = np.dot(r, r)
        a = rk / np.dot(A @ p, p)
        x = x + a * p
        
        r = r - a * A @ p
        beta = np.dot(r, r) / rk
        p = r + beta * p
        nevaz.append(np.linalg.norm(b - A @ x))
        iterations.append(i)

    return x, iterations, nevaz



A, b = create_system()
newA = A.T @ A
newb = A.T @ b
x, iterations, nevaz = solve(newA, newb)

plt.figure(figsize=(10, 6))
plt.semilogy(iterations, nevaz, 'b-', linewidth=2, label='Невязка')
plt.xlabel('Номер итерации')
plt.ylabel('Невязка (log scale)')
plt.title('Зависимость невязки от числа итераций для метода сопряженных градиентов')
plt.grid(True, which='both', linestyle='--', alpha=0.7)
plt.legend()

plt.legend()
plt.tight_layout()
plt.show()

print(x)