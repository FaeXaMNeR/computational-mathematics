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

# 5. Метод верхней релаксации
def solve(A, b):
    n = len(b)
    x = np.zeros(n)

    #A = L + D + U
    #x_k+1 = (D+wL)^-1 * (wb - (wU+(w-1)D)*x_k)
    
    L = np.tril(A, k=-1)
    D = np.diag(np.diag(A))
    U = np.triu(A, k=1)
    w = 1.5

    invLD = np.linalg.inv(D + w*L)

    nevaz = [] # невязка
    iterations = []
    
    for i in range(100):
        x = invLD @ (w*b - (w*U + (w-1)*D) @ x)
        nevaz.append(np.linalg.norm(b - A @ x))
        iterations.append(i)

    return x, iterations, nevaz



A, b = create_system()
x, iterations, nevaz = solve(A, b)

plt.figure(figsize=(10, 6))
plt.semilogy(iterations, nevaz, 'b-', linewidth=2, label='Невязка')
plt.xlabel('Номер итерации')
plt.ylabel('Невязка (log scale)')
plt.title('Зависимость невязки от числа итераций для метода верхней релаксации')
plt.grid(True, which='both', linestyle='--', alpha=0.7)
plt.legend()

plt.legend()
plt.tight_layout()
plt.show()

print(x)