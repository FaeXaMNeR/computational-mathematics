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

# 7. Метод минимальных невязок
def solve(A, b):
    n = len(b)
    x = np.zeros(n)

    #x_k+1 = x_k - t_k * r_k
    #t_k = (Ar_k, r_k) / (Ar_k, Ar_k)
    #r_k = Ax_k - b

    nevaz = [] # невязка
    iterations = []
    
    for i in range(100):
        r = A @ x - b
        if np.dot(A @ r, A @ r) != 0:
            t = np.dot(A @ r, r) / np.dot(A @ r, A @ r)
        else:
            continue
        x = x - t * r
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
plt.title('Зависимость невязки от числа итераций для метода минимальных невязок')
plt.grid(True, which='both', linestyle='--', alpha=0.7)
plt.legend()

plt.legend()
plt.tight_layout()
plt.show()

print(x)