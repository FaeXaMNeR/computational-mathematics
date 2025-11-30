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

# 9. Стабилизированный метод бисопряженных градиентов
def solve(A, b):
    n = len(b)
    x = np.zeros(n)

    nevaz = [] # невязка
    iterations = []
    
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

        nevaz.append(np.linalg.norm(b - A @ x))
        iterations.append(i)

    return x, iterations, nevaz



A, b = create_system()
x, iterations, nevaz = solve(A, b)

plt.figure(figsize=(10, 6))
plt.semilogy(iterations, nevaz, 'b-', linewidth=2, label='Невязка')
plt.xlabel('Номер итерации')
plt.ylabel('Невязка (log scale)')
plt.title('Зависимость невязки от числа итераций для стабилизированного метода бисопряженных градиентов')
plt.grid(True, which='both', linestyle='--', alpha=0.7)
plt.legend()

plt.legend()
plt.tight_layout()
plt.show()

print(x)