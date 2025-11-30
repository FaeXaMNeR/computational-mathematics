import numpy as np
import matplotlib.pyplot as plt

def f(x):
    return np.log(x+3)

def df_analytical(x):
    return 1 / (x + 3)

def numerical_derivative1(x, h):
    return (f(x + h) - f(x)) / h

def numerical_derivative2(x, h):
    return (f(x) - f(x - h)) / h

def numerical_derivative3(x, h):
    return (f(x + h) - f(x - h)) / (2 * h)

def numerical_derivative4(x, h):
    return (4/3) * (f(x + h) - f(x - h)) / (2 * h) - (1/3) * (f(x + 2*h) - f(x - 2*h)) / (4 * h)

def numerical_derivative5(x, h):
    term1 = (3/2) * (f(x + h) - f(x - h)) / (2 * h)
    term2 = (3/5) * (f(x + 2*h) - f(x - 2*h)) / (4 * h)
    term3 = (1/10) * (f(x + 3*h) - f(x - 3*h)) / (6 * h)
    return term1 - term2 + term3

x0 = 2.0  
n_values = np.arange(1, 22) 
h_values = 2.0 / (2.0**n_values)  

true_derivative = df_analytical(x0)

errors1, errors2, errors3, errors4, errors5 = [], [], [], [], []

for h in h_values:
    num_deriv1 = numerical_derivative1(x0, h)
    errors1.append(abs(num_deriv1 - true_derivative))
    
    num_deriv2 = numerical_derivative2(x0, h)
    errors2.append(abs(num_deriv2 - true_derivative))
    
    num_deriv3 = numerical_derivative3(x0, h)
    errors3.append(abs(num_deriv3 - true_derivative))
    
    num_deriv4 = numerical_derivative4(x0, h)
    errors4.append(abs(num_deriv4 - true_derivative))
    
    num_deriv5 = numerical_derivative5(x0, h)
    errors5.append(abs(num_deriv5 - true_derivative))

plt.figure(figsize=(12, 8))
plt.loglog(h_values, errors1, 'bo-', linewidth=2, markersize=4, label='Формула 1')
plt.loglog(h_values, errors2, 'ro-', linewidth=2, markersize=4, label='Формула 2')
plt.loglog(h_values, errors3, 'go-', linewidth=2, markersize=4, label='Формула 3')
plt.loglog(h_values, errors4, 'mo-', linewidth=2, markersize=4, label='Формула 4')
plt.loglog(h_values, errors5, 'co-', linewidth=2, markersize=4, label='Формула 5')

plt.grid(True, which="both", ls="-", alpha=0.3)
plt.xlabel('Шаг дифференцирования h', fontsize=12)
plt.ylabel('Абсолютная погрешность', fontsize=12)
plt.title('Сравнение погрешностей различных формул численного дифференцирования\nдля функции y = ln(x+3)', fontsize=12)

plt.text(0.02, 0.02, f'Точка вычисления: x = {x0}', 
         transform=plt.gca().transAxes, fontsize=10,
         bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))

plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()