import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def f1(x, y):
    return x**2 + x*y - 10

def f2(x, y):
    return y + 3*x*y**2 - 57

def f1_x(x, y):
    return 2*x + y

def f1_y(x, y):
    return x

def f2_x(x, y):
    return 3*y**2

def f2_y(x, y):
    return 1 + 6*x*y

def jacobian(x, y):
    J = np.array([
        [f1_x(x, y), f1_y(x, y)],
        [f2_x(x, y), f2_y(x, y)]
    ])
    return J

def g1A_jacobi(x, y):
    """g1A: x = sqrt(10 - xy)"""
    return np.sqrt(max(10 - x*y, 0))

def g2B_jacobi(x, y):
    """g2B: y = sqrt((57 - y) / (3*x))"""
    if x != 0 and (57 - y) / (3*x) >= 0:
        return np.sqrt((57 - y) / (3*x))
    return y


def g1A_seidel(x, y):
    """g1A: x = sqrt(10 - xy)"""
    return np.sqrt(max(10 - x*y, 0))

def g2B_seidel(x_new, y):
    """g2B: y = sqrt((57 - y) / (3*x_new))"""
    if x_new != 0 and (57 - y) / (3*x_new) >= 0:
        return np.sqrt((57 - y) / (3*x_new))
    return y


def jacobi_method(x0, y0, epsilon, max_iter=100):
    x, y = x0, y0
    iterations = []
    
    for k in range(max_iter):
        x_new = g1A_jacobi(x, y)
        y_new = g2B_jacobi(x, y)
        
        iterations.append({
            'k': k,
            'x': x,
            'y': y,
            'f1': f1(x, y),
            'f2': f2(x, y),
            'x_new': x_new,
            'y_new': y_new,
            'error_x': abs(x_new - x),
            'error_y': abs(y_new - y),
            'max_error': max(abs(x_new - x), abs(y_new - y))
        })
        
        if max(abs(x_new - x), abs(y_new - y)) < epsilon:
            x, y = x_new, y_new
            iterations.append({
                'k': k+1,
                'x': x,
                'y': y,
                'f1': f1(x, y),
                'f2': f2(x, y),
                'x_new': x,
                'y_new': y,
                'error_x': 0,
                'error_y': 0,
                'max_error': 0
            })
            break
        
        x, y = x_new, y_new
    
    return x, y, iterations


def gauss_seidel_method(x0, y0, epsilon, max_iter=100):
    x, y = x0, y0
    iterations = []
    
    for k in range(max_iter):
        x_new = g1A_seidel(x, y)
        y_new = g2B_seidel(x_new, y)  
        
        iterations.append({
            'k': k,
            'x': x,
            'y': y,
            'f1': f1(x, y),
            'f2': f2(x, y),
            'x_new': x_new,
            'y_new': y_new,
            'error_x': abs(x_new - x),
            'error_y': abs(y_new - y),
            'max_error': max(abs(x_new - x), abs(y_new - y))
        })
        
        if max(abs(x_new - x), abs(y_new - y)) < epsilon:
            x, y = x_new, y_new
            iterations.append({
                'k': k+1,
                'x': x,
                'y': y,
                'f1': f1(x, y),
                'f2': f2(x, y),
                'x_new': x,
                'y_new': y,
                'error_x': 0,
                'error_y': 0,
                'max_error': 0
            })
            break
        
        x, y = x_new, y_new
    
    return x, y, iterations

def newton_raphson_method(x0, y0, epsilon, max_iter=100):
    x, y = x0, y0
    iterations = []
    
    for k in range(max_iter):
        F = np.array([f1(x, y), f2(x, y)])
        J = jacobian(x, y)
        
        try:
            delta = np.linalg.solve(J, -F)
            x_new = x + delta[0]
            y_new = y + delta[1]
        except np.linalg.LinAlgError:
            print("Jacobian singular pada iterasi ke-", k)
            break
        
        iterations.append({
            'k': k,
            'x': x,
            'y': y,
            'f1': f1(x, y),
            'f2': f2(x, y),
            'x_new': x_new,
            'y_new': y_new,
            'error_x': abs(x_new - x),
            'error_y': abs(y_new - y),
            'max_error': max(abs(x_new - x), abs(y_new - y))
        })
        
        if max(abs(x_new - x), abs(y_new - y)) < epsilon:
            x, y = x_new, y_new
            iterations.append({
                'k': k+1,
                'x': x,
                'y': y,
                'f1': f1(x, y),
                'f2': f2(x, y),
                'x_new': x,
                'y_new': y,
                'error_x': 0,
                'error_y': 0,
                'max_error': 0
            })
            break
        
        x, y = x_new, y_new
    
    return x, y, iterations

def secant_method(x0, y0, epsilon, max_iter=100, h=0.01):
    x_prev, y_prev = x0, y0
    x, y = x0 + h, y0 + h
    iterations = []
    
    for k in range(max_iter):
        F_curr = np.array([f1(x, y), f2(x, y)])
        F_prev = np.array([f1(x_prev, y_prev), f2(x_prev, y_prev)])
        
        delta_x = x - x_prev
        delta_y = y - y_prev
        
        if abs(delta_x) < 1e-10 or abs(delta_y) < 1e-10:
            break
        
        J_approx = np.array([
            [(f1(x + h, y) - f1(x, y)) / h, (f1(x, y + h) - f1(x, y)) / h],
            [(f2(x + h, y) - f2(x, y)) / h, (f2(x, y + h) - f2(x, y)) / h]
        ])
        
        try:
            delta = np.linalg.solve(J_approx, -F_curr)
            x_new = x + delta[0]
            y_new = y + delta[1]
        except np.linalg.LinAlgError:
            print("Jacobian singular pada iterasi ke-", k)
            break
        
        iterations.append({
            'k': k,
            'x': x,
            'y': y,
            'f1': f1(x, y),
            'f2': f2(x, y),
            'x_new': x_new,
            'y_new': y_new,
            'error_x': abs(x_new - x),
            'error_y': abs(y_new - y),
            'max_error': max(abs(x_new - x), abs(y_new - y))
        })
        
        if max(abs(x_new - x), abs(y_new - y)) < epsilon:
            x, y = x_new, y_new
            iterations.append({
                'k': k+1,
                'x': x,
                'y': y,
                'f1': f1(x, y),
                'f2': f2(x, y),
                'x_new': x,
                'y_new': y,
                'error_x': 0,
                'error_y': 0,
                'max_error': 0
            })
            break
        
        x_prev, y_prev = x, y
        x, y = x_new, y_new
    
    return x, y, iterations

x0 = 1.5
y0 = 3.5
epsilon = 0.000001

print("="*90)
print("SOLUSI SISTEM PERSAMAAN NONLINEAR")
print("="*90)
print(f"f₁(x, y) = x² + xy - 10 = 0")
print(f"f₂(x, y) = y + 3xy² - 57 = 0")
print(f"\nNilai awal: x₀ = {x0}, y₀ = {y0}")
print(f"Toleransi (ε) = {epsilon}")
print("="*90)

print("\n\n" + "="*90)
print("METODE 1: JACOBI (g1A dan g2B)")
print("="*90)
x_jacobi, y_jacobi, iter_jacobi = jacobi_method(x0, y0, epsilon)

df_jacobi = pd.DataFrame(iter_jacobi)
print("\nTabel Iterasi Jacobi:")
print(df_jacobi.to_string(index=False))

print(f"\n\nHasil Akhir Jacobi:")
print(f"x = {x_jacobi:.10f}")
print(f"y = {y_jacobi:.10f}")
print(f"f₁(x, y) = {f1(x_jacobi, y_jacobi):.10e}")
print(f"f₂(x, y) = {f2(x_jacobi, y_jacobi):.10e}")
print(f"Jumlah iterasi: {len(iter_jacobi) - 1}")

print("\n\n" + "="*90)
print("METODE 2: GAUSS-SEIDEL (g1A dan g2B)")
print("="*90)
x_seidel, y_seidel, iter_seidel = gauss_seidel_method(x0, y0, epsilon)

df_seidel = pd.DataFrame(iter_seidel)
print("\nTabel Iterasi Gauss-Seidel:")
print(df_seidel.to_string(index=False))

print(f"\n\nHasil Akhir Gauss-Seidel:")
print(f"x = {x_seidel:.10f}")
print(f"y = {y_seidel:.10f}")
print(f"f₁(x, y) = {f1(x_seidel, y_seidel):.10e}")
print(f"f₂(x, y) = {f2(x_seidel, y_seidel):.10e}")
print(f"Jumlah iterasi: {len(iter_seidel) - 1}")

print("\n\n" + "="*90)
print("METODE 3: NEWTON-RAPHSON")
print("="*90)
x_nr, y_nr, iter_nr = newton_raphson_method(x0, y0, epsilon)

df_nr = pd.DataFrame(iter_nr)
print("\nTabel Iterasi Newton-Raphson:")
print(df_nr.to_string(index=False))

print(f"\n\nHasil Akhir Newton-Raphson:")
print(f"x = {x_nr:.10f}")
print(f"y = {y_nr:.10f}")
print(f"f₁(x, y) = {f1(x_nr, y_nr):.10e}")
print(f"f₂(x, y) = {f2(x_nr, y_nr):.10e}")
print(f"Jumlah iterasi: {len(iter_nr) - 1}")

print("\n\n" + "="*90)
print("METODE 4: SECANT")
print("="*90)
x_sec, y_sec, iter_sec = secant_method(x0, y0, epsilon)

df_sec = pd.DataFrame(iter_sec)
print("\nTabel Iterasi Secant:")
print(df_sec.to_string(index=False))

print(f"\n\nHasil Akhir Secant:")
print(f"x = {x_sec:.10f}")
print(f"y = {y_sec:.10f}")
print(f"f₁(x, y) = {f1(x_sec, y_sec):.10e}")
print(f"f₂(x, y) = {f2(x_sec, y_sec):.10e}")
print(f"Jumlah iterasi: {len(iter_sec) - 1}")

print("\n\n" + "="*90)
print("PERBANDINGAN KEEMPAT METODE")
print("="*90)
print(f"{'Metode':<20} {'x':<20} {'y':<20} {'Iterasi':<15}")
print("-"*75)
print(f"{'Jacobi':<20} {x_jacobi:<20.10f} {y_jacobi:<20.10f} {len(iter_jacobi)-1:<15}")
print(f"{'Gauss-Seidel':<20} {x_seidel:<20.10f} {y_seidel:<20.10f} {len(iter_seidel)-1:<15}")
print(f"{'Newton-Raphson':<20} {x_nr:<20.10f} {y_nr:<20.10f} {len(iter_nr)-1:<15}")
print(f"{'Secant':<20} {x_sec:<20.10f} {y_sec:<20.10f} {len(iter_sec)-1:<15}")

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

ax1 = axes[0, 0]
if len(iter_jacobi) > 1:
    iter_nums_jacobi = [d['k'] for d in iter_jacobi[:-1]]
    errors_jacobi = [d['max_error'] for d in iter_jacobi[:-1]]
    ax1.semilogy(iter_nums_jacobi, errors_jacobi, 'o-', label='Jacobi', linewidth=2, markersize=6)

if len(iter_seidel) > 1:
    iter_nums_seidel = [d['k'] for d in iter_seidel[:-1]]
    errors_seidel = [d['max_error'] for d in iter_seidel[:-1]]
    ax1.semilogy(iter_nums_seidel, errors_seidel, 's-', label='Gauss-Seidel', linewidth=2, markersize=6)

if len(iter_nr) > 1:
    iter_nums_nr = [d['k'] for d in iter_nr[:-1]]
    errors_nr = [d['max_error'] for d in iter_nr[:-1]]
    ax1.semilogy(iter_nums_nr, errors_nr, '^-', label='Newton-Raphson', linewidth=2, markersize=6)

if len(iter_sec) > 1:
    iter_nums_sec = [d['k'] for d in iter_sec[:-1]]
    errors_sec = [d['max_error'] for d in iter_sec[:-1]]
    ax1.semilogy(iter_nums_sec, errors_sec, 'd-', label='Secant', linewidth=2, markersize=6)

ax1.axhline(y=epsilon, color='r', linestyle='--', label=f'ε = {epsilon}')
ax1.set_xlabel('Iterasi (k)', fontsize=11)
ax1.set_ylabel('Error Maksimal', fontsize=11)
ax1.set_title('Konvergensi Error - Semua Metode', fontsize=12, fontweight='bold')
ax1.grid(True, alpha=0.3)
ax1.legend()

ax2 = axes[0, 1]
x_jacobi_path = [d['x'] for d in iter_jacobi]
y_jacobi_path = [d['y'] for d in iter_jacobi]

x_seidel_path = [d['x'] for d in iter_seidel]
y_seidel_path = [d['y'] for d in iter_seidel]

ax2.plot(x_jacobi_path, y_jacobi_path, 'o-', label='Jacobi', linewidth=2, markersize=6)
ax2.plot(x_seidel_path, y_seidel_path, 's-', label='Gauss-Seidel', linewidth=2, markersize=6)
ax2.plot(x0, y0, 'g*', markersize=15, label='Start Point')
ax2.plot(x_jacobi, y_jacobi, 'r*', markersize=12)
ax2.plot(x_seidel, y_seidel, 'b*', markersize=12)
ax2.set_xlabel('x', fontsize=11)
ax2.set_ylabel('y', fontsize=11)
ax2.set_title('Lintasan Solusi - Jacobi & Gauss-Seidel', fontsize=12, fontweight='bold')
ax2.grid(True, alpha=0.3)
ax2.legend()

ax3 = axes[1, 0]
x_nr_path = [d['x'] for d in iter_nr]
y_nr_path = [d['y'] for d in iter_nr]

x_sec_path = [d['x'] for d in iter_sec]
y_sec_path = [d['y'] for d in iter_sec]

ax3.plot(x_nr_path, y_nr_path, '^-', label='Newton-Raphson', linewidth=2, markersize=6)
ax3.plot(x_sec_path, y_sec_path, 'd-', label='Secant', linewidth=2, markersize=6)
ax3.plot(x0, y0, 'g*', markersize=15, label='Start Point')
ax3.plot(x_nr, y_nr, 'r*', markersize=12)
ax3.plot(x_sec, y_sec, 'b*', markersize=12)
ax3.set_xlabel('x', fontsize=11)
ax3.set_ylabel('y', fontsize=11)
ax3.set_title('Lintasan Solusi - Newton-Raphson & Secant', fontsize=12, fontweight='bold')
ax3.grid(True, alpha=0.3)
ax3.legend()

ax4 = axes[1, 1]
methods = ['Jacobi', 'Gauss-Seidel', 'Newton-Raphson', 'Secant']
iterations = [len(iter_jacobi)-1, len(iter_seidel)-1, len(iter_nr)-1, len(iter_sec)-1]
colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A']

bars = ax4.bar(methods, iterations, color=colors, edgecolor='black', linewidth=2)
ax4.set_ylabel('Jumlah Iterasi', fontsize=11)
ax4.set_title('Perbandingan Jumlah Iterasi', fontsize=12, fontweight='bold')
ax4.grid(True, alpha=0.3, axis='y')

for bar, iter_count in zip(bars, iterations):
    height = bar.get_height()
    ax4.text(bar.get_x() + bar.get_width()/2., height,
            f'{int(iter_count)}',
            ha='center', va='bottom', fontsize=11, fontweight='bold')

plt.tight_layout()
plt.savefig('nonlinear_system_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n\nGrafik telah disimpan sebagai 'nonlinear_system_comparison.png'")