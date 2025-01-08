"""
# Одновимірне рівняння теплопровідності (анімована версія)

## Постановка задачі
Даний скрипт розв'язує одновимірне рівняння теплопровідності:
∂u/∂t = a * ∂²u/∂x²

де:
- u(x,t) - температура в точці x в момент часу t
- a - коефіцієнт температуропровідності
- t - час
- x - просторова координата
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate
from matplotlib.animation import FuncAnimation
plt.style.use('default')

# Визначаємо основні фізичні параметри системи
a = 1.0  # Коефіцієнт температуропровідності
dx = 1.0  # Крок просторової сітки

def f_1D(t, u):
    """
    Функція, що реалізує праву частину рівняння теплопровідності.
    """
    unew = np.zeros(len(u))
    unew[1:-1] = u[2:] - 2*u[1:-1] + u[:-2]
    return unew * a/dx**2

# Параметри для обох випадків
size = 100      # Кількість точок у просторі
n_frames = 100  # Кількість кадрів для анімації
center_point = size // 2  # Центральна точка простору

# Випадок 1: Нагрів з одного кінця
tStart1, tEnd1 = 0, 5000
t_eval1 = np.linspace(tStart1, tEnd1, n_frames)
u0_case1 = np.zeros([size])
u0_case1[0] = 1

solution_case1 = integrate.solve_ivp(
    f_1D, [tStart1, tEnd1], u0_case1,
    method='RK45', t_eval=t_eval1
)

# Випадок 2: Нагрів з обох кінців
tStart2, tEnd2 = 0, 2000
t_eval2 = np.linspace(tStart2, tEnd2, n_frames)
u0_case2 = np.zeros([size])
u0_case2[0] = u0_case2[-1] = 1

solution_case2 = integrate.solve_ivp(
    f_1D, [tStart2, tEnd2], u0_case2,
    method='RK45', t_eval=t_eval2
)

# Створюємо фігури для анімації
fig1, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))
fig1.suptitle('Еволюція температури (нагрів з одного кінця)')

# Налаштування для першого випадку
line1, = ax1.plot([], [], 'b-', lw=2)
ax1.set_xlim(0, tEnd1)
ax1.set_ylim(-0.1, 0.5)  # Змінено максимум до 0.5
ax1.set_xlabel('Час')
ax1.set_ylabel('Температура в точці #50')
ax1.grid(True)

# Створюємо теплову карту
img1 = ax2.imshow(np.zeros((size, n_frames)), 
                  aspect='auto',
                  extent=[0, tEnd1, 0, size-1],  # Змінено порядок координат
                  cmap='viridis',
                  vmin=0, vmax=0.5,  # Змінено максимум до 0.5
                  interpolation='bilinear',
                  origin='upper')  # Додано параметр origin
plt.colorbar(img1, ax=ax2, label='Температура')
ax2.set_ylabel('Координата')
ax2.set_xlabel('Час')

# Створюємо другу фігуру для другого випадку
fig2, (ax3, ax4) = plt.subplots(2, 1, figsize=(10, 12))
fig2.suptitle('Еволюція температури (нагрів з обох кінців)')

line2, = ax3.plot([], [], 'b-', lw=2)
ax3.set_xlim(0, tEnd2)
ax3.set_ylim(-0.1, 0.9)  # Змінено максимум до 0.9
ax3.set_xlabel('Час')
ax3.set_ylabel('Температура в точці #50')
ax3.grid(True)

# Створюємо теплову карту
img2 = ax4.imshow(np.zeros((size, n_frames)),
                  aspect='auto',
                  extent=[0, tEnd2, 0, size-1],
                  cmap='viridis',
                  vmin=0, vmax=0.9,  # Також змінено максимум до 0.9
                  interpolation='bilinear',
                  origin='upper')
plt.colorbar(img2, ax=ax4, label='Температура')
ax4.set_ylabel('Координата')
ax4.set_xlabel('Час')

def init1():
    line1.set_data([], [])
    img1.set_array(np.zeros((size, n_frames)))
    return line1, img1

def animate1(frame):
    # Оновлюємо графік температури в центральній точці
    times = t_eval1[:frame+1]
    temps = solution_case1.y[center_point, :frame+1]
    line1.set_data(times, temps)
    
    # Оновлюємо теплову карту
    data = solution_case1.y[::-1, :frame+1]  # Перевертаємо дані
    padded_data = np.pad(data, ((0,0), (0, n_frames-frame-1)), mode='edge')
    img1.set_array(padded_data)
    
    return line1, img1

def init2():
    line2.set_data([], [])
    img2.set_array(np.zeros((size, n_frames)))
    return line2, img2

def animate2(frame):
    # Оновлюємо графік температури в центральній точці
    times = t_eval2[:frame+1]
    temps = solution_case2.y[center_point, :frame+1]
    line2.set_data(times, temps)
    
    # Оновлюємо теплову карту
    data = solution_case2.y[::-1, :frame+1]  # Перевертаємо дані
    padded_data = np.pad(data, ((0,0), (0, n_frames-frame-1)), mode='edge')
    img2.set_array(padded_data)
    
    return line2, img2

# Створюємо анімації
anim1 = FuncAnimation(fig1, animate1, init_func=init1, frames=n_frames,
                     interval=50, blit=True)
anim2 = FuncAnimation(fig2, animate2, init_func=init2, frames=n_frames,
                     interval=50, blit=True)

plt.show()
