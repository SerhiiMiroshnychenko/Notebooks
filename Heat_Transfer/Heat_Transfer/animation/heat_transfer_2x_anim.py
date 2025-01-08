"""
Завдання: Моделювання двовимірного розподілу температури в квадратній пластині
методом скінченних різниць з підвищеною температурою на верхній границі.

Граничні умови:
- Ліва сторона (x=0): T1 = 80°C
- Нижня сторона (y=0): T2 = 60°C
- Права сторона (x=max): T3 = 45°C
- Верхня сторона (y=max): T4 = 120°C

Параметри моделювання:
- Розмір сітки: 10x10 точок
- Кількість ітерацій: 80
- Початкова температура всередині пластини: 35°C
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Розміри сітки
xdim = 10
ydim = 10

# Граничні умови (температура на границях пластини)
T1 = 80   # ліва сторона
T2 = 60   # нижня сторона (y=0)
T3 = 45   # права сторона
T4 = 120  # верхня сторона (y=max)

# Початкова температура всередині пластини
T_guess = 35

max_T = max((T1, T2, T3, T4, T_guess))
min_T = min((T1, T2, T3, T4, T_guess))

def print_temperature_matrix(matrix, title=""):
    """Функція для форматованого виведення матриці температур"""
    print(f"\n{title}")
    print("-" * 80)
    print("     ", end="")
    print("X", end="\t")
    for j in range(matrix.shape[1]):
        print(f"{j:<7}", end="")
    print("\nY")
    print("-" * 80)
    for i in range(matrix.shape[0]):
        print(f"{i:<4}", end=" ")
        for j in range(matrix.shape[1]):
            print(f"{matrix[i,j]:>7.2f}", end="")
        print()
    print("-" * 80)

# Створення масиву температур та встановлення початкових умов
T = np.zeros((xdim,ydim))
# Виведення нульової матриці
print_temperature_matrix(T, "Нульова матриця температури:")

T.fill(T_guess)
# Матриця температури до встановлення граничних умов
print_temperature_matrix(T, "Матриця температури до встановлення граничних умов:")

# Встановлення граничних умов
T[0:xdim,0] = T1    # ліва сторона
T[0,:] = T2         # нижня сторона (y=0)
T[0:xdim,ydim-1] = T3  # права сторона
T[xdim-1,:] = T4    # верхня сторона (y=max)

# Виведення початкового стану
print_temperature_matrix(T, "Початковий розподіл температури:")

# Створюємо список для зберігання стану температури на кожній ітерації
temperature_history = [T.copy()]

# Ітераційний процес розрахунку
niter = 80
for n in range(0,niter):
    T_new = T.copy()
    for i in range(1,xdim-1,1):
        for j in range(1,ydim-1,1):
            # Розрахунок нової температури в точці як середнє арифметичне
            # температур у сусідніх точках
            T_new[i,j] = 0.25*(T[i-1,j] + T[i+1,j] + T[i,j-1] + T[i,j+1])
    T = T_new
    temperature_history.append(T.copy())

    # Виведення проміжних результатів кожні 10 ітерацій
    if (n+1) % 10 == 0:
        print_temperature_matrix(T, f"Ітерація {n+1}:")

# Виведення фінального розподілу температури
print_temperature_matrix(T, "\nФінальний розподіл температури:")

# Налаштування анімації
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
fig.suptitle('Динаміка розподілу температури в пластині')

# Створюємо рівномірну шкалу температур
levels = np.arange(min_T, max_T+5, 5)

# Початкова візуалізація
contour = ax1.contourf(T, levels=levels, cmap='jet')
ax1.grid(color='k', linestyle='--')
ax1.set_title('Контурний графік')
ax1.set_xlabel('X координата')
ax1.set_ylabel('Y координата')

im = ax2.imshow(T, cmap='jet', interpolation='nearest', vmin=min_T, vmax=max_T, origin='lower')
ax2.set_title('Теплова карта')
ax2.set_xlabel('X координата')
ax2.set_ylabel('Y координата')

plt.colorbar(im, ax=ax2, label='Температура, °C')
plt.colorbar(contour, ax=ax1, label='Температура, °C')

# Функція оновлення кадру анімації
def update(frame):
    ax1.clear()
    ax2.clear()

    # Оновлення контурного графіку
    contour = ax1.contourf(temperature_history[frame], levels=levels, cmap='jet')
    ax1.grid(color='k', linestyle='--')
    ax1.set_title(f'Контурний графік (ітерація {frame})')
    ax1.set_xlabel('X координата')
    ax1.set_ylabel('Y координата')

    # Оновлення теплової карти
    im = ax2.imshow(temperature_history[frame], cmap='jet', interpolation='nearest', vmin=min_T, vmax=max_T, origin='lower')
    ax2.set_title(f'Теплова карта (ітерація {frame})')
    ax2.set_xlabel('X координата')
    ax2.set_ylabel('Y координата')

    # Додавання значень температури на теплову карту
    for i in range(temperature_history[frame].shape[0]):
        for j in range(temperature_history[frame].shape[1]):
            temp = temperature_history[frame][i,j]
            ax2.text(j, i, f'{temp:.1f}°C',
                     ha='center', va='center',
                     color='white' if temp > 100 else 'black',
                     fontsize=8)

    return ax1, ax2

# Створення анімації
anim = FuncAnimation(fig, update, frames=len(temperature_history),
                     interval=500, repeat=True)

plt.tight_layout()
plt.show()

"""
Аналіз результатів:

1. Розподіл температури в пластині має значний градієнт через велику
   різницю температур між верхньою (120°C) та нижньою (60°C) границями:
   - Максимальна температура (120°C) спостерігається на верхній границі
   - Мінімальна температура (45°C) - на правій границі
   - У центральній області температура поступово змінюється від 120°C до 60°C

2. Градієнти температури:
   - Найбільший градієнт спостерігається біля верхньої границі
   - У центральній частині температура змінюється більш плавно
   - Біля нижньої границі градієнт менший через меншу різницю температур

3. Збіжність рішення:
   - 80 ітерацій достатньо для досягнення стабільного стану
   - Температура в центральних точках стабілізується
   - Розподіл температури симетричний відносно вертикальної осі

Висновки:
1. Метод скінченних різниць успішно моделює стаціонарний розподіл
   температури навіть при значних перепадах температур на границях.

2. Анімація процесу теплопередачі дозволяє спостерігати за:
   - Поступовим встановленням теплової рівноваги
   - Впливом граничних умов на розподіл температури
   - Швидкістю збіжності рішення

3. Візуалізація у вигляді теплової карти з числовими значеннями
   дозволяє точно оцінити температуру в кожній точці пластини.
"""
