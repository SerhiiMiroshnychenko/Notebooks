# Імпортуємо необхідні бібліотеки
import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate

# Визначаємо основні фізичні параметри системи
a = 1.5e-6  # Коефіцієнт температуропровідності залізнорудного агломерату, м²/с
dx = 0.01  # Крок сітки по x, м (10 мм)
dy = 0.01  # Крок сітки по y, м (10 мм)


def check_temperature(u, target_temp, tolerance=1.0):
    """
    Перевіряє чи досягнута цільова температура у всіх точках шару
    з заданою точністю
    """
    u_reshaped = u.reshape(sizey, sizex)
    return np.all(np.abs(u_reshaped - target_temp) <= tolerance)


def format_time(total_seconds):
    """
    Форматує час із секунд у вигляд "x секунд (y годин)"
    """
    hours = total_seconds / 3600  # Переводимо в години
    return f"{total_seconds:.1f} секунд ({hours:.2f} годин)"


def f_2D_flattened(t, u):
    """
    Допоміжна функція для перетворення двовимірної задачі в одновимірну.
    Розраховує зміну температури в кожній точці шару.
    """
    # Перетворюємо одновимірний масив назад у двовимірний
    u = u.reshape(sizey, sizex)

    # Створюємо масив для похідних
    unew = np.zeros([sizey, sizex])

    # Розраховуємо похідні для всіх внутрішніх точок
    unew[1:-1, 1:-1] = (u[2:, 1:-1] - 2 * u[1:-1, 1:-1] + u[:-2, 1:-1]) * a / dx ** 2 + \
                       (u[1:-1, 2:] - 2 * u[1:-1, 1:-1] + u[1:-1, :-2]) * a / dy ** 2

    # Повертаємо розгорнутий одновимірний масив
    return unew.flatten()


# Визначаємо розміри розрахункової сітки
sizex = 250  # Кількість точок по x (2500 мм / 10 мм)
sizey = 40  # Кількість точок по y (400 мм / 10 мм)

# Задаємо параметри часової еволюції
tStart = 0  # Початковий час, с
time_step = 400  # Крок по часу для перевірки температури, с
max_time = 500000  # Максимальний час розрахунку, с (≈5.8 діб)

# Граничні умови
T_ambient = 250  # Температура навколишнього середовища, °C
T_initial = 400  # Початкова температура шихти після агломерації, °C

# Створення масиву температур та встановлення початкових умов
T = np.zeros([sizey, sizex])
T.fill(T_initial)  # Заповнюємо всі точки початковою температурою

# Встановлюємо граничні умови (температура навколишнього середовища на всіх границях)
T[0, :] = T_ambient  # Нижня границя
T[-1, :] = T_ambient  # Верхня границя
T[:, 0] = T_ambient  # Ліва границя
T[:, -1] = T_ambient  # Права границя

print("\nРозрахунок охолодження шару залізнорудної агломераційної шихти...")
print(f"Розмір шару: {sizex * dx * 1000:.0f} x {sizey * dy * 1000:.0f} мм")
print(f"Кількість точок сітки: {sizex} x {sizey}")
print(f"Початкова температура: {T_initial}°C")
print(f"Температура навколишнього середовища: {T_ambient}°C")

# Ініціалізуємо змінні для зберігання проміжних результатів
current_time = tStart  # Поточний час, с
solutions = []
times = []

while current_time < max_time:
    # Розв'язуємо систему рівнянь на короткому проміжку часу
    solution = integrate.solve_ivp(
        f_2D_flattened,
        [current_time, current_time + time_step],
        T.flatten() if current_time == tStart else solutions[-1].y[:, -1],
        method='RK45',
        vectorized=True
    )

    solutions.append(solution)
    times.extend(solution.t)

    # Перевіряємо чи досягнута цільова температура
    current_temp = solution.y[:, -1].reshape(sizey, sizex)
    if check_temperature(solution.y[:, -1], T_ambient, tolerance=1.0):
        print(f"\nШар охолонув до температури {T_ambient}±1°C за {format_time(current_time)}")
        break

    current_time += time_step

    # Виводимо інформацію про прогрес охолодження
    if current_time % 3600 == 0:
        max_temp = np.max(current_temp)
        min_temp = np.min(current_temp)
        avg_temp = np.mean(current_temp)
        print(f"Час: {format_time(current_time)}, "
              f"температура: мін = {min_temp:.1f}°C, "
              f"середня = {avg_temp:.1f}°C, "
              f"макс = {max_temp:.1f}°C")

# Об'єднуємо всі розв'язки для візуалізації
all_times = np.array(times)
all_solutions = np.hstack([sol.y for sol in solutions])

# Створюємо сітку для візуалізації
x_list, y_list = np.meshgrid(np.arange(sizex) * dx, np.arange(sizey) * dy)

# Візуалізуємо результати для різних моментів часу
viz_indices = [0,  # Початок
               len(all_times) // 3,  # 1/3 часу
               2 * len(all_times) // 3,  # 2/3 часу
               -1]  # Кінець

# Створюємо окремий графік для кожного моменту часу
titles = ['Початковий розподіл температури',
          'Розподіл температури через 1/3 часу охолодження',
          'Розподіл температури через 2/3 часу охолодження',
          'Кінцевий розподіл температури']

for idx, plot_idx in enumerate(viz_indices):
    plt.figure(figsize=(6, 4))
    plt.title(f'{titles[idx]}\nt = {format_time(all_times[plot_idx])}',
              pad=10, fontsize=8)
    plt.xlabel('x, м', labelpad=5, fontsize=8)
    plt.ylabel('y, м', labelpad=5, fontsize=8)
    plt.xticks(fontsize=7)
    plt.yticks(fontsize=7)

    # Встановлюємо фіксований діапазон температур для кожного графіка
    if idx == 0:  # Перший графік (початковий стан)
        vmin, vmax = T_ambient, T_initial
    else:  # Інші графіки
        vmin, vmax = T_ambient, max(np.max(all_solutions[:, plot_idx]), T_ambient + 1)

    temp = plt.contourf(x_list, y_list,
                        all_solutions[:, plot_idx].reshape(sizey, sizex),
                        levels=np.linspace(vmin, vmax, T_ambient),
                        vmin=vmin, vmax=vmax)
    cbar = plt.colorbar(temp, label='Температура, °C', pad=0.02)
    cbar.ax.tick_labels = [f'{x:.0f}' for x in cbar.get_ticks()]
    cbar.ax.tick_params(labelsize=7)
    cbar.ax.set_ylabel('Температура, °C', fontsize=8, labelpad=5)

    # Встановлюємо однакові пропорції для осей
    plt.gca().set_aspect('equal', adjustable='box')

    # Додаємо відступи
    plt.tight_layout()
    plt.show()