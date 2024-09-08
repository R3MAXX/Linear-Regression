import tkinter as tk
import sv_ttk
from tkinter import ttk, messagebox
from tkinterdnd2 import Tk, DND_FILES
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

class MyLinearRegression:
    def __init__(self):
        self.coef_ = None
        self.intercept_ = None

    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y)

        X = np.hstack((np.ones((X.shape[0], 1)), X))

        k = np.linalg.inv(X.T @ X) @ X.T @ y

        self.coef_ = k[1:]
        self.intercept_ = k[0]

    def predict(self, X):
        X = np.array(X)
        y_pred = np.dot(X, self.coef_) + self.intercept_
        return y_pred

    # Метод для оценки качества модели.
    # Используется для сравнения полученного коэффициента детерминации (R²)
    # с реальным, подсчитанным не в программе, чтобы убедиться в корректности
    # построения модели линейной регрессии.
    def score(self, X, y):
        y_pred = self.predict(X)
        y_mean = np.mean(y)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - y_mean) ** 2)
        r2 = 1 - (ss_res / ss_tot)
        print("Коэффициент детерминации: ", r2)


# Функция для построения графика
def plot_graph(X, y, my_lr):
    fig, ax = plt.subplots()
    ax.scatter(X, y, color='blue', s=10)
    x = np.array([np.min(X), np.max(X)])
    y_pred = my_lr.predict(x.reshape(-1, 1))
    ax.plot(x, y_pred, color='green', label='Линейная регрессия')

    ax.set_xlabel('Коэффициент износа')
    ax.set_ylabel('Время устранения')
    ax.legend()

    return fig


# Функция для обработки перетаскивания файла
def drop_file(event):
    global file_path
    file_path = event.data
    status_label.config(text="Файл выбран")


# Функция для обработки кнопки "Построить график"
def build_graph():
    global file_path
    try:
        if file_path:
            analyze_data(file_path)
    except Exception:
        messagebox.showerror("Ошибка", "Выберите файл для анализа")


# Функция для очистки графика
def clear_graph():
    global fig
    for widget in plot_frame.winfo_children():
        widget.destroy()
    fig = None
    status_label.config(text="График очищен")
    mean_time_label.config(text="")


# Функция для анализа данных и построения модели
def analyze_data(file_path):
    global X, y, my_lr, fig
    try:
        data = pd.read_csv(file_path, encoding='cp1251', sep=';')
        data[['Коэффициент']] = data[['Коэффициент']].apply(lambda x: x.str.replace(',', '.'))
        X = data[['Коэффициент']].astype(float)
        data['Время'] = data['Время'].str.replace(',', '.')
        y = data['Время'].astype(float)

        my_lr = MyLinearRegression()
        my_lr.fit(X, y)

        mean_time = y.mean()
        mean_time_label.config(text=f"Среднее время устранения: {mean_time:.2f}")

        fig = plot_graph(X, y, my_lr)
        draw_figure(fig)
    except Exception as e:
        status_label.config(text=f"Ошибка: {str(e)}")


# Функция для отрисовки графика в Tkinter
def draw_figure(fig):
    for widget in plot_frame.winfo_children():
        widget.destroy()
    canvas = FigureCanvasTkAgg(fig, master=plot_frame)
    canvas.draw()
    canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)


# Функция для сохранения графика
def save_graph():
    global fig
    if fig is not None:
        file_path = tk.filedialog.asksaveasfilename(defaultextension=".png",
                                                    filetypes=[("PNG", "*.png"), ("JPEG", "*.jpg")])
        if file_path:
            fig.savefig(file_path)
            status_label.config(text="График сохранен")
    else:
        messagebox.showerror("Ошибка", "Нет графика для сохранения")


root = Tk()
root.title("Анализ данных линейной регрессии")

root.geometry("800x600")

sv_ttk.set_theme("light")

root.drop_target_register(DND_FILES)

top_frame = ttk.Frame(root)
top_frame.pack(side=tk.TOP, pady=10)

status_label = ttk.Label(top_frame, text="Файл не выбран", relief="solid", borderwidth=1, padding=5)
status_label.pack(side=tk.LEFT, padx=10)

plot_button = ttk.Button(root, text="Построить график", command=build_graph)
plot_button.pack(side=tk.TOP, pady=10)

mean_time_label = ttk.Label(root)
mean_time_label.pack(side=tk.TOP, pady=10)

button_frame = ttk.Frame(root)
button_frame.pack(side=tk.BOTTOM, pady=10)

clear_button = ttk.Button(button_frame, text="Очистить график", command=clear_graph)
clear_button.pack(side=tk.LEFT, padx=5)

save_button = ttk.Button(button_frame, text="Сохранить график", command=save_graph)
save_button.pack(side=tk.RIGHT, padx=5)

drop_area = ttk.Label(root, text="Перетащите файл в формате CSV", relief="groove", borderwidth=2, width=30, padding=10)
drop_area.pack(side=tk.TOP, pady=20)

drop_area.drop_target_register(DND_FILES)
drop_area.dnd_bind('<<Drop>>', drop_file)

plot_frame = ttk.Frame(root)
plot_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

root.mainloop()