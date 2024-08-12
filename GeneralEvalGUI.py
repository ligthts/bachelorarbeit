import tkinter as tk
from tkinter import ttk, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import importlib
import os
import numpy as np
import time
import inspect
import main
import math


ALGORITHMS_DIR = 'optimization_algos'

class EvaluationFactors:
    def __init__(self):
        self.run_time = 0
        self.accuracy = 0
        self.iterations = 0


def load_algorithm(module_name, class_name):
    module = importlib.import_module(module_name)
    algorithm_class = getattr(module, class_name)
    return algorithm_class()


def find_classes_in_module(module):
    classes = []
    for name, obj in inspect.getmembers(module, inspect.isclass):
        if obj.__module__ == module.__name__:
            classes.append(name)
    return classes


def load_modules_and_find_classes(directory):
    modules_and_classes = {}
    for filename in os.listdir(directory):
        if filename.endswith('.py') and not filename.startswith('__'):
            module_name = filename[:-3]
            module_path = directory.replace('/', '.') + '.' + module_name
            try:
                module = importlib.import_module(module_path)
                classes = find_classes_in_module(module)
                modules_and_classes[module_name] = classes
            except Exception as e:
                print(f"Fehler beim Importieren von {module_name}: {e}")
    return modules_and_classes


def create_scalable_problems(num_params, scale):
    initial_params = np.random.rand(num_params) * scale
    bounds = [(0, 10000) for _ in range(num_params)]
    return initial_params, bounds


def evaluate_algorithm(algorithm, params, bounds=None):
    factors = EvaluationFactors()
    start_time = time.time()

    def function(x):
        w = 0
        for f in x:
            r = abs(math.sin(f) * f)
            w += r
        return w

    result, x = algorithm.optimize(function, params)
    factors.run_time = time.time() - start_time
    factors.accuracy = x
    try:
        factors.iterations = getattr(result, 'nfev', 0)
    except:
        factors.iterations = 0
    return factors


def perform_automatic_tests(algorithms, scales, params_per_scale):
    results = {alg_name.rsplit('.', 1)[0]: {'run_time': [], 'accuracy': [], 'iterations': [], 'param_count': [], 'scales': []} for alg_name in algorithms}

    for scale in scales:
        for param_count in params_per_scale:
            initial_params, bounds = create_scalable_problems(param_count, scale)
            for module_class in algorithms:
                module_name, class_name = module_class.rsplit('.', 1)
                algorithm = load_algorithm(module_name, class_name)
                factors = evaluate_algorithm(algorithm, initial_params, bounds)
                results[module_name]['run_time'].append(factors.run_time)
                results[module_name]['accuracy'].append(factors.accuracy)
                results[module_name]['iterations'].append(factors.iterations)
                results[module_name]['param_count'].append(param_count)
                results[module_name]['scales'].append(scale)

    return results


class OptimizationComparisonGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Optimierungsvergleich GUI")
        self.root.geometry("1200x800")  # Geänderte Höhe für mehr Platz

        self.notebook = ttk.Notebook(root)
        self.notebook.pack(fill='both', expand=True)

        self.test_page = ttk.Frame(self.notebook)
        self.notebook.add(self.test_page, text='Test Page')
        self.create_test_page()

        self.auto_test_page = ttk.Frame(self.notebook)
        self.notebook.add(self.auto_test_page, text='Automatic Tests')
        self.create_auto_test_page()

    def create_test_page(self):
        self.num_params_var = tk.IntVar(value=5)
        self.scale_var = tk.DoubleVar(value=10.0)

        ttk.Label(self.test_page, text="Anzahl der Parameter:").pack(pady=5)
        ttk.Entry(self.test_page, textvariable=self.num_params_var).pack()

        ttk.Label(self.test_page, text="Skalierungsfaktor:").pack(pady=5)
        ttk.Entry(self.test_page, textvariable=self.scale_var).pack()

        ttk.Button(self.test_page, text="Optimierung ausführen", command=self.run_optimization).pack(pady=10)

        self.figure = plt.Figure(figsize=(10, 8))
        self.canvas = FigureCanvasTkAgg(self.figure, self.test_page)
        self.canvas.get_tk_widget().pack(expand=True, fill='both')

    def run_optimization(self):
        num_params = self.num_params_var.get()
        scale = self.scale_var.get()

        if num_params <= 0 or scale <= 0:
            messagebox.showerror("Ungültige Eingabe",
                                 "Die Anzahl der Parameter und der Skalierungsfaktor müssen größer als 0 sein.")
            return

        algorithms = load_modules_and_find_classes(ALGORITHMS_DIR)
        module_class_pairs = []
        for module, classes in algorithms.items():
            for cls in classes:
                module_class_pairs.append(f"{module}.{cls}")

        initial_params, bounds = create_scalable_problems(num_params, scale)

        results = []
        for module_class in reversed(module_class_pairs):
            module_name, class_name = module_class.rsplit('.', 1)
            algorithm = load_algorithm(module_name, class_name)

            factors = evaluate_algorithm(algorithm, initial_params, bounds)

            results.append((class_name, factors.run_time, factors.accuracy, factors.iterations))

        self.update_plots(results)

    def update_plots(self, results):
        self.figure.clf()

        class_names, run_times, accuracies, iterations = zip(*results)

        ax1 = self.figure.add_subplot(131)
        bars = ax1.bar(class_names, run_times)
        ax1.set_title("Laufzeit (s)")
        ax1.set_ylabel("Zeit in Sekunden")
        ax1.tick_params(axis='x', rotation=45)

        for bar in bars:
            yval = bar.get_height()
            ax1.text(
                bar.get_x() + bar.get_width() / 2,
                yval,
                f'{yval:.2f}',
                ha='center',
                va='bottom'
            )

        ax2 = self.figure.add_subplot(132)
        bars = ax2.bar(class_names, accuracies)
        ax2.set_title("Genauigkeit")
        ax2.set_ylabel("Kostenwert")
        ax2.tick_params(axis='x', rotation=45)

        for bar in bars:
            yval = bar.get_height()
            ax2.text(
                bar.get_x() + bar.get_width() / 2,
                yval,
                f'{yval:.2f}',
                ha='center',
                va='bottom'
            )

        ax3 = self.figure.add_subplot(133)
        bars = ax3.bar(class_names, iterations)
        ax3.set_title("Iterationen")
        ax3.set_ylabel("Anzahl der Iterationen")
        ax3.tick_params(axis='x', rotation=45)

        for bar in bars:
            yval = bar.get_height()
            ax3.text(
                bar.get_x() + bar.get_width() / 2,
                yval,
                f'{yval:.2f}',
                ha='center',
                va='bottom'
            )

        self.figure.tight_layout()
        self.canvas.draw()

    def create_auto_test_page(self):
        algorithms = load_modules_and_find_classes(ALGORITHMS_DIR)
        module_class_pairs = []
        for module, classes in algorithms.items():
            for cls in classes:
                module_class_pairs.append(f"{module}.{cls}")

        scales = [5, 50, 500]
        params_per_scale = [5, 50, 500]

        results = perform_automatic_tests(
            module_class_pairs, scales, params_per_scale)

        figure, axes = plt.subplots(3, 2, figsize=(12, 10))  # Kleinere Graphen

        def plot_metric(metric, ax, x_data_key):
            for alg_name, data in results.items():
                ax.plot(data[x_data_key], data[metric], marker='o', label=alg_name)

            ax.set_title(f'{metric.capitalize()} vs. Anzahl der Parameter' if x_data_key == 'param_count' else f'{metric.capitalize()} vs. Skala')
            ax.set_xlabel('Anzahl der Parameter' if x_data_key == 'param_count' else 'Skala')
            ax.set_ylabel(metric.capitalize())
            ax.set_xscale('log')
            ax.legend()

            # Filter outliers using IQR
            q75, q25 = np.percentile(data[metric], [75, 25])
            iqr = q75 - q25
            lower_bound = max(q25 - 1.5 * iqr, 0)
            upper_bound = q75 + 1.5 * iqr
            ax.set_ylim(lower_bound, upper_bound)

        for idx, metric in enumerate(['run_time', 'accuracy', 'iterations']):
            plot_metric(metric, axes[idx, 0], 'param_count')
            plot_metric(metric, axes[idx, 1], 'scales')

        figure.tight_layout()

        canvas = FigureCanvasTkAgg(figure, master=self.auto_test_page)
        canvas.draw()
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)


if __name__ == "__main__":
    root = tk.Tk()
    app = OptimizationComparisonGUI(root)
    root.mainloop()
