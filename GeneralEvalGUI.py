import tkinter as tk
from tkinter import ttk, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import importlib
import os
import numpy as np
import time
import inspect

# Definition des Verzeichnisses für die Algorithmen
ALGORITHMS_DIR = 'optimization_algos'

class EvaluationFactors:
    def __init__(self):
        self.run_time = 0
        self.accuracy = 0
        self.iterations = 0

def load_algorithm(module_name, class_name):
    """ Lädt die Algorithmusklasse aus dem angegebenen Modul. """
    module = importlib.import_module(module_name)
    algorithm_class = getattr(module, class_name)
    return algorithm_class()

def find_classes_in_module(module):
    """ Findet alle Klassen in einem Modul. """
    classes = []
    for name, obj in inspect.getmembers(module, inspect.isclass):
        if obj.__module__ == module.__name__:
            classes.append(name)
    return classes

def load_modules_and_find_classes(directory):
    """ Lädt alle Module und ihre Klassen aus dem angegebenen Verzeichnis. """
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
    """ Erstellt skalierbare Optimierungsprobleme. """
    initial_params = np.random.rand(num_params) * scale
    bounds = [(0, 10000) for _ in range(num_params)]
    print(initial_params)
    print(bounds)
    return initial_params, bounds

def evaluate_algorithm(algorithm, params, bounds=None):
    """ Bewertet einen Algorithmus auf einem gegebenen Problem. """
    factors = EvaluationFactors()
    start_time = time.time()

    def function(x):
        return abs(sum((x - 100) ** 5) - 1300) / 15 + 30
    result, x = algorithm.optimize(function,params)
    print(algorithm,result)
    factors.run_time = time.time() - start_time
    factors.accuracy = x
    try:
        factors.iterations = getattr(result, 'nfev', 0)
    except:
        factors.iterations=0
    return factors

class OptimizationComparisonGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Optimierungsvergleich GUI")
        self.root.geometry("1200x600")

        # Erstelle Widgets
        self.num_params_var = tk.IntVar(value=5)
        self.scale_var = tk.DoubleVar(value=10.0)

        # Anzahl der Parameter
        ttk.Label(root, text="Anzahl der Parameter:").pack(pady=5)
        ttk.Entry(root, textvariable=self.num_params_var).pack()

        # Skalierungsfaktor
        ttk.Label(root, text="Skalierungsfaktor:").pack(pady=5)
        ttk.Entry(root, textvariable=self.scale_var).pack()

        # Button zum Ausführen der Optimierung
        ttk.Button(root, text="Optimierung ausführen", command=self.run_optimization).pack(pady=10)

        # Platz für Diagramme
        self.figure = plt.Figure(figsize=(10, 8))
        self.canvas = FigureCanvasTkAgg(self.figure, root)
        self.canvas.get_tk_widget().pack(expand=True, fill='both')

    def run_optimization(self):
        num_params = self.num_params_var.get()
        scale = self.scale_var.get()

        if num_params <= 0 or scale <= 0:
            messagebox.showerror("Ungültige Eingabe", "Die Anzahl der Parameter und der Skalierungsfaktor müssen größer als 0 sein.")
            return

        # Lade Algorithmen
        algorithms = load_modules_and_find_classes(ALGORITHMS_DIR)
        module_class_pairs = []
        for module, classes in algorithms.items():
            for cls in classes:
                module_class_pairs.append(f"{module}.{cls}")

        # Erstelle skalierbare Probleme
        initial_params, bounds = create_scalable_problems(num_params, scale)

        results = []
        for module_class in reversed(module_class_pairs):
            module_name, class_name = module_class.rsplit('.', 1)
            algorithm = load_algorithm(module_name, class_name)

            # Führe Algorithmus aus und sammle Metriken
            factors = evaluate_algorithm(algorithm, initial_params, bounds)

            results.append((class_name, factors.run_time, factors.accuracy, factors.iterations))

        # Aktualisiere Diagramme
        self.update_plots(results)

    def update_plots(self, results):
        self.figure.clf()

        # Entpacke Ergebnisse
        class_names, run_times, accuracies, iterations = zip(*results)

        # Laufzeitdiagramm
        ax1 = self.figure.add_subplot(131)
        bars=ax1.bar(class_names, run_times)
        ax1.set_title("Laufzeit (s)")
        ax1.set_ylabel("Zeit in Sekunden")
        ax1.tick_params(axis='x', rotation=45)

        for bar in bars:
            # Höhe des Balkens (Wert)
            yval = bar.get_height()
            # Text über dem Balken hinzufügen
            ax1.text(
                bar.get_x() + bar.get_width() / 2,  # x-Position
                yval,  # y-Position
                f'{yval:.2f}',  # Formatierter Text
                ha='center',  # Horizontale Ausrichtung
                va='bottom'  # Vertikale Ausrichtung
            )
        # Genauigkeitsdiagramm
        ax2 = self.figure.add_subplot(132)
        bars=ax2.bar(class_names, accuracies)
        ax2.set_title("Genauigkeit")
        ax2.set_ylabel("Kostenwert")
        ax2.tick_params(axis='x', rotation=45)

        for bar in bars:
            # Höhe des Balkens (Wert)
            yval = bar.get_height()
            # Text über dem Balken hinzufügen
            ax2.text(
                bar.get_x() + bar.get_width() / 2,  # x-Position
                yval,  # y-Position
                f'{yval:.2f}',  # Formatierter Text
                ha='center',  # Horizontale Ausrichtung
                va='bottom'  # Vertikale Ausrichtung
            )

        # Iterationsdiagramm
        ax3 = self.figure.add_subplot(133)
        bars=ax3.bar(class_names, iterations)
        ax3.set_title("Iterationen")
        ax3.set_ylabel("Anzahl der Iterationen")
        ax3.tick_params(axis='x', rotation=45)

        for bar in bars:
            # Höhe des Balkens (Wert)
            yval = bar.get_height()
            # Text über dem Balken hinzufügen
            ax3.text(
                bar.get_x() + bar.get_width() / 2,  # x-Position
                yval,  # y-Position
                f'{yval:.2f}',  # Formatierter Text
                ha='center',  # Horizontale Ausrichtung
                va='bottom'  # Vertikale Ausrichtung
            )

        # Aktualisiere die Canvas
        self.figure.tight_layout()
        self.canvas.draw()

if __name__ == "__main__":
    root = tk.Tk()
    app = OptimizationComparisonGUI(root)
    root.mainloop()
