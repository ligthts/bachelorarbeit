import tkinter as tk
from tkinter import ttk, messagebox
import optimization_algos
from main import load_modules_and_find_classes, load_algorithm, evaluate_algorithm
from Dataadder import DataManager
from empirical import Empiricalschatzung
import matplotlib.pyplot as plt
import numpy as np
import json


class App(tk.Tk):
    def __init__(self, base_dir, specialization):
        super().__init__()
        self.title("Optimization Algorithm Evaluation")
        self.geometry("800x600")

        self.base_dir = base_dir
        self.specialization = specialization

        self.data_manager = DataManager(self.base_dir)
        self.data = self.data_manager.load_data(self.specialization)
        self.algorithms = load_modules_and_find_classes('optimization_algos')
        print(self.algorithms)
        self.create_widgets()

    def create_widgets(self):
        self.tabs = ttk.Notebook(self)
        self.tabs.pack(expand=1, fill="both")

        self.algorithm_tabs = {}

        for module, classes in self.algorithms.items():
            for cls in classes:
                tab = ttk.Frame(self.tabs)
                self.tabs.add(tab, text=f"{module}.{cls}")
                self.algorithm_tabs[f"{module}.{cls}"] = tab
                self.evaluate_and_display_results(tab, module, cls)

        overall_avg_button = ttk.Button(self, text="Show Overall Averages", command=self.show_overall_averages)
        overall_avg_button.pack(pady=10)

    def evaluate_and_display_results(self, tab, module_name, class_name):
        algorithm = load_algorithm(module_name, class_name)
        results = []

        for data in self.data:
            try:
                factors, result = evaluate_algorithm(algorithm, data)
                results.append((factors, result))
                self.plot_result(data, algorithm, result)
            except:
                print(1)

        self.display_results(tab, results)

    def plot_result(self, data, algorithm, result):
        fig, ax = plt.subplots()
        x = np.array(data['x'])
        y = np.array(data['y'])
        model_y = model(result.x, x)

        ax.plot(x, y, 'o', label='Data')
        ax.plot(x, model_y, label='Fitted model')
        ax.legend()

        #Empiricalschatzung.Plotting(Empiricalschatzung, algorithm)

        fig.savefig(f"plot_{algorithm.__class__.__name__}.png")
        plt.close(fig)

    def display_results(self, tab, results):
        tree = ttk.Treeview(tab, columns=('Run Time', 'Accuracy', 'Memory Usage'), show='headings')
        tree.heading('Run Time', text='Run Time')
        tree.heading('Accuracy', text='Accuracy')
        tree.heading('Memory Usage', text='Memory Usage')
        tree.pack(expand=1, fill="both")

        avg_runtime = sum(result[0].run_time for result in results) / len(results)
        avg_accuracy = sum(result[0].accuracy for result in results) / len(results)
        avg_memory = sum(result[0].memory_usage for result in results) / len(results)

        for result in results:
            factors = result[0]
            tree.insert('', tk.END, values=(factors.run_time, factors.accuracy, factors.memory_usage))

        avg_label = ttk.Label(tab,
                              text=f"Average Run Time: {avg_runtime}\nAverage Accuracy: {avg_accuracy}\nAverage Memory Usage: {avg_memory}")
        avg_label.pack(pady=10)

    def show_overall_averages(self):
        total_runtime, total_accuracy, total_memory, count = 0, 0, 0, 0

        for module, classes in self.algorithms.items():
            for cls in classes:
                algorithm = load_algorithm(module, cls)
                results = []
                for data in self.data:
                    factors, result = evaluate_algorithm(algorithm, data)
                    results.append((factors, result))
                total_runtime += sum(result[0].run_time for result in results)
                total_accuracy += sum(result[0].accuracy for result in results)
                total_memory += sum(result[0].memory_usage for result in results)
                count += len(results)

        avg_runtime = total_runtime / count
        avg_accuracy = total_accuracy / count
        avg_memory = total_memory / count

        messagebox.showinfo("Overall Averages",
                            f"Average Run Time: {avg_runtime}\nAverage Accuracy: {avg_accuracy}\nAverage Memory Usage: {avg_memory}")


def model(params, x):
    f=0
    result=0
    for i in reversed(params):
        if f<len(x):
            result+=i*x**f
            f+=1
    return result


if __name__ == "__main__":
    app = App(base_dir="data", specialization="camera_calibration")
    app.mainloop()
