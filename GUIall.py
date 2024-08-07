import os
import time
import importlib
import inspect
import psutil
import shutil
import json
import uuid
import pandas as pd
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from empirical import Empiricalschatzung
from Dataadder import DataManager
import cv2

# Constants
ALGORITHMS_DIR = "optimization_algos"
PROBLEMS_DIR = "shape_estimation_results"
RESULTS_CSV = "results.csv"

class EvaluationFactors:
    def __init__(self):
        self.run_time = 0
        self.accuracy = 0
        self.memory_usage = 0

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

def run_all_evaluations(base_dir='camera_calibration_problems', algorithms=[]):
    """
    Läuft alle Bewertungen für jedes Kalibrierungsproblem und jeden Algorithmus.

    :param base_dir: Hauptverzeichnis, das die Kalibrierungsprobleme enthält.
    :param algorithms: Liste von Algorithmen, die zur Bewertung verwendet werden sollen.
    """
    # Lade alle Kalibrierungsprobleme
    module_class_pairs=[]
    python_functions, json_data = DataManager.load_all_calibration_problems(base_dir)
    for module, classes in algorithms.items():
        for cls in classes:
            module_class_pairs.append(f"{module}.{cls}")

    # Iteriere über alle geladenen Kalibrierungsprobleme
    for problem in python_functions:
        for data in json_data:
            print("hier sind die additional", data['additional_params'])
            for module_class in reversed(module_class_pairs):
                module_name, class_name = module_class.rsplit('.', 1)
                print(module_name, class_name)
                algorithm = load_algorithm(module_name, class_name)

                # Iteriere über alle Algorithmen
                print(algorithm)
                try:
                    # print(f"Evaluating problem '{problem['name']}' with algorithm '{algorithm.__name__}'")
                    factors, result_x, minvalue = evaluate_algorithm(algorithm, data, problem)
                    try:
                        print(result_x)
                        print(f"Problemart:{base_dir}")
                        print(f"Algorithm: {class_name}")
                        print(f"Run Time: {factors.run_time} seconds")
                        print(f"Accuracy: {factors.accuracy}")
                        print(f"Memory Usage: {factors.memory_usage} bytes")

                        try:
                            print("Anzahl Iterationen:", result_x.nfev)
                        except:
                            print("kann keine Iterationen bestimmen")
                        print(f"Minimum x: {result_x}, Minimum value: {minvalue}")
                        print(f"Result: {minvalue}")
                    except:
                        print("nicht möglich mit :", result_x)
                except ():
                    print("nicht möglich")

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

def evaluate_algorithm(algorithm, data, function):
    factors = EvaluationFactors()

    process = psutil.Process()
    mem_before = process.memory_info().rss
    start_time = time.time()
    print("hier2:", data['additional_params'])
    # Optimierung der Funktion auf den x-Wert, der das Minimum erreicht
    result_x, minvalue = algorithm.optimize(
        function,  # Pass the function from the data dictionary
        data['initial_params'],  # Pass the initial value for x
        **data.get('additional_params', {})  # Ensure this is a dictionary
    )

    end_time = time.time()

    mem_after = process.memory_info().rss

    factors.run_time = end_time - start_time
    factors.accuracy = result_x
    # calculate_accuracy(data['function'], result_x)
    factors.memory_usage = mem_after - mem_before

    return factors, result_x, minvalue

def calculate_accuracy(function, x):
    # Hier wird der Funktionswert an der Stelle x berechnet
    y = function(x)
    return y

# Define func1 before it's used
def func1(x):
    return (x + 1) ** 2

class OptimizationGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Optimization Problem Solver")
        self.create_widgets()

    def create_widgets(self):
        # Create Tabs
        self.tab_control = ttk.Notebook(self.root)

        # Tab 1: Add Problem
        self.tab_add_problem = ttk.Frame(self.tab_control)
        self.tab_control.add(self.tab_add_problem, text='Add Problem')
        self.create_add_problem_tab()

        # Tab 2: Solve Problems
        self.tab_solve_problems = ttk.Frame(self.tab_control)
        self.tab_control.add(self.tab_solve_problems, text='Solve Problems')
        self.create_solve_problems_tab()

        # Tab 3: Add Algorithm
        self.tab_add_algorithm = ttk.Frame(self.tab_control)
        self.tab_control.add(self.tab_add_algorithm, text='Add Algorithm')
        self.create_add_algorithm_tab()

        # Tab 4: Compare Algorithms
        self.tab_compare_algorithms = ttk.Frame(self.tab_control)
        self.tab_control.add(self.tab_compare_algorithms, text='Compare Algorithms')
        self.create_compare_algorithms_tab()

        self.tab_control.pack(expand=1, fill='both')

    def create_add_problem_tab(self):
        """Create the Add Problem tab."""
        ttk.Label(self.tab_add_problem, text="Python File:").grid(column=0, row=0, padx=10, pady=10)
        self.problem_file_entry = ttk.Entry(self.tab_add_problem, width=50)
        self.problem_file_entry.grid(column=1, row=0, padx=10, pady=10)
        ttk.Button(self.tab_add_problem, text="Browse", command=self.browse_problem_file).grid(column=2, row=0, padx=10, pady=10)

        ttk.Label(self.tab_add_problem, text="Initial Parameters:").grid(column=0, row=1, padx=10, pady=10)
        self.initial_params_entry = ttk.Entry(self.tab_add_problem, width=50)
        self.initial_params_entry.grid(column=1, row=1, padx=10, pady=10)

        ttk.Label(self.tab_add_problem, text="Additional Parameters (Optional):").grid(column=0, row=2, padx=10, pady=10)
        self.additional_params_entry = ttk.Entry(self.tab_add_problem, width=50)
        self.additional_params_entry.grid(column=1, row=2, padx=10, pady=10)

        ttk.Label(self.tab_add_problem, text="Folder Name:").grid(column=0, row=3, padx=10, pady=10)
        self.folder_name_entry = ttk.Entry(self.tab_add_problem, width=50)
        self.folder_name_entry.grid(column=1, row=3, padx=10, pady=10)

        ttk.Button(self.tab_add_problem, text="Add Problem", command=self.add_problem).grid(column=1, row=4, padx=10, pady=20)

    def browse_problem_file(self):
        """Browse for a Python file."""
        filename = filedialog.askopenfilename(filetypes=[("Python files", "*.py")])
        self.problem_file_entry.delete(0, tk.END)
        self.problem_file_entry.insert(0, filename)

    def add_problem(self):
        """Add a new optimization problem."""
        file_path = self.problem_file_entry.get()
        initial_params = self.initial_params_entry.get()
        additional_params = self.additional_params_entry.get()
        folder_name = self.folder_name_entry.get()

        if not file_path or not initial_params:
            messagebox.showerror("Error", "Please provide the required fields.")
            return

        # Convert initial_params and additional_params to dictionary
        try:
            initial_params_dict = eval(initial_params)
            additional_params_dict = eval(additional_params) if additional_params else {}
        except Exception as e:
            messagebox.showerror("Error", f"Invalid parameter format: {e}")
            return

        # Create problem directory
        problem_id = str(uuid.uuid4())
        problem_dir = os.path.join(PROBLEMS_DIR, folder_name, problem_id)
        os.makedirs(problem_dir, exist_ok=True)

        # Copy Python file to the problem directory
        shutil.copy(file_path, os.path.join(problem_dir, 'calibration_function.py'))

        # Create JSON file with problem data
        problem_data = {
            'initial_params': initial_params_dict,
            'additional_params': additional_params_dict
        }
        with open(os.path.join(problem_dir, 'calibration_function.json'), 'w') as file:
            json.dump(problem_data, file, indent=4)

        messagebox.showinfo("Success", f"Problem '{folder_name}' added successfully!")

    def create_solve_problems_tab(self):
        """Create the Solve Problems tab."""
        ttk.Button(self.tab_solve_problems, text="Run All Evaluations", command=self.run_all_evaluations).pack(padx=10, pady=10)

        self.output_text = tk.Text(self.tab_solve_problems, wrap=tk.WORD, height=20)
        self.output_text.pack(padx=10, pady=10, expand=True, fill='both')

    def run_all_evaluations(self):
        """Run all evaluations on problems."""
        algorithms = load_modules_and_find_classes(ALGORITHMS_DIR)
        results = []

        def log_output(message):
            self.output_text.insert(tk.END, message + "\n")
            self.output_text.see(tk.END)
            self.root.update()

        def evaluate():
            print(PROBLEMS_DIR)
            da=DataManager("camera_calibration_problems")
            python_functions, json_data = DataManager.load_all_calibration_problems(da,PROBLEMS_DIR)
            print(python_functions)
            module_class_pairs=[]
            for module, classes in algorithms.items():
                for cls in classes:
                    module_class_pairs.append(f"{module}.{cls}")

            for problem, data in zip(python_functions, json_data):
                for module_class in reversed(module_class_pairs):
                    module_name, class_name = module_class.rsplit('.', 1)
                    algorithm = load_algorithm(module_name, class_name)
                    try:
                        factors, result_x, minvalue = evaluate_algorithm(algorithm, data, problem)
                        result_info = {
                            'problem': problem,
                            'algorithm': class_name,
                            'run_time': factors.run_time,
                            'accuracy': factors.accuracy,
                            'memory_usage': factors.memory_usage,
                            'min_value': minvalue
                        }
                        results.append(result_info)
                        log_output(f"Problem: {problem}, Algorithm: {class_name}, Run Time: {factors.run_time} seconds, "
                                   f"Accuracy: {factors.accuracy}, Memory Usage: {factors.memory_usage} bytes, "
                                   f"Minimum Value: {minvalue}")
                    except Exception as e:
                        log_output(f"Evaluation failed for algorithm '{class_name}' on problem '{problem}': {e}")

        evaluate()

        # Save results to CSV
        try:
            import pandas as pd
            df = pd.DataFrame(results)
            df.to_csv(RESULTS_CSV, index=False)
            log_output(f"Results saved to '{RESULTS_CSV}'")
        except Exception as e:
            log_output(f"Failed to save results to CSV: {e}")

    def create_add_algorithm_tab(self):
        """Create the Add Algorithm tab."""
        ttk.Label(self.tab_add_algorithm, text="Python File:").grid(column=0, row=0, padx=10, pady=10)
        self.algorithm_file_entry = ttk.Entry(self.tab_add_algorithm, width=50)
        self.algorithm_file_entry.grid(column=1, row=0, padx=10, pady=10)
        ttk.Button(self.tab_add_algorithm, text="Browse", command=self.browse_algorithm_file).grid(column=2, row=0, padx=10, pady=10)

        ttk.Button(self.tab_add_algorithm, text="Add Algorithm", command=self.add_algorithm).grid(column=1, row=1, padx=10, pady=20)

    def browse_algorithm_file(self):
        """Browse for a Python file."""
        filename = filedialog.askopenfilename(filetypes=[("Python files", "*.py")])
        self.algorithm_file_entry.delete(0, tk.END)
        self.algorithm_file_entry.insert(0, filename)

    def add_algorithm(self):
        """Add a new optimization algorithm."""
        file_path = self.algorithm_file_entry.get()

        if not file_path:
            messagebox.showerror("Error", "Please select an algorithm file.")
            return

        # Copy algorithm file to the algorithms directory
        root_dir = os.path.dirname(os.path.abspath(__file__))  # Ermittelt das Root-Verzeichnis des aktuellen Skripts
        shutil.copy(file_path, os.path.join(ALGORITHMS_DIR, os.path.basename(file_path)))
        shutil.copy(file_path, os.path.join(root_dir, os.path.basename(file_path)))
        messagebox.showinfo("Success", "Algorithm added successfully!")

    def create_compare_algorithms_tab(self):
        """Create the Compare Algorithms tab."""
        ttk.Button(self.tab_compare_algorithms, text="Compare Algorithms", command=self.compare_algorithms).pack(padx=10, pady=10)

        self.compare_output_text = tk.Text(self.tab_compare_algorithms, wrap=tk.WORD, height=20)
        self.compare_output_text.pack(padx=10, pady=10, expand=True, fill='both')

    def compare_algorithms(self):
        """Compare algorithms based on results."""
        self.compare_output_text.delete(1.0, tk.END)

        def log_compare_output(message):
            self.compare_output_text.insert(tk.END, message + "\n")
            self.compare_output_text.see(tk.END)
            self.root.update()

        try:
            import pandas as pd
            df = pd.read_csv(RESULTS_CSV)
            log_compare_output("Comparing Algorithms...\n")
            log_compare_output(df.to_string(index=False))
        except Exception as e:
            log_compare_output(f"Failed to load results for comparison: {e}")

if __name__ == "__main__":
    root = tk.Tk()
    gui = OptimizationGUI(root)
    root.mainloop()
