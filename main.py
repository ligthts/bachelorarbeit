# main.py

import time
import importlib
import os
import inspect
import psutil
import numpy as np
from empirical import Empiricalschatzung
from Dataadder import DataManager
import cv2

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
    python_functions, json_data = DataManager.load_all_calibration_problems(base_dir)
    for module, classes in algorithms.items():
        for cls in classes:
            module_class_pairs.append(f"{module}.{cls}")

    # Iteriere über alle geladenen Kalibrierungsprobleme
    for problem in python_functions:
        for data in json_data:
            print("hier sind die additional",data['additional_params'])
            for module_class in (module_class_pairs):
                module_name, class_name = module_class.rsplit('.', 1)
                print(module_name, class_name)
                algorithm = load_algorithm(module_name, class_name)

                # Iteriere über alle Algorithmen
                print(algorithm)
                try:
                    #print(f"Evaluating problem '{problem['name']}' with algorithm '{algorithm.__name__}'")
                    factors,result_x,minvalue = evaluate_algorithm(algorithm, data,problem)
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
                        print("nicht möglich mit :",result_x)
                except():
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

def evaluate_algorithm(algorithm, data,function):
    factors = EvaluationFactors()

    process = psutil.Process()
    mem_before = process.memory_info().rss
    start_time = time.time()
    print("hier2:",data['additional_params'])
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
        #calculate_accuracy(data['function'], result_x)
    factors.memory_usage = mem_after - mem_before

    return factors, result_x,minvalue
        #data['function'](result_x)

def calculate_accuracy(function, x):
    # Hier wird der Funktionswert an der Stelle x berechnet
    y = function(x)
    return y

# Define func1 before it's used
def func1(x):
    return (x + 1) ** 2

if __name__ == "__main__":
    # Beispiel: Optimierung der Funktion x^2
    data = {
        'function': lambda x: x**2,  # Die Funktion, die minimiert werden soll
        'initial_x': 10000,             # Startwert für x
        'additional_params': {       # Zusätzliche Parameter als Dictionary
            'learning_rate': 0.1,
            'max_iterations': 1000,
            'tolerance': 1e-6
        }
    }

    adder = DataManager(r"data\test")
    # adder.add_data("function_optimization", data)
    data = DataManager.load_data(adder, r"C:\Users\fabia\PycharmProjects\BachelorArbeit\data\camera_calibration_results.json")

    # Importiere die Kalibrierungsfunktion aus der externen Datei
    #from calibration_function import calibration_function

    initial_params = data['initial_params']
    additional_params = data['additional_params']

    da = {
        'function': lambda x:x**2,
        'initial_x': 1,
        'additional_params': {}
    }

    algorithms = load_modules_and_find_classes('optimization_algos')
    module_class_pairs = []
    for module, classes in algorithms.items():
        for cls in classes:
            module_class_pairs.append(f"{module}.{cls}")
    run_all_evaluations("shape_estimation_results",algorithms)
    for module_class in reversed(module_class_pairs):
        module_name, class_name = module_class.rsplit('.', 1)
        print(module_name, class_name)
        algorithm = load_algorithm(module_name, class_name)
        if False:
            factors, result_x = evaluate_algorithm(algorithm, da)
            # Empiricalschatzung.Plotting(Empiricalschatzung, algorithm)
            print(f"Algorithm: {class_name}")
            print(f"Run Time: {factors.run_time} seconds")
            print(f"Accuracy: {factors.accuracy}")
            print(f"Memory Usage: {factors.memory_usage} bytes")
            print(f"Minimum x: {result_x}, Minimum value: {result_x}")
        else:
            print("nicht möglich")


