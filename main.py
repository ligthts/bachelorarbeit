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


def evaluate_algorithm(algorithm, data):
    factors = EvaluationFactors()

    process = psutil.Process()
    mem_before = process.memory_info().rss
    start_time = time.time()

    # Optimierung der Funktion auf den x-Wert, der das Minimum erreicht
    result_x, result_value = algorithm.optimize(
        data['function'],
        data['initial_x'],
        **data.get('additional_params', {})  # Sicherstellen, dass dies ein Dictionary ist
    )

    end_time = time.time()

    mem_after = process.memory_info().rss

    factors.run_time = end_time - start_time
    factors.accuracy = calculate_accuracy(data['function'], result_x)
    factors.memory_usage = mem_after - mem_before

    return factors, result_x, result_value


def calculate_accuracy(function, x):
    # Hier wird der Funktionswert an der Stelle x berechnet
    y = function(x)
    return y


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
    #adder.add_data("function_optimization", data)
    data = DataManager.load_data(adder, r"C:\Users\fabia\PycharmProjects\BachelorArbeit\data\camera_calibration_results.json")
    calibration_function_code = data['calibration_function']
    initial_params = data['initial_params']
    additional_params = data['additional_params']
    print(calibration_function_code)
    # Konvertiere den Code in eine ausführbare Funktion
    local_scope = {}
    exec(calibration_function_code, {"np": np, "cv2": cv2}, local_scope)
    calibration_function = local_scope['calibration_function']
    da={
        'function':calibration_function,
        'initial_x':initial_params,
        'additional_params':additional_params
    }
    algorithms = load_modules_and_find_classes('optimization_algos')
    module_class_pairs = []
    for module, classes in algorithms.items():
        for cls in classes:
            module_class_pairs.append(f"{module}.{cls}")

    for module_class in module_class_pairs:
        module_name, class_name = module_class.rsplit('.', 1)
        print(module_name, class_name)
        algorithm = load_algorithm(module_name, class_name)
        if True:
            factors, result_x, result_value = evaluate_algorithm(algorithm, da)
            #Empiricalschatzung.Plotting(Empiricalschatzung, algorithm)
            print(f"Algorithm: {class_name}")
            print(f"Run Time: {factors.run_time} seconds")
            print(f"Accuracy: {factors.accuracy}")
            print(f"Memory Usage: {factors.memory_usage} bytes")
            print(f"Minimum x: {result_x}, Minimum value: {result_value}")
        else:
            print("nicht möglich")