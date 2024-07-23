import time
import importlib
import pandas as pd
import numpy as np
import os
import inspect
import psutil
from empirical import Empiricalschatzung
from Dataadder import DataManager


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
    result = algorithm.optimize(
        lambda params, x, y: model(params, x, y) - y,
        data['initial_params'],
        data['x'],
        data['y'],
        **data.get('additional_params', {})
    )
    end_time = time.time()

    mem_after = process.memory_info().rss

    factors.run_time = end_time - start_time
    factors.accuracy = calculate_accuracy(result.x, data['x'], data['y'])
    factors.memory_usage = mem_after - mem_before

    return factors, result


def calculate_accuracy(params, x, y):
    x = np.array(x)
    y = np.array(y)
    model_y = model(params, x, y)
    mse = np.mean((y - model_y) ** 2)
    rmse = np.sqrt(mse)
    return rmse


def model(params, x, y):
    f = 0
    result = 0
    for i in reversed(params):
        result += i * x ** f
        f += 1
    return result


if __name__ == "__main__":
    data = {
        'x': [0, 1, 2, 3, 4, 5],
        'y': [0, 1, 4, 9, 16, 25],
        'initial_params': [1, 0, 0],
        'residuals': lambda params, x, y: model(params, x) - y,
        'additional_params': {
            'method': 'lm'
        }
    }
    adder = DataManager("data")
    adder.add_data("camera_calibration", data)
    data_list = DataManager.load_data(adder, "camera_calibration")
    algorithms = load_modules_and_find_classes('optimization_algos')
    module_class_pairs = []
    for module, classes in algorithms.items():
        for cls in classes:
            module_class_pairs.append(f"{module}.{cls}")

    for module_class in module_class_pairs:
        module_name, class_name = module_class.rsplit('.', 1)
        print(module_name, class_name)
        algorithm = load_algorithm(module_name, class_name)
        for da in data_list:
            factors, result = evaluate_algorithm(algorithm, da)
            Empiricalschatzung.Plotting(Empiricalschatzung, algorithm)
            print(f"Algorithm: {class_name}")
            print(f"Run Time: {factors.run_time} seconds")
            print(f"Accuracy: {factors.accuracy}")
            print(f"Memory Usage: {factors.memory_usage} bytes")
            print(f"Optimized Parameters: {result.x}")

# DataManager.py
