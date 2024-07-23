import ast
import inspect
from scipy.optimize import least_squares
from levenberg_marquardt import LevenbergMarquardtAlgorithm


# Funktion zur Extraktion des Quellcodes eines gegebenen Objekts
def get_source_code(obj):
    return inspect.getsource(obj)


# Funktion zur statischen Analyse des Quellcodes
class ComplexityAnalyzer(ast.NodeVisitor):
    def __init__(self):
        self.loops = 0
        self.function_calls = 0
        self.recursive_calls = 0
        self.current_function = None
        self.functions = {}

    def visit_FunctionDef(self, node):
        self.current_function = node.name
        self.functions[node.name] = 0
        self.generic_visit(node)
        self.current_function = None

    def visit_For(self, node):
        self.loops += 1
        self.generic_visit(node)

    def visit_While(self, node):
        self.loops += 1
        self.generic_visit(node)

    def visit_Call(self, node):
        if isinstance(node.func, ast.Name) and node.func.id == self.current_function:
            self.recursive_calls += 1
        self.function_calls += 1
        self.generic_visit(node)

    def analyze(self, source_code):
        tree = ast.parse(source_code)
        self.visit(tree)

        return {
            "loops": self.loops,
            "function_calls": self.function_calls,
            "recursive_calls": self.recursive_calls
        }


# Funktion zur Rekursion durch die Methodenaufrufe und deren Analyse
def analyze_method_calls(class_obj, analyzed_methods=set()):
    class_name = class_obj.__class__.__name__
    module_name = class_obj.__module__

    if class_name in analyzed_methods:
        return {"loops": 0, "function_calls": 0, "recursive_calls": 0}

    analyzed_methods.add(class_name)
    methods = [func for func in dir(class_obj) if callable(getattr(class_obj, func)) and not func.startswith("__")]

    result = {"loops": 0, "function_calls": 0, "recursive_calls": 0}

    for method in methods:
        method_name = f"{module_name}.{class_name}.{method}"
        try:
            func = getattr(class_obj, method)
            source_code = get_source_code(func)
        except (ImportError, AttributeError) as e:
            print(f"Fehler beim Importieren der Methode {method_name}: {e}")
            continue

        analyzer = ComplexityAnalyzer()
        method_result = analyzer.analyze(source_code)

        for sub_func in analyzer.functions.keys():
            sub_func_fullname = f"{module_name}.{class_name}.{sub_func}"
            sub_result = analyze_method_calls(getattr(class_obj, sub_func), analyzed_methods)
            method_result["loops"] += sub_result["loops"]
            method_result["function_calls"] += sub_result["function_calls"]
            method_result["recursive_calls"] += sub_result["recursive_calls"]

        result["loops"] += method_result["loops"]
        result["function_calls"] += method_result["function_calls"]
        result["recursive_calls"] += method_result["recursive_calls"]

    return result


# Quellcode der Hauptklasse analysieren
main_obj = LevenbergMarquardtAlgorithm()
main_result = analyze_method_calls(main_obj)

# Ergebnis anzeigen
print(f"Number of loops: {main_result['loops']}")
print(f"Number of function calls: {main_result['function_calls']}")
print(f"Number of recursive calls: {main_result['recursive_calls']}")

# Einfache Heuristik zur Bestimmung der Komplexität
if main_result["recursive_calls"] > 0:
    print("Estimated Time Complexity: Recursive")
elif main_result["loops"] > 1:
    print("Estimated Time Complexity: Potentially O(n^2) or higher")
elif main_result["loops"] == 1:
    print("Estimated Time Complexity: O(n)")
else:
    print("Estimated Time Complexity: O(1)")
