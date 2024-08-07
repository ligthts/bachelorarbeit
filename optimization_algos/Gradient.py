from scipy.optimize import least_squares

class GradientDescentAlgorithm:
    def __init__(self, learning_rate=0.01, max_iterations=1000, tolerance=1e-6):
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.tolerance = tolerance

    def optimize_least_squares(self,func, initial_x, **kwargs):
        """
        Optimiert eine Funktion mit Hilfe des Least-Squares-Algorithmus von SciPy.

        Parameters:
        - func: Die Funktion, die optimiert werden soll.
        - initial_x: Liste oder Array von Startwerten für die Parameter.
        - **kwargs: Zusätzliche Optionen und Parameter, die an least_squares übergeben werden.

        Returns:
        - Ein OptimizeResult-Objekt mit den Optimierungsergebnissen.
        """
        # Filtern der relevanten Schlüsselwörter, die least_squares akzeptiert
        valid_keys = {
            'jac', 'bounds', 'method', 'ftol', 'xtol',
            'x_scale', 'loss', 'f_scale', 'diff_step', 'tr_solver',
            'tr_options', 'jac_sparsity', 'max_nfev', 'verbose',
            'args', 'kwargs'
        }
        print(initial_x)
        # Filtere nur die gültigen Argumente für least_squares
        filtered_kwargs = {key: value for key, value in kwargs.items() if key in valid_keys}

        # Rufe least_squares mit den gefilterten Argumenten auf
        result = least_squares(func, initial_x, **filtered_kwargs,verbose=2)
        result_x=func(result)
        return result, result_x
