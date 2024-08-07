from scipy.optimize import dual_annealing

class SimulatedAnnealingAlgorithm:
    def __init__(self, learning_rate=0.01, max_iterations=1000, tolerance=1e-6):
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.tolerance = tolerance

    def optimize(self,func, initial_x, **kwargs):
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
        #gtol entfernt
        print(initial_x)
        # Filtere nur die gültigen Argumente für least_squares
        filtered_kwargs = {key: value for key, value in kwargs.items() if key in valid_keys}
        print(filtered_kwargs)
        upper_bounds=[]
        lower_bounds=[]
        percent_deviation=30
        absolute_deviation=10000
        for param in initial_x:
            if param != 0:
                lower_bound = param * (1 - percent_deviation)
                upper_bound = param * (1 + percent_deviation)
            else:
                lower_bound = -absolute_deviation
                upper_bound = absolute_deviation
            if lower_bound >= upper_bound:
                lower_bound, upper_bound = min(lower_bound, upper_bound), max(lower_bound, upper_bound)
            upper_bounds.append(upper_bound)
            lower_bounds.append(lower_bound)
        bounds = (lower_bounds, upper_bounds)
        # Rufe least_squares mit den gefilterten Argumenten auf
        result = dual_annealing(func, bounds, **filtered_kwargs)
        result_x = func(result.x)
        return result, result_x
