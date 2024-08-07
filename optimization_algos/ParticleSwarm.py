from pyswarm import pso

class ParticleAlgorithm:
    def __init__(self, learning_rate=0.01, max_iterations=100, tolerance=1e-6):
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
        percent_deviation=30
        absolute_deviation=10000
        bounds_lower = []
        bounds_upper = []
        bounds=[]
        for param in initial_x:
            if param != 0:
                lower_bound = param * (1 - percent_deviation)
                upper_bound = param * (1 + percent_deviation)
            else:
                lower_bound = -absolute_deviation
                upper_bound = absolute_deviation
            if lower_bound >= upper_bound:
                lower_bound, upper_bound = min(lower_bound, upper_bound), max(lower_bound, upper_bound)
            if lower_bound<0:
                lower_bound=0
            bounds.append((lower_bound,upper_bound))
        print(bounds)
        # Rufe least_squares mit den gefilterten Argumenten auf
        result = pso(func, bounds,lb=[b[0] for b in bounds], ub=[b[1] for b in bounds])
        result_x = func(result.x)
        return result, result_x
