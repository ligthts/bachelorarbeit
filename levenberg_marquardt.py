from scipy.optimize import least_squares
from optimization_algorithm import OptimizationAlgorithm
import numpy as np

class LevenbergMarquardtAlgorithm(OptimizationAlgorithm):
    def __init__(self, max_iterations=1000, tolerance=1e-6):
        """
        Initialisiert den Levenberg-Marquardt-Algorithmus.

        :param max_iterations: Maximale Anzahl von Iterationen für den Optimierungsprozess.
        :param tolerance: Toleranz für die Abbruchbedingung.
        """
        self.max_iterations = max_iterations
        self.tolerance = tolerance

    def optimize(self, function, initial_x, **kwargs):
        """
        Optimiert die Eingabefunktion und findet das x, das den Funktionswert minimiert.

        :param function: Eine Funktion, die minimiert werden soll.
        :param initial_x: Der Startwert für die Optimierung.
        :param kwargs: Zusätzliche Parameter für den Algorithmus, die an `least_squares` übergeben werden.
        :return: Der x-Wert, der die Funktion minimiert, und der minimale Funktionswert.
        """

        # Filtere die unerwünschten Parameter heraus
        valid_kwargs = {key: value for key, value in kwargs.items() if key in {'max_nfev', 'ftol', 'xtol','bounds'}}
        def function1(x):
            res=function(x)
            if len(res)!=len(x):
                r=[]
                for d in range(len(x)):
                    r.append(res)
                return r
            else:
                return res
        # Verwenden Sie Levenberg-Marquardt mit least_squares zur Optimierung
        try:
            result = least_squares(
            function,
            x0=initial_x,
            **valid_kwargs,
            verbose=2,
            method="lm"
            )

        # Der optimierte x-Wert
            optimized_x = result.x
            print("hier sind die extrhierten werte",optimized_x)
        # Der minimale Wert der Funktion bei diesem x
            min_value = function(optimized_x)

            return result, min_value
        except Exception as e:
            print(e)
            return 0, 0
