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
        :param kwargs: Zusätzliche Parameter für den Algorithmus.
        :return: Der x-Wert, der die Funktion minimiert, und der minimale Funktionswert.
        """

        # Verwenden Sie Levenberg-Marquardt mit least_squares zur Optimierung
        result = least_squares(
            fun=lambda x: function(x),
            x0=np.array([initial_x]),
            max_nfev=self.max_iterations,
            ftol=self.tolerance,
            xtol=self.tolerance,
            method="lm",
            **kwargs,
            verbose=2
        )

        # Der optimierte x-Wert
        optimized_x = result.x
        # Der minimale Wert der Funktion bei diesem x
        min_value = function(optimized_x)

        return result, min_value