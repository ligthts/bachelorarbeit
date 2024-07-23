from scipy.optimize import least_squares
from optimization_algorithm import OptimizationAlgorithm
import numpy as np

class LevenbergMarquardtAlgorithm(OptimizationAlgorithm):
    def optimize(self, residuals, initial_params, x_data, y_data, **kwargs):
    # Konvertiere Daten in Numpy-Arrays
        x_data = np.array(x_data)
        y_data = np.array(y_data)

        method = kwargs.get('method', 'lm')
        result = least_squares(residuals, initial_params, args=(x_data, y_data), method=method)
        return result
