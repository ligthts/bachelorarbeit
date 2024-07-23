import scipy.optimize
from scipy.optimize import least_squares
from optimization_algorithm import OptimizationAlgorithm
import numpy as np

class Ersta(OptimizationAlgorithm):
    def optimize(self, residuals, initial_params, x_data, y_data, **kwargs):
        # Konvertiere Daten in Numpy-Arrays
        x_data = np.array(x_data)
        y_data = np.array(y_data)
        print(residuals)
        result = least_squares(residuals, initial_params, args=(x_data, y_data), method="BFGS")
        print(type(result))
        return result