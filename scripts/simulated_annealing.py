from scipy.optimize import least_squares
from optimization_algorithm import OptimizationAlgorithm
import numpy as np

class SimulatedAnnelaingalgo(OptimizationAlgorithm):
    initial_temperature=1000
    cooling_rate=0.95
    min_temperature = 1e-3

    def optimize(self, residuals, initial_params, x_data, y_data, **kwargs):
        current_solution = np.array(initial_params)
        best_solution = np.copy(current_solution)
        best_value = np.sum(residuals(current_solution, x_data, y_data) ** 2)
        current_temperature = self.initial_temperature

        while current_temperature > self.min_temperature:
            new_solution = self._generate_neighbor(current_solution, bounds=np.array([[-5, 5]] * len(initial_params)))
            current_value = np.sum(residuals(current_solution, x_data, y_data) ** 2)
            new_value = np.sum(residuals(new_solution, x_data, y_data) ** 2)

            if self._acceptance_probability(current_value, new_value, current_temperature):
                current_solution = new_solution
                if new_value < best_value:
                    best_solution = np.copy(new_solution)
                    best_value = new_value

            current_temperature *= self.cooling_rate

        return best_solution, best_value

    def _generate_neighbor(self, solution, bounds):
        neighbor = np.copy(solution)
        index = np.random.randint(0, len(solution))
        change = np.random.uniform(-1, 1)
        neighbor[index] += change
        neighbor = np.clip(neighbor, bounds[:, 0], bounds[:, 1])
        return neighbor

    def _acceptance_probability(self, current_value, new_value, temperature):
        if new_value < current_value:
            return True
        else:
            probability = np.exp((current_value - new_value) / temperature)
            return np.random.rand() < probability