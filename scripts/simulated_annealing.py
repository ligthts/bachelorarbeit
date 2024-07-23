import numpy as np

class SimulatedAnnealing:
    def __init__(self, initial_temperature=1000, cooling_rate=0.95, min_temperature=1e-3):
        self.initial_temperature = initial_temperature
        self.cooling_rate = cooling_rate
        self.min_temperature = min_temperature

    def optimize(self, function, initial_x, **kwargs):
        current_x = initial_x
        best_x = current_x
        best_value = function(current_x)
        current_temperature = self.initial_temperature

        while current_temperature > self.min_temperature:
            new_x = self._generate_neighbor(current_x)
            current_value = function(current_x)
            new_value = function(new_x)

            if self._acceptance_probability(current_value, new_value, current_temperature):
                current_x = new_x
                if new_value < best_value:
                    best_x = new_x
                    best_value = new_value

            current_temperature *= self.cooling_rate

        return best_x, best_value

    def _generate_neighbor(self, x):
        # Generiere einen neuen Nachbarwert von x
        change = np.random.uniform(-1, 1)
        return x + change

    def _acceptance_probability(self, current_value, new_value, temperature):
        if new_value < current_value:
            return True
        else:
            probability = np.exp((current_value - new_value) / temperature)
            return np.random.rand() < probability
