# In der Datei optimization_algos/gradient_descent.py

class GradientDescentAlgorithm:
    def __init__(self, learning_rate=0.01, max_iterations=1000, tolerance=1e-6):
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.tolerance = tolerance

    def optimize(self, function, initial_x, **kwargs):
        x = initial_x
        for _ in range(self.max_iterations):
            grad = self._compute_gradient(function, x)
            new_x = x - self.learning_rate * grad

            if abs(new_x - x) < self.tolerance:
                break

            x = new_x

        return x, function(x)

    def _compute_gradient(self, function, x, epsilon=1e-8):
        # Numerische Ableitung zur Approximation des Gradienten
        return (function(x + epsilon) - function(x - epsilon)) / (2 * epsilon)
