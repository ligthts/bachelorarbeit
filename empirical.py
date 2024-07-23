import numpy as np
import time
import matplotlib


class Empiricalschatzung:
    def empiricalschatzung(self, algorithm, sizes):
        times = []
        for size in sizes:
            data = self.generate_data(size)
            start_time = time.time()
            algorithm.optimize(**data)
            end_time = time.time()
            times.append(end_time - start_time)
        return sizes, times
    def empiricalschatzung2(self, algorithm, sizes):
        times = []
        for size in sizes:
            data = self.generate_data2(size)
            start_time = time.time()
            algorithm.optimize(**data)
            end_time = time.time()
            times.append(end_time - start_time)
        return sizes, times

    def generate_data(size):
        x_data = np.linspace(0, 10, size)
        y_data = 3 * x_data ** 2 + 2 * x_data + 1
        initial_params = [1, 1, 1]
        return {
            'residuals': lambda params, x, y: params[0] * x ** 2 + params[1] * x + params[2] - y,
            'initial_params': initial_params,
            'x_data': x_data,
            'y_data': y_data
        }

    def generate_data2(param_count):
        x_data = np.linspace(0, 10, 100)  # Feste Größe der Eingabedaten
        y_data = sum((i + 1) * x_data ** i for i in range(param_count))  # Polynom mit 'param_count' Parametern
        initial_params = [1] * param_count
        return {
            'residuals': lambda params, x, y: sum(params[i] * x ** i for i in range(param_count)) - y,
            'initial_params': initial_params,
            'x_data': x_data,
            'y_data': y_data
        }
    def Plotting(self,main_obj ):
        sizes = [5000, 10000, 15000,20000 ,30000]
        sizes, times = self.empiricalschatzung(self,main_obj, sizes)
        sizes2=[10,20,30,40,50]
        sizes2,times2=self.empiricalschatzung2(self,main_obj,sizes2)

    # Ergebnis der empirischen Schätzung anzeigen
        for size, time_taken in zip(sizes, times):
            print(f"Size: {size}, Time taken: {time_taken:.5f} seconds")
        for size, time_taken in zip(sizes2, times2):
            print(f"Size: {size}, Time taken: {time_taken:.5f} seconds")

    # Optional: Plotte die Ergebnisse zur Veranschaulichung
        try:
            import matplotlib.pyplot as plt

            plt.plot(sizes, times, marker='o')
            plt.xlabel('Input Size')
            plt.ylabel('Time (seconds)')
            plt.title('Empirical Time Complexity Estimation')
            plt.grid(True)
            plt.show()
        except ImportError:
            print("Matplotlib is not installed. Skipping the plot.")
        try:
            import matplotlib.pyplot as plt

            plt.plot(sizes2, times2, marker='o')
            plt.xlabel('Input Size')
            plt.ylabel('Time (seconds)')
            plt.title('Empirical Time Complexity Estimation')
            plt.grid(True)
            plt.show()
        except ImportError:
            print("Matplotlib is not installed. Skipping the plot.")