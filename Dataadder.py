import os
import time
import pandas as pd
import numpy as np
import json
import uuid

class DataManager:
    def __init__(self, base_dir):
        self.base_dir = base_dir

    def add_data(self, specialization, data):
        specialization_dir = os.path.join(self.base_dir, specialization)
        if not os.path.exists(specialization_dir):
            os.makedirs(specialization_dir)

        timestamp = int(time.time())  # Aktuelle Zeit in Sekunden seit der Epoch
        fd=str(uuid.uuid4())
        filename = f'data_{fd}.json'  # Eindeutiger Dateiname basierend auf der Zeit

        file_path = os.path.join(specialization_dir, filename)

        # Speichere Daten als JSON-Datei
        with open(file_path, 'w') as f:
            # Konvertiere die Residuals-Funktion in eine speicherbare Form
            data_copy = data.copy()
            data_copy['residuals'] = {
                'expression': 'np.concatenate((params[:9].reshape((3, 3)).flatten(), params[9:])) - np.concatenate((x, y))',
                'variables': ['params', 'x', 'y']
            }
            json.dump(data_copy, f, indent=4)
        print(f"Daten in {file_path} gespeichert.")

    def load_data(self, specialization):
        specialization_dir = os.path.join(self.base_dir, specialization)
        if not os.path.exists(specialization_dir):
            print(f"Keine Daten für {specialization} gefunden.")
            return None

        data_files = [f for f in os.listdir(specialization_dir) if f.endswith('.json')]

        if not data_files:
            print(f"Keine Daten für {specialization} gefunden.")
            return None

        all_data = []

        for data_file in data_files:
            file_path = os.path.join(specialization_dir, data_file)
            with open(file_path, 'r') as f:
                data = json.load(f)

            # Rekonstruiere die Residuals-Funktion
            residuals_str = data['residuals']['expression']
            residuals_func = eval(f"lambda params, x, y: {residuals_str}")

            # Konvertiere Daten zurück in das gewünschte Output-Format
            output = {
                'x': data['x'],
                'y': data['y'],
                'initial_params': data['initial_params'],
                'residuals': residuals_func,
                'additional_params': data['additional_params']
            }
            all_data.append(output)

        return all_data

