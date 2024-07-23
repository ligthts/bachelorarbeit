# DataManager.py

import json
import os


class DataManager:
    def __init__(self, directory):
        self.directory = directory
        if not os.path.exists(directory):
            os.makedirs(directory)

    def add_data(self, filename, data):
        # Konvertieren Sie die Funktion zu einem String, bevor Sie sie speichern
        if 'function' in data:
            data['function'] = self.function_to_string(data['function'])

        # Speichern der Daten in einer JSON-Datei
        file_path = os.path.join(self.directory, f"{filename}.json")
        with open(file_path, 'w') as file:
            json.dump(data, file, indent=4)

    @staticmethod
    def load_data(self, filename):
        # Laden der Daten aus einer JSON-Datei
        file_path = os.path.join(self.directory, f"{filename}.json")
        with open(file_path, 'r') as file:
            data = json.load(file)

        # Konvertieren Sie den String zurück in eine Funktion
        if 'function' in data:
            data['function'] = self.string_to_function(data['function'])

        return [data]  # Wir geben eine Liste zurück, um mit dem bestehenden Code kompatibel zu bleiben

    @staticmethod
    def function_to_string(func):
        # Konvertiert eine Funktion zu einem String
        return func.__name__

    @staticmethod
    def string_to_function(func_name):
        # Konvertiert einen Funktionsnamen (String) zurück in eine Funktion
        if func_name == "<lambda>":
            return lambda x: x ** 2  # Beispiel-Funktion, dies muss manuell für jede Funktion angepasst werden
        # Sie können hier zusätzliche Funktionen hinzufügen
        raise ValueError(f"Unbekannte Funktion: {func_name}")

