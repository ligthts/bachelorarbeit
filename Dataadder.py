import json
import os

class DataManager:
    def __init__(self, directory):
        self.directory = directory
        if not os.path.exists(directory):
            os.makedirs(directory)

    def add_data(self, filename, data):
        # Speichern der Daten in einer JSON-Datei
        file_path = os.path.join(self.directory, f"{filename}.json")
        with open(file_path, 'w') as file:
            json.dump(data, file, indent=4)

    def load_data(self, filename):
        # Laden der Daten aus einer JSON-Datei
        file_path = os.path.join(self.directory, f"{filename}.json")
        with open(file_path, 'r') as file:
            data = json.load(file)
        return data
