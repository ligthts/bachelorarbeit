import json
import os
import shutil
import uuid

import cv2
import numpy as np


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
        file_path = os.path.join(self.directory, f"{filename}")
        with open(file_path, 'r') as file:
            data = json.load(file)
        return data


    def create_calibration_problem(self,problem_name,data, calibration_function_code):
        # Hauptordner für Kalibrierungsprobleme
        base_dir = 'camera_calibration_problems'
        f=str(uuid.uuid4())
        # Verzeichnis für das spezifische Problem erstellen
        problem_dir = os.path.join(base_dir, problem_name,f)
        if not os.path.exists(problem_dir):
            os.makedirs(problem_dir)

        # Python-Datei erstellen
        file_path = os.path.join(problem_dir, 'calibration_function.py')
        with open(file_path, 'w') as file:
            file.write(calibration_function_code)
        file_path = os.path.join(problem_dir, 'calibration_function.json')
        with open(file_path, 'w') as file:
            json.dump(data, file,indent=4)
        print(f"Kalibrierungsproblem '{problem_name}' erstellt.")

    import os

    def load_all_calibration_problems(base_dir):
        """
        Lädt alle Kalibrierungsprobleme aus den Unterordnern des Basisverzeichnisses.

        :param base_dir: Hauptverzeichnis, das die Kalibrierungsprobleme enthält.
        :return: Zwei Listen: eine für die geladenen Python-Funktionen und eine für die geladenen JSON-Daten.
        """
        python_functions = []
        json_data = []
        print("load")
        direct=r"camera_calibration_problems"
        try:
            # Durchsuche alle Unterordner im Basisverzeichnis
                problem_folder_path = os.path.join(direct,base_dir)
                print(problem_folder_path)
                if os.path.isdir(problem_folder_path):
                    print("2")
                    # Durchsuche UUID-Unterordner
                    for uuid_folder in os.listdir(problem_folder_path):
                        uuid_folder_path = os.path.join(problem_folder_path, uuid_folder)
                        if os.path.isdir(uuid_folder_path):
                            print(3)
                            python_file = os.path.join(uuid_folder_path, 'calibration_function.py')
                            json_file = os.path.join(uuid_folder_path, 'calibration_function.json')
                            #print(python_file)
                            # Lade Python-Datei, wenn vorhanden
                            if os.path.exists(python_file):
                                print(4)
                                with open(python_file, 'r') as file:
                                    code = file.read()
                                local_vars = {'np': np, 'cv2': cv2}

                                exec(code, globals(), local_vars)
                                if 'calibration_function' in local_vars:
                                    print(5)
                                    python_functions.append(

                                        local_vars['calibration_function'])
                                else:
                                    print(f"Keine Funktion 'calibration_function' in '{python_file}' gefunden.")
                                #print(python_functions)
                            # Lade JSON-Datei, wenn vorhanden
                            if os.path.exists(json_file):
                                print(6)
                                with open(json_file, 'r') as file:
                                    data = json.load(file)
                                    print(7)
                                json_data.append(data
                                )


        except Exception as e:
            print(f"Fehler beim Laden der Kalibrierungsprobleme: {e}")

        return python_functions, json_data
def list_calibration_problems(self,base_dir):
        """
        Listet alle Kalibrierungsprobleme im angegebenen Verzeichnis auf.

        :param base_dir: Hauptverzeichnis, das die Kalibrierungsprobleme enthält.
        :return: Liste der Namen der Kalibrierungsprobleme.
        """
        problems = []
        try:
            for item in os.listdir(base_dir):
                item_path = os.path.join(base_dir, item)
                if os.path.isdir(item_path):
                    if os.path.exists(os.path.join(item_path, 'calibration_function.py')):
                        problems.append(item)
        except Exception as e:
            print(f"Fehler beim Auflisten der Kalibrierungsprobleme: {e}")
        return problems