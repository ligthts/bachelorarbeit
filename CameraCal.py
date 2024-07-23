import numpy as np
import cv2
import glob
import os
import pandas as pd
from Dataadder import DataManager

def residuals(params, x, y):
    mtx = params[:9].reshape((3, 3))  # Kameramatrix wiederherstellen
    dist = params[9:]  # Verzerrungskoeffizienten wiederherstellen
    return np.concatenate((mtx.flatten(), dist.flatten())) - np.concatenate((x, y))

# Beispielbilder für die Kalibrierung
image_paths = glob.glob('calibration_images/*.jpg')
directory = r"C:\Users\fabia\Downloads\archive\data\imgs\leftcamera"
image_paths1 = [os.path.join(directory, file) for file in os.listdir(directory) if file.endswith('.jpg') or file.endswith('.png')]
image_paths += image_paths1
print(f"Total number of images: {len(image_paths)}")

# Größen der Schachbrettmuster
pattern_sizes = [(11,7), (12, 8),(9,6),(8,6),(7,5)]  # Liste der verschiedenen Schachbrettmuster-Größen

calibration_data = []
for path in image_paths:
    img = cv2.imread(path)
    if img is None:
        print(f"Could not read image {path}")
        continue
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    corners_found = False
    for pattern_size in pattern_sizes:
        # Vorbereitung der Schachbrettmuster-Punkte
        objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
        objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2)

        # Finde die Schachbrettmuster-Ecken
        ret, corners = cv2.findChessboardCorners(gray, pattern_size, None)

        if ret:
            print(f"Chessboard corners found in image: {path} with pattern size: {pattern_size}")
            corners_found = True
            obj_points = [objp]
            img_points = [corners]

            # Kamera-Kalibrierung für das aktuelle Bild
            ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, gray.shape[::-1], None, None)

            # Extrahiere die Kameraparameter in die gewünschte Struktur
            x_data = mtx.flatten()  # Flattened Kameramatrix
            y_data = dist.flatten()  # Flattened Verzerrungskoeffizienten

            # Gleiche Länge sicherstellen, überschüssige Elemente entfernen
            min_length = min(len(x_data), len(y_data))
            x_data = x_data[:min_length]
            y_data = y_data[:min_length]

            initial_params = np.concatenate((x_data, y_data))  # Initialparameter für die Optimierung

            # Struktur für die Rückgabe
            output = {
                'x': x_data.tolist(),
                'y': y_data.tolist(),
                'initial_params': initial_params.tolist(),
                'residuals': residuals,
                'additional_params': {
                    'method': 'lm'
                }
            }

            calibration_data.append(output)
            man = DataManager('data')
            man.add_data("shape_estimation", output)
            break  # Wenn ein passendes Muster gefunden wurde, die Schleife verlassen

    if not corners_found:
        print(f"Chessboard corners not found in image: {path}")

# Konvertiere die Liste der Kalibrierungsdaten in ein DataFrame
calibration_df = pd.DataFrame(calibration_data)
print(calibration_df)

# Speichere die Kalibrierungsdaten
man = DataManager('data')
