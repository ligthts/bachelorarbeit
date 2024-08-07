# CameraCal.py

import os
import cv2
import numpy as np
from Dataadder import DataManager
import matplotlib.pyplot as plt


class Shapeestimation:
    def __init__(self, directory):
        self.data_manager = DataManager(directory)

    def load_images_from_directory(self, image_directory):
        """
        Lädt alle Bilder aus einem angegebenen Verzeichnis.

        :param image_directory: Pfad zum Bildverzeichnis.
        :return: Liste von geladenen Bildern.
        """
        images = []
        for filename in os.listdir(image_directory):
            if filename.endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(image_directory, filename)
                img = cv2.imread(img_path)
                if img is not None:
                    images.append(img)
        return images

    def find_chessboard_corners(self, images, chessboard_size):
        """
        Findet Schachbrettecken in den gegebenen Bildern.

        :param images: Liste von Bildern.
        :param chessboard_size: Anzahl der inneren Ecken des Schachbretts (width, height).
        :return: Bildpunkte und Weltpunkte.
        """
        obj_points = []  # 3D-Punkte in realer Welt
        img_points = []  # 2D-Punkte in Bildebene

        # Definiere Weltkoordinaten für die Schachbrett-Ecken
        objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
        objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)

        for img in images:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # Finde die Schachbrettecken
            ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)

            if ret:
                img_points.append(corners)
                obj_points.append(objp)

        return obj_points, img_points

    def estimateshape(self, image_directory):
        """
        Kalibriert die Kamera mit Bildern aus einem Verzeichnis und speichert die Kalibrierungsfunktion.

        :param image_directory: Verzeichnis, das die Kalibrierungsbilder enthält.
        :param chessboard_size: Anzahl der inneren Ecken des Schachbretts (width, height).
        :return: Kameramatrix und Verzerrungskoeffizienten.
        """
        images = self.load_images_from_directory(image_directory)
        for image in images:
            _, binary_image = cv2.threshold(image, 200, 255, cv2.THRESH_BINARY)
            points = np.column_stack(np.where(binary_image > 0))
            plt.imshow(image, cmap='gray')
            plt.scatter(points[:, 1], points[:, 0], color='red', s=2)
            plt.title('Extrahierte Punkte')
            plt.show()
            initial_params = [
            0,0,0,1000
            ]
            calibration_function_code = f"""
import numpy as np
def calibration_function(x):
    points={points.tolist()}
    distances=0
    center=np.array([x[0],x[1],x[2]])
    for point in points:
        distance=np.linalg.norm(point - center[:,np.newaxis], axis=0)
        distances=distance+distances
    #f=0
    #for distanc in distances:
    #    f=distanc+f
    return distances-x[3]
                """
            data = {
            "points": points.tolist(),
            "calibration_function": calibration_function_code,
            "initial_params": initial_params,
            "additional_params": {
                "gtol":1e-3
                # Hier könnten später zusätzliche Parameter hinzugefügt werden
                }
            }
            self.data_manager.create_calibration_problem("shape_estimation_results", data,calibration_function_code)

        # Speichern Sie die Kalibrierungsfunktion in einer separaten Datei
        #with open('calibration_function.py', 'w') as f:
         #   f.write(calibration_function_code)

        return True
shape=Shapeestimation("camera_calibration_problems")
shape.estimateshape(r"C:\Users\fabia\Downloads\archive\data\imgs\rightcamera")
