import os
import cv2
import numpy as np
from Dataadder import DataManager


class CameraCalibration:
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

    def calibrate_camera(self, image_directory, chessboard_size):
        """
        Kalibriert die Kamera mit Bildern aus einem Verzeichnis und speichert die Kalibrierungsfunktion.

        :param image_directory: Verzeichnis, das die Kalibrierungsbilder enthält.
        :param chessboard_size: Anzahl der inneren Ecken des Schachbretts (width, height).
        :return: Kameramatrix und Verzerrungskoeffizienten.
        """
        images = self.load_images_from_directory(image_directory)
        obj_points, img_points = self.find_chessboard_corners(images, chessboard_size)

        # Kalibriere die Kamera
        ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points,
                                                                            images[0].shape[1::-1], None, None)

        # Erstelle die Kalibrierungsfunktion als ausführbaren Code
        calibration_function_code = f"""
def calibration_function(x):
    # x = [fx, fy, cx, cy, k1, k2, p1, p2, k3]
    camera_matrix = np.array([[x[0], 0, x[2]], [0, x[1], x[3]], [0, 0, 1]])
    dist_coeffs = np.array([x[4], x[5], x[6], x[7], x[8]])
    # Berechne reprojection error
    total_error = 0
    for i in range(len(obj_points)):
        img_points_proj, _ = cv2.projectPoints(obj_points[i], rvecs[i], tvecs[i], camera_matrix, dist_coeffs)
        error = cv2.norm(img_points[i], img_points_proj, cv2.NORM_L2) / len(img_points_proj)
        total_error += error
    return total_error
"""
        # Speichern der Kalibrierungsergebnisse
        initial_params = [
            camera_matrix[0, 0], camera_matrix[1, 1], camera_matrix[0, 2],
            camera_matrix[1, 2], dist_coeffs[0, 0], dist_coeffs[0, 1],
            dist_coeffs[0, 2], dist_coeffs[0, 3], dist_coeffs[0, 4]
        ]

        data = {
            "camera_matrix": camera_matrix.tolist(),
            "dist_coeffs": dist_coeffs.tolist(),
            "calibration_function": calibration_function_code,
            "initial_params": initial_params,
            "additional_params": {
                # Hier könnten später zusätzliche Parameter hinzugefügt werden
            }
        }
        self.data_manager.add_data("camera_calibration_results", data)

        return camera_matrix, dist_coeffs

