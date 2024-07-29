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

        # Debugging-Ausgaben
        print("Kamera-Matrix:", camera_matrix)
        print("Verzerrungskoeffizienten:", dist_coeffs)

        # Überprüfe die Dimensionen der Kamera-Matrix und Verzerrungskoeffizienten
        if camera_matrix.shape != (3, 3):
            raise ValueError("camera_matrix hat nicht die erwartete Form (3, 3)")

        dist_coeffs = dist_coeffs.flatten()  # Wandle dist_coeffs in ein eindimensionales Array um
        dist_coeffs_len = len(dist_coeffs)
        print("Länge der Verzerrungskoeffizienten:", dist_coeffs_len)

        if dist_coeffs_len not in [4, 5, 8, 12, 14]:
            raise ValueError(f"dist_coeffs hat unerwartete Länge: {dist_coeffs_len}. Erwartet: 4, 5, 8, 12 oder 14.")

        # Erstelle die Kalibrierungsfunktion als ausführbaren Code
        calibration_function_code =  """
def calibration_function(x):
    import numpy as np
    import cv2

    # Extrahiere Kamera-Matrix und Verzerrungskoeffizienten
    fx, fy, cx, cy = x[0:4]
    k1, k2, p1, p2, k3 = x[4:9]
    camera_matrix = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
    dist_coeffs = np.array([k1, k2, p1, p2, k3])

    # Extrahiere restliche Parameter
    num_obj_points = int(x[9])
    obj_points_flat = x[10:10 + 3 * num_obj_points]
    img_points_flat = x[10 + 3 * num_obj_points:10 + 3 * num_obj_points + 2 * num_obj_points]
    rvecs_flat = x[10 + 3 * num_obj_points + 2 * num_obj_points:10 + 3 * num_obj_points + 2 * num_obj_points + 3 * num_obj_points]
    tvecs_flat = x[10 + 3 * num_obj_points + 2 * num_obj_points + 3 * num_obj_points:]

    obj_points = [np.array(obj_points_flat[i:i + 3]).reshape(-1, 3) for i in range(0, len(obj_points_flat), 3)]
    img_points = [np.array(img_points_flat[i:i + 2]).reshape(-1, 2) for i in range(0, len(img_points_flat), 2)]
    rvecs = [np.array(rvecs_flat[i:i + 3]) for i in range(0, len(rvecs_flat), 3)]
    tvecs = [np.array(tvecs_flat[i:i + 3]) for i in range(0, len(tvecs_flat), 3)]

    # Berechne den Reprojektion Fehler
    total_error = 0
    for i in range(len(obj_points)):
        img_points_proj, _ = cv2.projectPoints(obj_points[i], rvecs[i], tvecs[i], camera_matrix, dist_coeffs)
        img_points_proj = img_points_proj.reshape(-1, 2)
        error = cv2.norm(img_points[i], img_points_proj, cv2.NORM_L2) / len(img_points_proj)
        total_error += error

    return total_error
"""
        # Flatten und Speichern der Kalibrierungsergebnisse
        obj_points_flat = np.concatenate([p.flatten() for p in obj_points])
        img_points_flat = np.concatenate([p.flatten() for p in img_points])
        rvecs_flat = np.concatenate([r.flatten() for r in rvecs])
        tvecs_flat = np.concatenate([t.flatten() for t in tvecs])
        initial_params = [
            float(camera_matrix[0, 0]), float(camera_matrix[1, 1]), float(camera_matrix[0, 2]),
            float(camera_matrix[1, 2]), *dist_coeffs.tolist(),  # Hier passen wir die Verzerrungskoeffizienten an
            len(obj_points),
            *obj_points_flat.tolist(),
            *img_points_flat.tolist(),
            *rvecs_flat.tolist(),
            *tvecs_flat.tolist()
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



