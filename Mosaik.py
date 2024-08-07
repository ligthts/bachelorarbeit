# ImageMosaicing.py

import os
import cv2
import numpy as np
from Dataadder import DataManager
import matplotlib.pyplot as plt
from scipy.optimize import least_squares

class ImageMosaicing:
    def __init__(self, directory):
        self.data_manager = DataManager(directory)

    def load_images_from_directory(self, image_directory):
        """
        Lädt alle Bilder aus einem angegebenen Verzeichnis.

        :param image_directory: Pfad zum Bildverzeichnis.
        :return: Liste von geladenen Bildern.
        """
        images = []
        for filename in sorted(os.listdir(image_directory)):  # Achte auf die Reihenfolge der Bilder
            if filename.endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(image_directory, filename)
                img = cv2.imread(img_path)
                if img is not None:
                    images.append(img)
        return images

    def find_keypoints_and_descriptors(self, images):
        """
        Findet Schlüsselmerkmale und deren Deskriptoren in den gegebenen Bildern.

        :param images: Liste von Bildern.
        :return: Liste von Schlüsselmerkmalen und Deskriptoren.
        """
        sift = cv2.SIFT_create()  # Erstelle ein SIFT-Objekt
        keypoints_and_descriptors = []

        for img in images:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            keypoints, descriptors = sift.detectAndCompute(gray, None)
            keypoints_and_descriptors.append((keypoints, descriptors))

        return keypoints_and_descriptors

    def match_features(self, descriptors1, descriptors2):
        """
        Vergleicht Merkmale zwischen zwei Bilddeskriptoren und findet Übereinstimmungen.

        :param descriptors1: Deskriptoren des ersten Bildes.
        :param descriptors2: Deskriptoren des zweiten Bildes.
        :return: Liste von Übereinstimmungen.
        """
        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
        matches = bf.match(descriptors1, descriptors2)
        matches = sorted(matches, key=lambda x: x.distance)  # Sortiere nach Abstand
        return matches

    def estimate_homography(self, keypoints1, keypoints2, matches):
        """
        Schätzt die Homographie-Matrix zwischen zwei Bildern basierend auf den Merkmalübereinstimmungen.

        :param keypoints1: Schlüsselmerkmale des ersten Bildes.
        :param keypoints2: Schlüsselmerkmale des zweiten Bildes.
        :param matches: Liste von Merkmalübereinstimmungen.
        :return: Homographie-Matrix.
        """
        points1 = np.float32([keypoints1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        points2 = np.float32([keypoints2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

        H, _ = cv2.findHomography(points1, points2, cv2.RANSAC)
        return H

    def warp_image(self, image, H, output_shape):
        """
        Wendet eine Homographie auf ein Bild an und gibt das transformierte Bild zurück.

        :param image: Eingabebild.
        :param H: Homographie-Matrix.
        :param output_shape: Ausgabeform des transformierten Bildes.
        :return: Transformiertes Bild.
        """
        warped_image = cv2.warpPerspective(image, H, output_shape)
        return warped_image

    def create_mosaic(self, images, homographies):
        """
        Erstellt ein Mosaik aus einer Liste von Bildern und den dazugehörigen Homographien.

        :param images: Liste von Bildern.
        :param homographies: Liste von Homographie-Matrizen.
        :return: Erstelltes Mosaikbild.
        """
        # Bestimme die Größe des Ausgabemosaiks
        height, width = images[0].shape[:2]
        mosaic = np.zeros((height * 2, width * 2, 3), dtype=np.uint8)

        # Bestimme den mittleren Index für den Ausgangspunkt
        mid_index = len(images) // 2
        mosaic[height // 2:height // 2 + height, width // 2:width // 2 + width] = images[mid_index]

        # Iteriere über alle Bilder
        for i, (image, H) in enumerate(zip(images, homographies)):
            if i != mid_index:
                # Verwende die Homographie, um das Bild in den Mosaikbereich zu projizieren
                warped_image = self.warp_image(image, H, (mosaic.shape[1], mosaic.shape[0]))
                mask = (warped_image > 0)
                mosaic[mask] = warped_image[mask]

        return mosaic

    def optimize_transformations(self, images, initial_params):
        """
        Optimiert die Transformationsparameter zwischen Bildern, um den Überlappungsfehler zu minimieren.

        :param images: Liste von Bildern.
        :param initial_params: Anfangsparameter der Optimierung.
        :return: Optimierte Transformationsparameter.
        """
        keypoints_and_descriptors = self.find_keypoints_and_descriptors(images)

        # Initialisiere eine Liste der Matches zwischen den Bildern
        matches = []
        for i in range(len(images) - 1):
            keypoints1, descriptors1 = keypoints_and_descriptors[i]
            keypoints2, descriptors2 = keypoints_and_descriptors[i + 1]
            match = self.match_features(descriptors1, descriptors2)
            matches.append((keypoints1, keypoints2, match))

        def calibration_function(x):
            """
            Kalibrierungsfunktion zur Optimierung der Transformationsparameter.

            :param x: Zu optimierende Parameter (z.B. Translations- und Rotationsparameter).
            :return: Fehler der aktuellen Parameter.
            """
            total_error = 0
            current_homography = np.eye(3)

            for i, (keypoints1, keypoints2, match) in enumerate(matches):
                # Berechne die aktuelle Transformationsmatrix aus den Parametern
                dx, dy, da = x[3 * i:3 * (i + 1)]
                transformation_matrix = np.array([
                    [np.cos(da), -np.sin(da), dx],
                    [np.sin(da), np.cos(da), dy],
                    [0, 0, 1]
                ])

                # Aktualisiere die aktuelle Homographie
                current_homography = current_homography @ transformation_matrix

                # Projiziere die Punkte des ersten Bildes und berechne den Fehler
                points1 = np.float32([keypoints1[m.queryIdx].pt for m in match]).reshape(-1, 1, 2)
                points2 = np.float32([keypoints2[m.trainIdx].pt for m in match]).reshape(-1, 1, 2)

                projected_points = cv2.perspectiveTransform(points1, current_homography)
                error = np.linalg.norm(projected_points - points2, axis=2)
                total_error += np.sum(error)

            return total_error

        # Optimierung der Transformationsparameter mit SciPy
        result = least_squares(
            calibration_function,
            initial_params,
            gtol=1e-4,
            xtol=1e-4,
            ftol=1e-4
        )
        return result.x

    def estimate_mosaic(self, image_directory):
        """
        Führt die Bildmosaikschätzung durch und speichert die Ergebnisse.

        :param image_directory: Verzeichnis mit den Mosaikbildern.
        :return: Erfolgsstatus der Mosaikschätzung.
        """
        images = self.load_images_from_directory(image_directory)
        keypoints_and_descriptors = self.find_keypoints_and_descriptors(images)

        # Initialisiere eine Liste der Matches zwischen den Bildern
        matches = []
        for i in range(len(images) - 1):
            keypoints1, descriptors1 = keypoints_and_descriptors[i]
            keypoints2, descriptors2 = keypoints_and_descriptors[i + 1]
            match = self.match_features(descriptors1, descriptors2)
            matches.append((keypoints1, keypoints2, match))
        initial_params = np.zeros(len(images) * 3 - 3)  # [dx1, dy1, da1, dx2, dy2, da2, ...]

        # Optimiere die Transformationsparameter
        optimized_params = self.optimize_transformations(images, initial_params)

        # Erzeuge Homographien aus den optimierten Parametern
        homographies = [np.eye(3)]
        current_homography = np.eye(3)

        for i in range(len(images) - 1):
            dx, dy, da = optimized_params[3 * i:3 * (i + 1)]
            transformation_matrix = np.array([
                [np.cos(da), -np.sin(da), dx],
                [np.sin(da), np.cos(da), dy],
                [0, 0, 1]
            ])
            current_homography = current_homography @ transformation_matrix
            homographies.append(current_homography)

        # Erstelle das Mosaik
        mosaic = self.create_mosaic(images, homographies)

        # Zeige das Mosaik an
        plt.imshow(cv2.cvtColor(mosaic, cv2.COLOR_BGR2RGB))
        plt.title('Erstelltes Mosaik')
        plt.axis('off')
        plt.show()

        # Speichere das Ergebnis
        calibration_function_code = f"""
import numpy as np

def calibration_function(x):
    # Optimierung der Transformationsparameter für das Mosaik
    # x sind die zu optimierenden Parameter (Translations- und Rotationsparameter)
    # Berechne den Gesamtfehler für die aktuelle Parameterauswahl
    matches=({matches})
    total_error = 0
    current_homography = np.eye(3)

    for i, (keypoints1, keypoints2, match) in enumerate(matches):
        dx, dy, da = x[3 * i:3 * (i + 1)]
        transformation_matrix = np.array([
            [np.cos(da), -np.sin(da), dx],
            [np.sin(da), np.cos(da), dy],
            [0, 0, 1]
        ])
        current_homography = current_homography @ transformation_matrix
        points1 = np.float32([keypoints1[m.queryIdx].pt for m in match]).reshape(-1, 1, 2)
        points2 = np.float32([keypoints2[m.trainIdx].pt for m in match]).reshape(-1, 1, 2)
        projected_points, _ = cv2.perspectiveTransform(points1, current_homography)
        error = np.linalg.norm(projected_points - points2, axis=2)
        total_error += np.sum(error)

    return total_error
"""

        data = {
            "mosaic": mosaic.tolist(),
            "calibration_function": calibration_function_code,
            "initial_params": initial_params.tolist(),
            "optimized_params": optimized_params.tolist(),
            "additional_params": {
                "gtol": 1e-3
            }
        }
        self.data_manager.create_calibration_problem("mosaic_estimation_results", data, calibration_function_code)

        return True


# Beispielverwendung
mosaicing = ImageMosaicing("image_mosaicing_problems")
mosaicing.estimate_mosaic(r"C:\Users\fabia\Downloads\archive\data\imgs\rightcamera")
