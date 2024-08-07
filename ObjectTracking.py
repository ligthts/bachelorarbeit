import os
import cv2
import numpy as np
from scipy.optimize import least_squares
from Dataadder import DataManager


class ObjectTracking:
    def __init__(self, directory):
        self.data_manager = DataManager(directory)

    def load_video(self, video_path):
        """
        Lädt ein Video aus einem angegebenen Pfad.

        :param video_path: Pfad zum Videodatei.
        :return: VideoCapture Objekt.
        """
        video_capture = cv2.VideoCapture(video_path)
        if not video_capture.isOpened():
            raise ValueError(f"Video konnte nicht geladen werden: {video_path}")
        return video_capture

    def detect_objects(self, frame, lower_color, upper_color):
        """
        Erkennt Objekte in einem einzelnen Frame basierend auf einer Farbgrenze.

        :param frame: Einzelner Frame des Videos.
        :param lower_color: Untere Farbgrenze im HSV-Raum.
        :param upper_color: Obere Farbgrenze im HSV-Raum.
        :return: Zentroid-Position des erkannten Objekts.
        """
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv_frame, lower_color, upper_color)

        # Finden Sie die Konturen im maskierten Bild
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            # Größte Kontur finden
            largest_contour = max(contours, key=cv2.contourArea)
            M = cv2.moments(largest_contour)
            if M['m00'] > 0:
                cX = int(M['m10'] / M['m00'])
                cY = int(M['m01'] / M['m00'])
                return np.array([cX, cY])
        return None

    def track_object(self, video_path, lower_color, upper_color, max_frames=100):
        """
        Verfolgt ein Objekt in einem Video und optimiert die Bewegungsbahn.

        :param video_path: Pfad zum Videodatei.
        :param lower_color: Untere Farbgrenze im HSV-Raum.
        :param upper_color: Obere Farbgrenze im HSV-Raum.
        :param max_frames: Maximale Anzahl der zu verarbeitenden Frames.
        :return: Optimierte Bewegungsbahn.
        """
        video_capture = self.load_video(video_path)
        centroids = []

        frame_count = 0
        while video_capture.isOpened() and frame_count < max_frames:
            ret, frame = video_capture.read()
            if not ret:
                break

            centroid = self.detect_objects(frame, lower_color, upper_color)
            if centroid is not None:
                centroids.append(centroid)
                cv2.circle(frame, (centroid[0], centroid[1]), 5, (0, 255, 0), -1)

            frame_count += 1

        video_capture.release()
        centroids = np.array(centroids)

        if centroids.shape[0] < 2:
            raise ValueError("Nicht genügend Punkte zur Optimierung gefunden.")

        # Optimierung der Bewegungsbahn
        def trajectory_error(params, points):
            a, b, c, d = params
            errors = points[:, 1] - (a * points[:, 0] ** 3 + b * points[:, 0] ** 2 + c * points[:, 0] + d)
            return errors

        initial_params = np.array([0, 0, 0, 0])
        result = least_squares(trajectory_error, initial_params, args=(centroids,))
        optimized_params = result.x

        return optimized_params, centroids

    def create_tracking_problem(self, video_path, lower_color, upper_color, max_frames=100):
        """
        Erstellt ein Tracking-Problem, speichert die Ergebnisse und die Funktion.

        :param video_path: Pfad zum Videodatei.
        :param lower_color: Untere Farbgrenze im HSV-Raum.
        :param upper_color: Obere Farbgrenze im HSV-Raum.
        :param max_frames: Maximale Anzahl der zu verarbeitenden Frames.
        :return: Optimierte Bewegungsbahn.
        """
        optimized_params, centroids = self.track_object(video_path, lower_color, upper_color, max_frames)

        # Debugging-Ausgaben
        print("Optimierte Parameter der Bewegungsbahn:", optimized_params)

        # Erstellen der Funktion als ausführbaren Code
        tracking_function_code = f"""
import numpy as np

def tracking_function(x, points):
    a, b, c, d = x
    errors = points[:, 1] - (a * points[:, 0]**3 + b * points[:, 0]**2 + c * points[:, 0] + d)
    return errors

def get_trajectory_error(params, centroids):
    return tracking_function(params, centroids)
        """

        data = {
            "video_path": video_path,
            "lower_color": lower_color.tolist(),
            "upper_color": upper_color.tolist(),
            "max_frames": max_frames,
            "optimized_params": optimized_params.tolist(),
            "centroids": centroids.tolist(),
            "tracking_function": tracking_function_code,
            "initial_params": [0, 0, 0, 0]
        }

        # Speichern der Ergebnisse
        self.data_manager.create_calibration_problem("object_tracking_results", data, tracking_function_code)

        # Speichern Sie die Tracking-Funktion in einer separaten Datei
        unique_filename = os.path.join('object_tracking_results',
                                       f'tracking_function_{os.path.basename(video_path)}.py')
        with open(unique_filename, 'w') as f:
            f.write(tracking_function_code)

        return optimized_params


# Beispiel für die Verwendung der Klasse
if __name__ == "__main__":
    directory = "data_directory"  # Das Verzeichnis, in dem die Ergebnisse gespeichert werden
    object_tracker = ObjectTracking(directory)

    video_path = "path_to_video.mp4"  # Pfad zu einem Beispielvideo
    lower_color = np.array([30, 100, 100])  # Untere Grenze für die Farbmaske im HSV
    upper_color = np.array([80, 255, 255])  # Obere Grenze für die Farbmaske im HSV

    # Erstelle ein Tracking-Problem und speichere die Ergebnisse
    optimized_params = object_tracker.create_tracking_problem(video_path, lower_color, upper_color)

    print("Optimierte Bewegungsbahnparameter:", optimized_params)
