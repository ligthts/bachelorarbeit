import cv2
import numpy as np
import glob
import os


def calibrate_camera(images, chessboard_size):
    objp = np.zeros((np.prod(chessboard_size), 3), np.float32)
    objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)

    objpoints = []
    imgpoints = []

    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)
        if ret:
            objpoints.append(objp)
            imgpoints.append(corners)
            cv2.drawChessboardCorners(img, chessboard_size, corners, ret)
            cv2.imshow('img', img)
            cv2.waitKey(500)

    cv2.destroyAllWindows()

    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None,
                                                                        None)
    return camera_matrix, dist_coeffs, rvecs, tvecs


def create_optimization_problem(camera_matrix, dist_coeffs, degree=5, num_points=100):
    x_data = np.linspace(0, 10, num_points)

    if len(dist_coeffs) < degree + 1:
        raise ValueError("Distortion coefficients must have at least degree + 1 elements")

    y_data = np.polynomial.polynomial.polyval(x_data, dist_coeffs[:degree + 1])

    initial_params = np.zeros(degree + 1)

    return {
        'x': x_data.tolist(),
        'y': y_data.tolist(),
        'initial_params': initial_params.tolist(),
        'residuals': lambda params, x, y: np.polynomial.polynomial.polyval(x, params) - y,
        'additional_params': {
            'method': 'lm'
        }
    }


if __name__ == "__main__":
    chessboard_size = (9, 6)
    images = glob.glob('calibration_images/*.jpg')
    camera_matrix, dist_coeffs, rvecs, tvecs = calibrate_camera(images, chessboard_size)

    try:
        optimization_problem = create_optimization_problem(camera_matrix, dist_coeffs, degree=10, num_points=100)
        print("Optimization problem created successfully.")
    except ValueError as e:
        print(f"Error: {e}")
