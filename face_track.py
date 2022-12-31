"""
Trans Rights are Human Rights

This is a script to handle reading a webcam, running MediaPipe FaceMesh, and doing some basic facial pose estimation
    on that returned FaceMesh data.
"""
# SYSTEM IMPORTS
import time
import cv2
import mediapipe as mp
import numpy

# STANDARD LIBRARY IMPORTS

# LOCAL APPLICATION IMPORTS
import talker
from proto_math import get_eye_ear_equation, remap_value, clamp_float, distance_with_normalize

RIGHT_IRIS_INNER = 476
RIGHT_IRIS_OUTER = 474
RIGHT_IRIS_TOP = 475
RIGHT_IRIS_BOTTOM = 477
RIGHT_IRIS_CENTRE = 473
RIGHT_EYE_INNER = 362
RIGHT_EYE_OUTER = 263
RIGHT_EYE_TOP = 386
RIGHT_EYE_BOTTOM = 374

RIGHT_EYEBROW_INNER = 107
RIGHT_EYEBROW_MID = 105

LEFT_IRIS_INNER = 469
LEFT_IRIS_OUTER = 471
LEFT_IRIS_TOP = 470
LEFT_IRIS_BOTTOM = 472
LEFT_IRIS_CENTRE = 468
LEFT_EYE_INNER = 133
LEFT_EYE_OUTER = 130
LEFT_EYE_TOP = 159
LEFT_EYE_BOTTOM = 145

LEFT_EYEBROW_INNER = 336
LEFT_EYEBROW_MID = 334

EYE_CENTRE_ON_NOSE = 6

MOUTH_MIDDLE_TOP = 13
MOUTH_MIDDLE_BOTTOM = 14
MOUTH_LEFT = 61
MOUTH_RIGHT = 291

# The eye open/close data uses a different equation, so the IDXs are supplied in a list format.
LEFT_EYE_IDXS = [362, 385, 387, 263, 373, 380]
RIGHT_EYE_IDXS = [33, 160, 158, 133, 153, 144]

# talker.Talker() will error out if a COM device isn't attached, this just bypasses it if need be.
use_talker = True
talker_inst = None
try:
    talker_inst = talker.Talker()
except:
    use_talker = False

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh

ZERO_ONE_CLAMP = {"min_val": 0, "max_val": 1}
ZERO_ONE_REMAP_KWARGS = {"new_min": 0, "new_max": 1}

EYE_OPEN_REMAP_KWARGS = {"old_min": .07, "old_max": .30}
EYEBROW_INNER_REMAP_KWARGS = {"old_min": 15, "old_max": 17.5}
EYEBROW_MID_REMAP_KWARGS = {"old_min": 44, "old_max": 48}
EYE_IRIS_DISTANCE_REMAP_KWARGS = {"old_min": 35, "old_max": 48}
MOUTH_OPEN_REMAP_KWARGS = {"old_min": 0, "old_max": 11}
MOUTH_WIDE_REMAP_KWARGS = {"old_min": 33, "old_max": 75}


# @decs.timeit
def pose_handler(lm: dict, frame_width: int, frame_height: int) -> dict:

    left_eye_open_amount_mapped = clamp_float(remap_value(
        get_eye_ear_equation(lm, LEFT_EYE_IDXS, frame_width=frame_width, frame_height=frame_height),
        **EYE_OPEN_REMAP_KWARGS, **ZERO_ONE_REMAP_KWARGS
     ), **ZERO_ONE_CLAMP)

    right_eye_open_amount_mapped = clamp_float(remap_value(
        get_eye_ear_equation(lm, RIGHT_EYE_IDXS, frame_width=frame_width, frame_height=frame_height),
        **EYE_OPEN_REMAP_KWARGS, **ZERO_ONE_REMAP_KWARGS
    ), **ZERO_ONE_CLAMP)

    mouth_open_amount_mapped = clamp_float(remap_value(
        distance_with_normalize(lm[MOUTH_MIDDLE_TOP], lm[MOUTH_MIDDLE_BOTTOM],
                                lm[RIGHT_EYE_OUTER], lm[LEFT_EYE_OUTER]),
        **MOUTH_OPEN_REMAP_KWARGS, **ZERO_ONE_REMAP_KWARGS
    ), **ZERO_ONE_CLAMP)

    mouth_wide_amount_mapped = clamp_float(remap_value(
        distance_with_normalize(lm[MOUTH_LEFT], lm[MOUTH_RIGHT],
                                lm[RIGHT_EYE_OUTER], lm[LEFT_EYE_OUTER]),
        **MOUTH_WIDE_REMAP_KWARGS, **ZERO_ONE_REMAP_KWARGS
    ), **ZERO_ONE_CLAMP)

    right_eyebrow_inner_amount_mapped = clamp_float(remap_value(
        distance_with_normalize(lm[RIGHT_EYEBROW_INNER], lm[EYE_CENTRE_ON_NOSE],
                                lm[RIGHT_EYE_OUTER], lm[LEFT_EYE_OUTER]),
        **EYEBROW_INNER_REMAP_KWARGS, **ZERO_ONE_REMAP_KWARGS
    ), **ZERO_ONE_CLAMP)

    right_eyebrow_mid_amount_mapped = clamp_float(remap_value(
        distance_with_normalize(lm[RIGHT_EYEBROW_MID], lm[EYE_CENTRE_ON_NOSE],
                                lm[RIGHT_EYE_OUTER], lm[LEFT_EYE_OUTER]),
        **EYEBROW_MID_REMAP_KWARGS, **ZERO_ONE_REMAP_KWARGS
    ), **ZERO_ONE_CLAMP)

    right_eye_iris_dist_from_nose_centre = clamp_float(remap_value(
        distance_with_normalize(lm[RIGHT_IRIS_CENTRE], lm[EYE_CENTRE_ON_NOSE],
                                lm[RIGHT_EYE_OUTER], lm[LEFT_EYE_OUTER]),
        **EYE_IRIS_DISTANCE_REMAP_KWARGS, **ZERO_ONE_REMAP_KWARGS
    ), **ZERO_ONE_CLAMP)

    left_eyebrow_inner_amount_mapped = clamp_float(remap_value(
        distance_with_normalize(lm[LEFT_EYEBROW_INNER], lm[EYE_CENTRE_ON_NOSE],
                                lm[RIGHT_EYE_OUTER], lm[LEFT_EYE_OUTER]),
        **EYEBROW_INNER_REMAP_KWARGS, **ZERO_ONE_REMAP_KWARGS
    ), **ZERO_ONE_CLAMP)

    left_eyebrow_mid_amount_mapped = clamp_float(remap_value(
        distance_with_normalize(lm[LEFT_EYEBROW_MID], lm[EYE_CENTRE_ON_NOSE],
                                lm[RIGHT_EYE_OUTER], lm[LEFT_EYE_OUTER]),
        **EYEBROW_MID_REMAP_KWARGS, **ZERO_ONE_REMAP_KWARGS
    ), **ZERO_ONE_CLAMP)

    left_eye_iris_dist_from_nose_centre = clamp_float(remap_value(
        distance_with_normalize(lm[LEFT_IRIS_CENTRE], lm[EYE_CENTRE_ON_NOSE],
                                lm[RIGHT_EYE_OUTER], lm[LEFT_EYE_OUTER]),
        **EYE_IRIS_DISTANCE_REMAP_KWARGS, **ZERO_ONE_REMAP_KWARGS
    ), **ZERO_ONE_CLAMP)

    print("l_eye:", f"{left_eye_open_amount_mapped:.2f}",
          "r_eye:", f"{right_eye_open_amount_mapped:.2f}",
          "mouth:", f"{mouth_open_amount_mapped:.2f}",
          "mouth_wide", f"{mouth_wide_amount_mapped:.2f}",
          "l_brow:", f"{left_eyebrow_inner_amount_mapped:.2f}",
          "r_brow:", f"{right_eyebrow_inner_amount_mapped:.2f}",
          "l_iris:", f"{left_eye_iris_dist_from_nose_centre:.2f}",
          "r_iris:", f"{right_eye_iris_dist_from_nose_centre:.2f}",
          end="\r")

    return_data = {
        "mouth": {
            "open_amount": mouth_open_amount_mapped
        },
        "eye_left": {
            "open_amount": left_eye_open_amount_mapped,
            "iris_distance": left_eye_iris_dist_from_nose_centre
        },
        "eye_right": {
            "open_amount": right_eye_open_amount_mapped,
            "iris_distance": right_eye_iris_dist_from_nose_centre
        },
        "eyebrow_left": {
            "mid_raise": left_eyebrow_mid_amount_mapped,
            "inner_raise": left_eyebrow_inner_amount_mapped
        },
        "eyebrow_right": {
            "mid_raise": right_eyebrow_mid_amount_mapped,
            "inner_raise": right_eyebrow_inner_amount_mapped
        }
    }

    return return_data


def face_direction_estimation(lm: dict, frame_width: int, frame_height: int):
    # TODO This needs fully rewriting to use my landmarks dict instead of results.multi_face_landmarks
    face_2d = []
    face_3d = []
    for face_landmarks in lm:
        for idx, lm in enumerate(face_landmarks.landmark):
            if idx == 33 or idx == 263 or idx == 1 or idx == 61 or idx == 291 or idx == 199:
                if idx == 1:
                    nose_2d = (lm.x * frame_width, lm.y * frame_height)
                    nose_3d = (lm.x * frame_width, lm.y * frame_height, lm.z * 3000)

                x, y = int(lm.x * frame_width), int(lm.y * frame_height)

                # Get the 2D Coordinates
                face_2d.append([x, y])

                # Get the 3D Coordinates
                face_3d.append([x, y, lm.z])

        # Convert it to the NumPy array
        face_2d = numpy.array(face_2d, dtype=numpy.float64)

        # Convert it to the NumPy array
        face_3d = numpy.array(face_3d, dtype=numpy.float64)

        # The camera matrix
        focal_length = 1 * frame_width

        cam_matrix = numpy.array([[focal_length, 0, frame_height / 2],
                               [0, focal_length, frame_width / 2],
                               [0, 0, 1]])

        # The distortion parameters
        dist_matrix = numpy.zeros((4, 1), dtype=numpy.float64)

        # Solve PnP
        success, rot_vec, trans_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)

        # Get rotational matrix
        rmat, jac = cv2.Rodrigues(rot_vec)

        # Get angles
        angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)

        # Get the y rotation degree
        x = angles[0] * 360
        y = angles[1] * 360
        z = angles[2] * 360

        nose_3d_projection, jacobian = cv2.projectPoints(nose_3d, rot_vec, trans_vec, cam_matrix, dist_matrix)

        p1 = (int(nose_2d[0]), int(nose_2d[1]))
        p2 = (int(nose_2d[0] + y * 10), int(nose_2d[1] - x * 10))

        return p1, p2


def run_face_tracking():
    FACEMESH_KWARGS = {"max_num_faces": 1,
                       "refine_landmarks": True,
                       "min_detection_confidence": 0.5,
                       "min_tracking_confidence": 0.5
                       }
    show_image = True
    drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
    cap = cv2.VideoCapture(0)
    # cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640/2)
    # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480/2)
    if not cap.isOpened():
        raise Exception("Unable to read camera feed!!")
    with mp_face_mesh.FaceMesh(**FACEMESH_KWARGS) as face_mesh:
        while cap.isOpened():
            start_time = time.time()
            success, image = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                # If loading a video, use 'break' instead of 'continue'.
                continue

            # To improve performance, optionally mark the image as not writeable to
            # pass by reference.
            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(image)

            # Draw the face mesh annotations on the image.
            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            image_height, image_width, _ = image.shape

            if results.multi_face_landmarks:
                landmarks = {}
                for index, face_landmarks in enumerate(results.multi_face_landmarks[0].landmark):
                    x_coordinate = face_landmarks.x
                    y_coordinate = face_landmarks.y
                    z_coordinate = face_landmarks.z
                    current_landmarks = [x_coordinate, y_coordinate, z_coordinate]
                    landmarks[index] = current_landmarks

                # Pass the modified landmarks dict into the posehandler, return the facial poses
                pose_dict = pose_handler(landmarks, frame_width=image_width, frame_height=image_height)

                # Head direction estimation
                # p1, p2 = face_direction_estimation(lm=results.multi_face_landmarks, frame_width=image_width, frame_height=image_height)
                # cv2.line(image, p1, p2, (255, 0, 0), 3)

                # This should only be run if a COM device is attached and Talker can be run
                if use_talker:
                    if pose_dict["eye_left"]["open_amount"] > .5:
                        talker_inst.send('show_image(shape="eye_static")')
                    else:
                        talker_inst.send('show_image(shape="eye_blink")')

                if show_image:
                    img = numpy.zeros((image_height, image_width, 3), numpy.uint8)
                    for face_landmarks in results.multi_face_landmarks:
                        mp_drawing.draw_landmarks(
                            image=image,
                            landmark_list=face_landmarks,
                            connections=mp_face_mesh.FACEMESH_TESSELATION,
                            landmark_drawing_spec=None,
                            connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style())
                        mp_drawing.draw_landmarks(
                            image=image,
                            landmark_list=face_landmarks,
                            connections=mp_face_mesh.FACEMESH_CONTOURS,
                            landmark_drawing_spec=None,
                            connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style())
                        mp_drawing.draw_landmarks(
                            image=image,
                            landmark_list=face_landmarks,
                            connections=mp_face_mesh.FACEMESH_IRISES,
                            landmark_drawing_spec=None,
                            connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_iris_connections_style())

            if show_image:
                end_time = time.time()
                fps = 1 / (end_time - start_time)
                cv2.putText(image, f'FPS: {int(fps)}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)

                cv2.imshow('MediaPipe Face Mesh', image)

                if cv2.waitKey(5) & 0xFF == 27:
                    break

    cap.release()

if __name__ == "__main__":
    run_face_tracking()
