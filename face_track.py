"""
Trans Rights are Human Rights

This is a script to handle reading a webcam, running MediaPipe FaceMesh, and doing some basic facial pose estimation
    on that returned FaceMesh data.
"""
# SYSTEM IMPORTS
import cv2
import mediapipe as mp
from mediapipe.python.solutions.drawing_utils import _normalized_to_pixel_coordinates as denormalize_coordinates
import time
import math
import numpy
import serial

# STANDARD LIBRARY IMPORTS

# LOCAL APPLICATION IMPORTS
from . import talker

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
except serial.SerialException:
    use_talker = False

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh


def timeit(method):
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()

        if 'log_time' in kw:
            name = kw.get('log_name', method.__name__.upper())
            kw['log_time'][name] = int((te - ts) * 1000)
        else:
            print('%r  %2.2f ms' % (method.__name__, (te - ts) * 1000))  # ew, need to rewrite this print
        return result

    return timed


def distance(xy1, xy2):
    dist = sum([(x - y) ** 2 for x, y in zip(xy1, xy2)]) ** 0.5
    return dist


def get_eye_ear_equation(lm: dict, reference_idxs: list, frame_width, frame_height) -> [float, list]:
    coordinate_points = []
    for idx in reference_idxs:
        coord = denormalize_coordinates(lm[idx][0], lm[idx][1], frame_width, frame_height)
        coordinate_points.append(coord)

    p2_p6 = distance(coordinate_points[1], coordinate_points[5])
    p3_p5 = distance(coordinate_points[2], coordinate_points[4])
    p1_p4 = distance(coordinate_points[0], coordinate_points[3])

    ear = (p2_p6 + p3_p5) / (2.0 * p1_p4)

    return ear


def distance_between(xyz_a: [int, int, int], xyz_b: [int, int, int]) -> float:
    dist = math.sqrt((xyz_b[0] - xyz_a[0]) ** 2 + (xyz_b[1] - xyz_b[1]) ** 2 + (xyz_b[2] - xyz_a[2]) ** 2)
    return dist


def remap_value(val: float, old_min: float, old_max: float, new_min: float, new_max: float):
    return (((val - old_min) * (new_max - new_min)) / (old_max - old_min)) + new_min


def clamp_float(val: float, min_val: float, max_val: float) -> object:
    return max(min(val, max_val), min_val)


ZERO_ONE_CLAMP = {"min_val": 0, "max_val": 1}
ZERO_ONE_REMAP_KWARGS = {"new_min": 0, "new_max": 1}

EYE_OPEN_REMAP_KWARGS = {"old_min": .07, "old_max": .30}
EYEBROW_INNER_REMAP_KWARGS = {"old_min": 15, "old_max": 17.5}
EYEBROW_MID_REMAP_KWARGS = {"old_min": 44, "old_max": 48}
EYE_IRIS_DISTANCE_REMAP_KWARGS = {"old_min": 35, "old_max": 48}
MOUTH_OPEN_REMAP_KWARGS = {"old_min": 0, "old_max": 11}
MOUTH_WIDE_REMAP_KWARGS = {"old_min": 33, "old_max": 75}


# @timeit
def pose_handler(lm: dict, frame_width: int, frame_height: int):

    def distance_with_normalize(xyz_a, xyz_b, norm_a, norm_b):
        world_open_amount = distance_between(xyz_a, xyz_b)
        scale_to_normalize_by = distance_between(norm_a, norm_b)
        open_normalized = (world_open_amount / scale_to_normalize_by) * 100  # x100 to get easier values to work with
        return open_normalized

    left_eye_open_amount_mapped = clamp_float(remap_value(
        get_eye_ear_equation(landmarks, LEFT_EYE_IDXS, frame_width=frame_width, frame_height=frame_height),
        **EYE_OPEN_REMAP_KWARGS, **ZERO_ONE_REMAP_KWARGS
     ), **ZERO_ONE_CLAMP)

    right_eye_open_amount_mapped = clamp_float(remap_value(
        get_eye_ear_equation(landmarks, RIGHT_EYE_IDXS, frame_width=frame_width, frame_height=frame_height),
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


FACEMESH_KWARGS = {"max_num_faces": 1,
                   "refine_landmarks": True,
                   "min_detection_confidence": 0.5,
                   "min_tracking_confidence": 0.5
                   }
show_image = True
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

if not cap.isOpened():
    raise Exception("Unable to read camera feed!!")

with mp_face_mesh.FaceMesh(**FACEMESH_KWARGS) as face_mesh:
    while cap.isOpened():
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

            # This should only be run if a COM device is attached and Talker can be run
            if use_talker:
                if pose_dict["eye_left"]["open_amount"] > .5:
                    talker_inst.send('show_image(shape="eye_static")')
                else:
                    talker_inst.send('show_image(shape="eye_blink")')

            img = numpy.zeros((image_height, image_width, 3), numpy.uint8)
            if show_image:
                for face_landmarks in results.multi_face_landmarks:
                    mp_drawing.draw_landmarks(
                        image=img,
                        landmark_list=face_landmarks,
                        connections=mp_face_mesh.FACEMESH_TESSELATION,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style())
                    mp_drawing.draw_landmarks(
                        image=img,
                        landmark_list=face_landmarks,
                        connections=mp_face_mesh.FACEMESH_CONTOURS,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style())
                    mp_drawing.draw_landmarks(
                        image=img,
                        landmark_list=face_landmarks,
                        connections=mp_face_mesh.FACEMESH_IRISES,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_iris_connections_style())

        # Flip the image horizontally for a selfie-view display.
        if show_image:
            cv2.imshow('MediaPipe Face Mesh', cv2.flip(image, 1))
            if cv2.waitKey(5) & 0xFF == 27:
                break

cap.release()
