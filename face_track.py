"""
Trans Rights are Human Rights

"""
# SYSTEM IMPORTS
import cv2
import mediapipe as mp
import pprint
import time
import math
import serial

# STANDARD LIBRARY IMPORTS

# LOCAL APPLICATION IMPORTS
import talker

RIGHT_IRIS_INNER = 476
RIGHT_IRIS_OUTER = 474
RIGHT_IRIS_TOP = 475
RIGHT_IRIS_BOTTOM = 477
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


def distance_between(xyz_a: [int, int, int], xyz_b: [int, int, int]) -> float:
    distance = math.sqrt((xyz_b[0] - xyz_a[0]) ** 2 + (xyz_b[1] - xyz_b[1]) ** 2 + (xyz_b[2] - xyz_a[2]) ** 2)
    return distance


def remap_value(val: float, old_min: float, old_max: float, new_min: float, new_max: float):
    return (((val - old_min) * (new_max - new_min)) / (old_max - old_min)) + new_min


def clamp_float(val: float, min_val: float, max_val: float):
    return max(min(val, max_val), min_val)


zero_one_clamp = {"min_val": 0, "max_val": 1}
ZERO_ONE_REMAP_KWARGS = {"new_min": 0, "new_max": 1}

EYE_OPEN_REMAP_KWARGS = {"old_min": 0, "old_max": 20}
EYEBROW_INNER_REMAP_KWARGS = {"old_min": 15, "old_max": 17.5}

# @timeit
def pose_handler(lm: dict):

    debug_print_data = []

    def distance_with_normalize(xyz_a, xyz_b, norm_a, norm_b):
        world_open_amount = distance_between(xyz_a, xyz_b)
        scale_to_normalize_by = distance_between(norm_a, norm_b)
        open_normalized = (world_open_amount / scale_to_normalize_by) * 100  # x100 to get easier values to work with
        return open_normalized

    left_eye_open_amount_mapped = clamp_float(remap_value(
        distance_with_normalize(lm[LEFT_EYE_TOP], lm[LEFT_EYE_BOTTOM],
                                lm[RIGHT_EYE_OUTER], lm[LEFT_EYE_OUTER]),
        **EYE_OPEN_REMAP_KWARGS, **ZERO_ONE_REMAP_KWARGS
    ), **zero_one_clamp)

    right_eye_open_amount_mapped = clamp_float(remap_value(
        distance_with_normalize(lm[RIGHT_EYE_TOP], lm[RIGHT_EYE_BOTTOM],
                                lm[RIGHT_EYE_INNER], lm[LEFT_EYE_OUTER]),
        **EYE_OPEN_REMAP_KWARGS, **ZERO_ONE_REMAP_KWARGS
    ), **zero_one_clamp)

    mouth_open_amount_mapped = clamp_float(remap_value(
        distance_with_normalize(lm[MOUTH_MIDDLE_TOP], lm[MOUTH_MIDDLE_BOTTOM],
                                lm[RIGHT_EYE_OUTER], lm[LEFT_EYE_OUTER]),
        0, 11, **ZERO_ONE_REMAP_KWARGS
    ), **zero_one_clamp)

    right_eyebrow_inner_amount_mapped = clamp_float(remap_value(
        distance_with_normalize(lm[RIGHT_EYEBROW_INNER], lm[EYE_CENTRE_ON_NOSE],
                                lm[RIGHT_EYE_OUTER], lm[LEFT_EYE_OUTER]),
        15, 17.5, **ZERO_ONE_REMAP_KWARGS
    ), **zero_one_clamp)

    right_eyebrow_mid_amount_mapped = clamp_float(remap_value(
        distance_with_normalize(lm[RIGHT_EYEBROW_MID], lm[EYE_CENTRE_ON_NOSE],
                                lm[RIGHT_EYE_OUTER], lm[LEFT_EYE_OUTER]),
        44, 48, **ZERO_ONE_REMAP_KWARGS
    ), **zero_one_clamp)

    left_eyebrow_inner_amount_mapped = clamp_float(remap_value(
        distance_with_normalize(lm[LEFT_EYEBROW_INNER], lm[EYE_CENTRE_ON_NOSE],
                                lm[RIGHT_EYE_OUTER], lm[LEFT_EYE_OUTER]),
        15.5, 16.5, **ZERO_ONE_REMAP_KWARGS
    ), **zero_one_clamp)

    left_eyebrow_mid_amount_mapped = clamp_float(remap_value(
        distance_with_normalize(lm[LEFT_EYEBROW_MID], lm[EYE_CENTRE_ON_NOSE],
                                lm[RIGHT_EYE_OUTER], lm[LEFT_EYE_OUTER]),
        44, 48, **ZERO_ONE_REMAP_KWARGS
    ), **zero_one_clamp)

    # print(f"{left_eye_open_amount_mapped:.2f}  ",
    #       f"{left_eyebrow_inner_amount_mapped:.2f}  ",
    #       f"{mouth_open_amount_mapped:.2f} ",
    #       end="\r")

    # print(f"{lm[LEFT_EYE_TOP]}:.2f", f"{lm[LEFT_EYE_BOTTOM]}:.2f", end="\r")
    print("left eyebrow inner up:   ", f"{right_eyebrow_inner_amount_mapped:.2f}", end="\r")

    return_data = {
        "mouth": {
            "open_amount": mouth_open_amount_mapped
        },
        "eye_left": {
            "open_amount": left_eye_open_amount_mapped
        },
        "eye_right": {
            "open_amount": right_eye_open_amount_mapped
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
show_image = False
# For webcam input:
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
cap = cv2.VideoCapture(0)
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
            pose_dict = pose_handler(landmarks)

            # This should only be run if a COM device is attached and Talker can be run
            if use_talker:
                if pose_dict["mouth"]["open_amount"] > .7:
                    talker_inst.send('show_image(shape="open_wide")')
                elif pose_dict["mouth"]["open_amount"] < .3:
                    talker_inst.send('show_image(shape="closed")')
                else:
                    talker_inst.send('show_image(shape="open")')

                # if pose_dict["eyebrow_right"]["inner_raise"] > .8:
                #     talker_inst.send('show_image(shape="open_wide")')
                # elif pose_dict["eyebrow_right"]["inner_raise"] < .2:
                #     talker_inst.send('show_image(shape="closed")')
                # else:
                #     talker_inst.send('show_image(shape="open")')
            if show_image:
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

        # time.sleep(.1)
        # Flip the image horizontally for a selfie-view display.
        if show_image:
            cv2.imshow('MediaPipe Face Mesh', cv2.flip(image, 1))
            if cv2.waitKey(5) & 0xFF == 27:
                break

cap.release()
