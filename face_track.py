import cv2
import mediapipe as mp
import pprint
import time
import math
import talker


RIGHT_IRIS_INNER = 476
RIGHT_IRIS_OUTER = 474
RIGHT_IRIS_TOP = 475
RIGHT_IRIS_BOTTOM = 477
RIGHT_EYE_INNER = 362
RIGHT_EYE_OUTER = 263
RIGHT_EYE_TOP = 386
RIGHT_EYE_BOTTOM = 373

RIGHT_EYEBROW_INNER = 107
RIGHT_EYEBROW_MID = 105

LEFT_IRIS_INNER = 469
LEFT_IRIS_OUTER = 471
LEFT_IRIS_TOP = 470
LEFT_IRIS_BOTTOM = 472
LEFT_EYE_INNER = 243
LEFT_EYE_OUTER = 33
LEFT_EYE_TOP = 27
LEFT_EYE_BOTTOM = 23

LEFT_EYEBROW_INNER = 336
LEFT_EYEBROW_MID = 334

EYE_CENTRE_ON_NOSE = 6

MOUTH_MIDDLE_TOP = 13
MOUTH_MIDDLE_BOTTOM = 14
MOUTH_LEFT = 61
MOUTH_RIGHT = 291

talker_inst = talker.Talker()

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
            print('%r  %2.2f ms' % \
                  (method.__name__, (te - ts) * 1000))
        return result
    return timed


def distance_between(xyz_a: [int, int, int], xyz_b: [int, int, int]) -> float:
    distance = math.sqrt((xyz_b[0] - xyz_a[0]) ** 2 + (xyz_b[1] - xyz_b[1]) ** 2 + (xyz_b[2] - xyz_a[2]) ** 2)
    return distance


def remap_value(val: float, old_min: float, old_max: float, new_min: float, new_max: float):
    return (((val - old_min) * (new_max - new_min)) / (old_max - old_min)) + new_min


def clamp_float(val: float, min_val: float, max_val: float):
    return max(min(val, max_val), min_val)


# @timeit
def pose_handler(landmarks: dict):
    eye_map_val_kwargs = {"old_min": 0, "old_max": 20, "new_min": 0, "new_max": 1}
    zero_one_clamp = {"min_val": 0, "max_val": 1}

    debug_print_data = []

    def distance_with_normalize(xyz_a, xyz_b, norm_a, norm_b):
        world_open_amount = distance_between(xyz_a, xyz_b)
        scale_to_normalize_by = distance_between(norm_a, norm_b)
        open_normalized = (world_open_amount / scale_to_normalize_by) * 100  # x100 to get easier values to work with
        return open_normalized

    left_eye_open_amount_mapped = clamp_float(remap_value(
        distance_with_normalize(landmarks[LEFT_EYE_BOTTOM], landmarks[LEFT_EYE_TOP],
                                landmarks[LEFT_EYE_INNER], landmarks[LEFT_EYE_OUTER]),
        **eye_map_val_kwargs
    ), **zero_one_clamp)
    debug_print_data.append(["left eye open amount:             ", left_eye_open_amount_mapped])

    right_eye_open_amount_mapped = clamp_float(remap_value(
        distance_with_normalize(landmarks[RIGHT_EYE_BOTTOM], landmarks[RIGHT_EYE_TOP],
                                landmarks[RIGHT_EYE_INNER], landmarks[RIGHT_EYE_OUTER]),
        **eye_map_val_kwargs
    ), **zero_one_clamp)
    debug_print_data.append(["right eye open amount:            ", right_eye_open_amount_mapped])

    mouth_open_amount_mapped = clamp_float(remap_value(
        distance_with_normalize(landmarks[MOUTH_MIDDLE_TOP], landmarks[MOUTH_MIDDLE_BOTTOM],
                                landmarks[RIGHT_EYE_OUTER], landmarks[LEFT_EYE_OUTER]),
        0, 11, 0, 1
    ), **zero_one_clamp)
    debug_print_data.append(["mouth open amount:                ", mouth_open_amount_mapped])

    right_eyebrow_inner_amount_mapped = clamp_float(remap_value(
        distance_with_normalize(landmarks[RIGHT_EYEBROW_INNER], landmarks[EYE_CENTRE_ON_NOSE],
                                landmarks[RIGHT_EYE_OUTER], landmarks[LEFT_EYE_OUTER]),
        15, 17, 0, 1
    ), **zero_one_clamp)
    debug_print_data.append(["right eyebrow inner open amount:  ", right_eyebrow_inner_amount_mapped])

    right_eyebrow_mid_amount_mapped = clamp_float(remap_value(
        distance_with_normalize(landmarks[RIGHT_EYEBROW_MID], landmarks[EYE_CENTRE_ON_NOSE],
                                landmarks[RIGHT_EYE_OUTER], landmarks[LEFT_EYE_OUTER]),
        44, 47, 0, 1
    ), **zero_one_clamp)
    debug_print_data.append(["right eyebrow mid open amount:    ", right_eyebrow_mid_amount_mapped])

    pprint.pprint(debug_print_data)
    # print("##################################")

    return mouth_open_amount_mapped


# For webcam input:
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
cap = cv2.VideoCapture(0)
with mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as face_mesh:
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
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    image_height, image_width, _ = image.shape

    if results.multi_face_landmarks:
      landmarks = {}
      # print(results.multi_face_landmarks)
      for index, face_landmarks in enumerate(results.multi_face_landmarks[0].landmark):
        x_coordinate = face_landmarks.x
        y_coordinate = face_landmarks.y
        z_coordinate = face_landmarks.z
        current_landmarks = [x_coordinate, y_coordinate, z_coordinate]
        landmarks[index] = current_landmarks

      mouth_open_amount = pose_handler(landmarks)

      if mouth_open_amount > .7:
          talker_inst.send('show_mouth_shape(shape="open_wide")')
      elif mouth_open_amount < .3:
          talker_inst.send('show_mouth_shape(shape="closed")')
      else:
          talker_inst.send('show_mouth_shape(shape="open")')

      for face_landmarks in results.multi_face_landmarks:
        mp_drawing.draw_landmarks(
            image=image,
            landmark_list=face_landmarks,
            connections=mp_face_mesh.FACEMESH_TESSELATION,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp_drawing_styles
            .get_default_face_mesh_tesselation_style())
        mp_drawing.draw_landmarks(
            image=image,
            landmark_list=face_landmarks,
            connections=mp_face_mesh.FACEMESH_CONTOURS,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp_drawing_styles
            .get_default_face_mesh_contours_style())
        mp_drawing.draw_landmarks(
            image=image,
            landmark_list=face_landmarks,
            connections=mp_face_mesh.FACEMESH_IRISES,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp_drawing_styles
            .get_default_face_mesh_iris_connections_style())

      time.sleep(.1)
    # Flip the image horizontally for a selfie-view display.
    cv2.imshow('MediaPipe Face Mesh', cv2.flip(image, 1))
    if cv2.waitKey(5) & 0xFF == 27:
      break
cap.release()
