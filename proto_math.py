"""
Trans Rights are Human Rights

"""
# SYSTEM IMPORTS
import math

# STANDARD LIBRARY IMPORTS
from mediapipe.python.solutions.drawing_utils import _normalized_to_pixel_coordinates as denormalize_coordinates

# LOCAL APPLICATION IMPORTS


def distance(xy_a, xy_b) -> float:
    dist = sum([(x - y) ** 2 for x, y in zip(xy_a, xy_b)]) ** 0.5
    return dist


def distance_between(xyz_a: [int, int, int], xyz_b: [int, int, int]) -> float:
    dist = math.sqrt((xyz_b[0] - xyz_a[0]) ** 2 + (xyz_b[1] - xyz_b[1]) ** 2 + (xyz_b[2] - xyz_a[2]) ** 2)
    return dist


def remap_value(val: float, old_min: float, old_max: float, new_min: float, new_max: float) -> float:
    return (((val - old_min) * (new_max - new_min)) / (old_max - old_min)) + new_min


def clamp_float(val: float, min_val: float, max_val: float) -> object:
    return max(min(val, max_val), min_val)


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


def distance_with_normalize(xyz_a, xyz_b, norm_a, norm_b):
    world_open_amount = distance_between(xyz_a, xyz_b)
    scale_to_normalize_by = distance_between(norm_a, norm_b)
    open_normalized = (world_open_amount / scale_to_normalize_by) * 100  # x100 to get easier values to work with
    return open_normalized
