"""
Trans Rights are Human Rights

Small script to convert an RGB image to a 2 axis RGBA array
"""
# SYSTEM IMPORTS
from PIL import Image
import pathlib

# STANDARD LIBRARY IMPORTS

# LOCAL APPLICATION IMPORTS


def get_pixels(image_name) -> list:
    img = Image.open(image_name, 'r')
    w, h = img.size
    pix = list(img.getdata())

    return [pix[n:n+w] for n in range(0, w*h, w)]


for file_image_name in ["eye_static", "eye_backward", "eye_forward", "eye_blink", "eye_angry",
                        "mouth_closed", "mouth_open", "mouth_open_wide"]:
    print(f"{file_image_name} =",
          get_pixels(pathlib.Path(pathlib.Path(__file__).parents[0], "assets", f"{file_image_name}.png")))
