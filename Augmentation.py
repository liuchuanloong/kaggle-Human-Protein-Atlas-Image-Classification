import numpy as np
import cv2
import math

def do_resize(image, SIZE):
    image = cv2.resize(image, dsize=(SIZE, SIZE))

    return image

#################################################################

# ---

def do_horizontal_flip(image):
    # flip left-right
    image = cv2.flip(image, 1)
    return image

def do_vertical_flip(image):
    # flip up-down
    image = cv2.flip(image, 0)
    return image

# ----
def do_invert_intensity(image):
    # flip left-right
    image = np.clip(1 - image, 0, 1)
    return image


def do_brightness_shift(image, alpha=0.125):
    image = image + alpha
    image = np.clip(image, 0, 1)
    return image


def do_brightness_multiply(image, alpha=1):
    image = alpha * image
    image = np.clip(image, 0, 1)
    return image


# https://www.pyimagesearch.com/2015/10/05/opencv-gamma-correction/
def do_gamma(image, gamma=1.0):
    image = image ** (1.0 / gamma)
    image = np.clip(image, 0, 1)
    return image


def do_flip_transpose(image, type=0):
    # choose one of the 8 cases

    if type == 0:  # rotate90
        image = image.transpose(1, 0, 2)
        image = cv2.flip(image, 1)

    if type == 1:  # rotate180
        image = cv2.flip(image, -1)

    if type == 2:  # rotate270
        image = image.transpose(1, 0, 2)
        image = cv2.flip(image, 0)

    if type == 3:  # flip left-right
        image = cv2.flip(image, 1)

    if type == 4:  # flip up-down
        image = cv2.flip(image, 0)

    if type == 5:
        image = cv2.flip(image, 1)
        image = image.transpose(1, 0, 2)
        image = cv2.flip(image, 1)

    if type == 6:
        image = cv2.flip(image, 0)
        image = image.transpose(1, 0, 2)
        image = cv2.flip(image, 1)

    return image

# ===========================================================================

def do_shift_scale_rotate(image, dx=0, dy=0, scale=1, angle=0):
    borderMode = cv2.BORDER_REFLECT_101
    # cv2.BORDER_REFLECT_101  cv2.BORDER_CONSTANT

    height, width = image.shape[:2]
    sx = scale
    sy = scale
    cc = math.cos(angle / 180 * math.pi) * (sx)
    ss = math.sin(angle / 180 * math.pi) * (sy)
    rotate_matrix = np.array([[cc, -ss], [ss, cc]])

    box0 = np.array([[0, 0], [width, 0], [width, height], [0, height], ], np.float32)
    box1 = box0 - np.array([width / 2, height / 2])
    box1 = np.dot(box1, rotate_matrix.T) + np.array([width / 2 + dx, height / 2 + dy])

    box0 = box0.astype(np.float32)
    box1 = box1.astype(np.float32)
    mat = cv2.getPerspectiveTransform(box0, box1)

    image = cv2.warpPerspective(image, mat, (width, height), flags=cv2.INTER_LINEAR,
                                borderMode=borderMode, borderValue=(
        0, 0, 0,))  # cv2.BORDER_CONSTANT, borderValue = (0, 0, 0))  #cv2.BORDER_REFLECT_101
    return image


# https://www.kaggle.com/ori226/data-augmentation-with-elastic-deformations
# https://github.com/letmaik/lensfunpy/blob/master/lensfunpy/util.py
def do_elastic_transform(image, grid=32, distort=0.2):
    borderMode = cv2.BORDER_REFLECT_101
    height, width = image.shape[:2]

    x_step = int(grid)
    xx = np.zeros(width, np.float32)
    prev = 0
    for x in range(0, width, x_step):
        start = x
        end = x + x_step
        if end > width:
            end = width
            cur = width
        else:
            cur = prev + x_step * (1 + np.random.uniform(-distort, distort))

        xx[start:end] = np.linspace(prev, cur, end - start)
        prev = cur

    y_step = int(grid)
    yy = np.zeros(height, np.float32)
    prev = 0
    for y in range(0, height, y_step):
        start = y
        end = y + y_step
        if end > height:
            end = height
            cur = height
        else:
            cur = prev + y_step * (1 + np.random.uniform(-distort, distort))

        yy[start:end] = np.linspace(prev, cur, end - start)
        prev = cur

    # grid
    map_x, map_y = np.meshgrid(xx, yy)
    map_x = map_x.astype(np.float32)
    map_y = map_y.astype(np.float32)

    # image = map_coordinates(image, coords, order=1, mode='reflect').reshape(shape)
    image = cv2.remap(image, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=borderMode,
                      borderValue=(0, 0, 0,))

    return image
