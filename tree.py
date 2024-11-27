import numpy as np
from PIL import Image, ImageDraw
import time
from functools import wraps


def timing_decorator(func):
    """
    Декоратор для измерения времени выполнения функции.
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()  # Начало отсчета времени
        result = func(*args, **kwargs)
        end_time = time.perf_counter()  # Конец отсчета времени
        elapsed_time = end_time - start_time
        print(f"Function '{func.__name__}' executed in {elapsed_time:.4f} seconds")
        return result

    return wrapper


MAX_DEPTH = 10
DETAIL_THRESHOLD = 13
SIZE_MULT = 1


def average_colour(image):
    # convert image to np array
    image_arr = np.asarray(image)

    # get average of whole image
    avg_color_per_row = np.average(image_arr, axis=0)
    avg_color = np.average(avg_color_per_row, axis=0)

    return int(avg_color[0]), int(avg_color[1]), int(avg_color[2])


def weighted_average(hist):
    total = sum(hist)
    error = value = 0

    if total > 0:
        value = sum(i * x for i, x in enumerate(hist)) / total
        error = sum(x * (value - i) ** 2 for i, x in enumerate(hist)) / total
        error = error ** 0.5

    return error


def get_detail(hist):
    red_detail = weighted_average(hist[:256])
    green_detail = weighted_average(hist[256:512])
    blue_detail = weighted_average(hist[512:768])

    detail_intensity = red_detail * 0.2989 + green_detail * 0.5870 + blue_detail * 0.1140

    return detail_intensity


class Quadrant:
    def __init__(self, image, bbox, depth):
        self.image_q = image
        self.bbox = bbox
        self.depth = depth
        self.children = None
        self.leaf = False

        # crop image to quadrant size
        image = image.crop(bbox)
        hist = image.histogram()

        self.detail = get_detail(hist)
        self.colour = average_colour(image)

    def split_quadrant(self):
        left, top, width, height = self.bbox

        # get the middle coords of bbox
        middle_x = left + (width - left) / 2
        middle_y = top + (height - top) / 2

        # split root quadrant into 4 new quadrants
        upper_left = Quadrant(self.image_q, (left, top, middle_x, middle_y), self.depth + 1)
        upper_right = Quadrant(self.image_q, (middle_x, top, width, middle_y), self.depth + 1)
        bottom_left = Quadrant(self.image_q, (left, middle_y, middle_x, height), self.depth + 1)
        bottom_right = Quadrant(self.image_q, (middle_x, middle_y, width, height), self.depth + 1)

        # add new quadrants to root children
        self.children = [upper_left, upper_right, bottom_left, bottom_right]


class QuadTree:
    def __init__(self, image):
        self.width, self.height = image.size
        self.image = image
        # keep track of max depth achieved by recursion
        self.max_depth = 0

        # start compression
        self.root = Quadrant(self.image, self.image.getbbox(), 0)
        self.build(self.root)

    def build(self, root):
        if root.depth >= MAX_DEPTH or root.detail <= DETAIL_THRESHOLD:
            if root.depth > self.max_depth:
                self.max_depth = root.depth

            # assign quadrant to leaf and stop recursing
            root.leaf = True
            return

            # split quadrant if there is too much detail
        root.split_quadrant()

        for children in root.children:
            self.build(children)

    def create_image(self, custom_depth, show_lines=False):
        # create blank image canvas
        new_image = Image.new('RGB', (self.width, self.height))
        draw = ImageDraw.Draw(new_image)
        draw.rectangle((0, 0, self.width, self.height), (0, 0, 0))

        leaf_quadrants = self.get_leaf_quadrants(custom_depth)

        # draw rectangle size of quadrant for each leaf quadrant
        for quadrant in leaf_quadrants:
            if show_lines:
                draw.rectangle(quadrant.bbox, quadrant.colour, outline=(0, 0, 0))
            else:
                draw.rectangle(quadrant.bbox, quadrant.colour)

        return new_image

    def get_leaf_quadrants(self, depth):
        if depth > self.max_depth:
            raise ValueError('A depth larger than the trees depth was given')

        quandrants = []

        # search recursively down the quadtree
        self.recursive_search(self.root, depth, quandrants.append)

        return quandrants

    def recursive_search(self, quadrant, max_depth, append_leaf):
        # append if quadrant is a leaf
        if quadrant.leaf is True or quadrant.depth == max_depth:
            append_leaf(quadrant)

        # otherwise keep recursing
        elif quadrant.children is not None:
            for child in quadrant.children:
                self.recursive_search(child, max_depth, append_leaf)

    def create_gif(self, file_name, duration=500, loop=0, show_lines=False):
        gif = []
        end_product_image = self.create_image(self.max_depth, show_lines=show_lines)

        for i in range(self.max_depth):
            proc_image = self.create_image(i, show_lines=show_lines)
            gif.append(proc_image)

        # add extra images at end
        for _ in range(4):
            gif.append(end_product_image)
        gif.reverse()
        gif[0].save(
            file_name,
            save_all=True,
            append_images=gif[1:],
            duration=duration, loop=loop)


if __name__ == '__main__':
    # image_path = "./images/eye.jpg"
    start_time = time.perf_counter()  # Начало отсчета времени
    image_path = "img.png"

    # load image
    image_1 = Image.open(image_path)
    image_1 = image_1.resize((image_1.size[0] * SIZE_MULT, image_1.size[1] * SIZE_MULT))

    # create quadtree
    quadtree = QuadTree(image_1)

    # create image with custom depth
    depth = 7
    image_to_save = quadtree.create_image(depth, show_lines=True)
    quadtree.create_gif("mountain_quadtree.gif", show_lines=True)

    # show image
    # image.show()
    end_time = time.perf_counter()  # Конец отсчета времени
    elapsed_time = end_time - start_time
    image_to_save.save("mountain_quadtree.jpg")
    print(f"Function executed in {elapsed_time:.4f} seconds")