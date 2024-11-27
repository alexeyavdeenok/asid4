import numpy as np
from PIL import Image, ImageDraw
import time
import math


DETAIL_THRESHOLD = 13


def compute_integral_images(image):
    """Вычисляет интегральные изображения для каждого канала (R, G, B) и их квадратов."""
    image_arr = np.asarray(image).astype(np.float32)

    integral = {}
    for channel, name in zip(range(3), ["R", "G", "B"]):
        integral[name] = np.cumsum(np.cumsum(image_arr[:, :, channel], axis=0), axis=1)
        integral[f"{name}_squared"] = np.cumsum(np.cumsum(image_arr[:, :, channel] ** 2, axis=0), axis=1)

    return integral


def region_sum(data, bbox):
    """Вычисляет сумму интенсивностей для региона."""
    left, top, right, bottom = bbox

    total = data[bottom - 1, right - 1]
    if left > 0:
        total -= data[bottom - 1, left - 1]
    if top > 0:
        total -= data[top - 1, right - 1]
    if left > 0 and top > 0:
        total += data[top - 1, left - 1]

    return total


def region_stats(integral, bbox):
    """Вычисляет сумму, среднее и стандартное отклонение для региона."""
    stats = {}
    for name in ["R", "G", "B"]:
        sum_pixels = region_sum(integral[name], bbox)
        sum_squares = region_sum(integral[f"{name}_squared"], bbox)
        pixel_count = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])

        mean = sum_pixels / pixel_count
        variance = sum_squares / pixel_count - mean ** 2
        stats[name] = {
            "mean": mean,
            "std_dev": max(0, variance) ** 0.5  # Стандартное отклонение
        }

    return stats


def get_detail(stats):
    """Вычисляет интегрированное значение детализации для региона на основе отклонений."""
    red_detail = stats["R"]["std_dev"]
    green_detail = stats["G"]["std_dev"]
    blue_detail = stats["B"]["std_dev"]

    # Взвешенная сумма отклонений
    detail_intensity = red_detail * 0.2989 + green_detail * 0.5870 + blue_detail * 0.1140
    return detail_intensity


class Quadrant:
    def __init__(self, image_integrals, bbox, depth):
        self.bbox = bbox
        self.depth = depth
        self.children = None
        self.leaf = False

        # Получаем статистики региона
        stats = region_stats(image_integrals, bbox)
        self.image_integrals = image_integrals

        # Вычисляем уровень детализации и средний цвет
        self.detail = get_detail(stats)
        self.colour = tuple(int(stats[ch]["mean"]) for ch in ["R", "G", "B"])

    def split_quadrant(self):
        left, top, right, bottom = self.bbox

        # Вычисляем середины текущего региона
        middle_x = left + (right - left) // 2
        middle_y = top + (bottom - top) // 2

        # Создаём 4 новых квадранта
        upper_left = Quadrant(self.image_integrals, (left, top, middle_x, middle_y), self.depth + 1)
        upper_right = Quadrant(self.image_integrals, (middle_x, top, right, middle_y), self.depth + 1)
        bottom_left = Quadrant(self.image_integrals, (left, middle_y, middle_x, bottom), self.depth + 1)
        bottom_right = Quadrant(self.image_integrals, (middle_x, middle_y, right, bottom), self.depth + 1)

        self.children = [upper_left, upper_right, bottom_left, bottom_right]


class QuadTree:
    def __init__(self, image):
        self.width, self.height = image.size
        self.image = image
        self.max_depth = 0
        self.max_allowed_depth = self.max_quadtree_depth()

        # Создание интегральных изображений
        image_integrals = compute_integral_images(self.image)

        # Старт построения дерева
        self.root = Quadrant(image_integrals, (0, 0, self.width, self.height), 0)
        self.build(self.root)

    def build(self, root):
        if root.depth >= self.max_allowed_depth or root.detail <= DETAIL_THRESHOLD:
            if root.depth > self.max_depth:
                self.max_depth = root.depth
            root.leaf = True
            return

        root.split_quadrant()
        for child in root.children:
            self.build(child)

    def get_leaf_quadrants(self, depth):
        if depth > self.max_depth:
            raise ValueError("A depth larger than the tree's depth was given")
        quandrants = []
        self.recursive_search(self.root, depth, quandrants.append)
        return quandrants

    def recursive_search(self, quadrant, max_depth, append_leaf):
        if quadrant.leaf is True or quadrant.depth == max_depth:
            append_leaf(quadrant)
        elif quadrant.children is not None:
            for child in quadrant.children:
                self.recursive_search(child, max_depth, append_leaf)

    def max_quadtree_depth(self):
        """Вычисляет максимальную глубину QuadTree для изображения."""
        min_dim = min(self.width, self.height)
        return math.floor(math.log2(min_dim))


def create_image(tree, custom_depth, show_lines=False):
    new_image = Image.new('RGB', (tree.width, tree.height))
    draw = ImageDraw.Draw(new_image)
    draw.rectangle((0, 0, tree.width, tree.height), (0, 0, 0))

    leaf_quadrants = tree.get_leaf_quadrants(custom_depth)
    for quadrant in leaf_quadrants:
        if show_lines:
            draw.rectangle(quadrant.bbox, quadrant.colour, outline=(0, 0, 0))
        else:
            draw.rectangle(quadrant.bbox, quadrant.colour)

    return new_image


def create_gif(tree, custom_depth, file_name, duration=500, loop=0, show_lines=False):
    gif_frames = []
    for depth in range(custom_depth + 1):
        frame = create_image(tree, depth, show_lines=show_lines)
        gif_frames.append(frame)

    # Добавление повторяющихся кадров на последнем уровне
    end_frame = create_image(tree, custom_depth, show_lines=show_lines)
    for _ in range(4):
        gif_frames.append(end_frame)

    # Сохранение GIF
    gif_frames[0].save(
        file_name,
        save_all=True,
        append_images=gif_frames[1:],
        duration=duration,
        loop=loop
    )


if __name__ == '__main__':
    start_time = time.perf_counter()
    image_path = "test.jpg"
    image_1 = Image.open(image_path)
    #image_1 = image_1.resize((image_1.size[0] * SIZE_MULT, image_1.size[1] * SIZE_MULT))
    quadtree = QuadTree(image_1)

    # Создание изображения на конкретной глубине
    depth = 11
    print(quadtree.max_quadtree_depth())
    #image_to_save = create_image(quadtree, depth, show_lines=False)
    #image_to_save.save("mountain_quadtree.jpg")
    # Создание GIF файла
    #create_gif(quadtree, depth, "mountain_quadtree.gif", duration=500, show_lines=True)
    end_time = time.perf_counter()  # Конец отсчета времени
    elapsed_time = end_time - start_time
    print(f"Function executed in {elapsed_time:.4f} seconds")
