import argparse
import os
import time
from PIL import Image, ImageDraw
from tree import QuadTree


def parse_arguments():
    parser = argparse.ArgumentParser(description="QuadTree Image Processing")

    # Аргумент: путь к файлу
    parser.add_argument(
        '-f', '--filename',
        required=True,
        help="Path to the input image file."
    )

    # Аргумент: уровень детализации (1-5)
    parser.add_argument(
        '-d', '--detail',
        type=int,
        choices=range(1, 6),
        default=3,
        help="Detail level for image processing (1-5). Default is 3."
    )

    # Аргумент: пользовательский уровень детализации (целочисленный)
    parser.add_argument(
        '-cd', '--custom_detail',
        type=int,
        default=None,
        help="Custom detail threshold (integer). Overrides standard detail setting."
    )

    # Аргумент: глубина дерева (1-20)
    parser.add_argument(
        '-dep', '--depth',
        type=int,
        choices=range(1, 15),
        default=8,
        help="Maximum depth of the quadtree (1-20). Default is 10."
    )

    # Аргумент: сохранение изображения (bl/nl)
    parser.add_argument(
        '-sp', '--save_picture',
        choices=['bl', 'nl'],
        help="Save the output image: 'bl' (with black lines) or 'nl' (no lines)."
    )

    # Аргумент: сохранение GIF (rev, bl/nl)
    parser.add_argument(
        '-sg', '--save_gif',
        nargs='+',
        help="Save the output GIF. Specify 'rev' for reverse animation and 'bl'/'nl' (black lines/no lines)."
    )

    # Парсим аргументы
    args = parser.parse_args()

    if args.save_picture == 'bl':
        save_picture_value = True
    elif args.save_picture == 'nl':
        save_picture_value = False
    else:
        save_picture_value = None

    # Обработка аргумента -sg
    if args.save_gif:
        if 'rev' in args.save_gif:
            reverse_animation = True
            # Если 'rev' указан, находим 'bl' или 'nl' и присваиваем значение
            line_option = 'bl' in args.save_gif
        else:
            reverse_animation = False
            # Если 'rev' не указан, то находим 'bl' или 'nl' и присваиваем значение
            line_option = 'bl' in args.save_gif
    else:
        reverse_animation = None
        line_option = False  # Если 'save_gif' не указан, устанавливаем False по умолчанию

    detail_dict = {1: 4, 2: 9, 3: 14, 4: 19, 5: 25}
    choice_detail = detail_dict[args.detail]
    # Формируем словарь аргументов
    result = {
        'filename': args.filename,
        'detail': choice_detail,
        'custom_detail': args.custom_detail,
        'depth': args.depth,
        'save_picture': save_picture_value,
        'save_gif': {
            'reverse_animation': reverse_animation,
            'line_option': line_option
        }
    }

    return result


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


def create_gif(tree, custom_depth, file_name, duration=500, loop=0, show_lines=False, revers=False):
    gif_frames = []
    for depth in range(custom_depth + 1):
        frame = create_image(tree, depth, show_lines=show_lines)
        gif_frames.append(frame)

    # Добавление повторяющихся кадров на последнем уровне
    end_frame = create_image(tree, custom_depth, show_lines=show_lines)
    for _ in range(4):
        gif_frames.append(end_frame)
    if not revers:
        gif_frames.reverse()

    # Сохранение GIF
    gif_frames[0].save(
        file_name,
        save_all=True,
        append_images=gif_frames[1:],
        duration=duration,
        loop=loop
    )


def main():
    start_time = time.perf_counter()
    args_dict = parse_arguments()
    file_name = args_dict['filename']
    if args_dict['custom_detail'] is None:
        detail = args_dict['detail']
    else:
        detail = args_dict['custom_detail']
    picture = args_dict['save_picture']
    depth = args_dict['depth']
    gif_create = args_dict['save_gif']
    try:
        # Проверяем, существует ли файл перед открытием
        if not os.path.exists(file_name):
            raise FileNotFoundError(f"File not found: {file_name}")

        # Открываем изображение
        image = Image.open(file_name)

    except FileNotFoundError as e:
        print(f"Error: {e}")
        return
    except IOError as e:
        print(f"Error: Unable to open the image file. {e}")
        return
    else:
        print("Image successfully loaded.")

    quadtree = QuadTree(image, detail)
    if depth > quadtree.max_allowed_depth:
        depth = quadtree.max_allowed_depth

    if picture is not None:
        image_to_save = create_image(quadtree, depth, show_lines=picture)
        image_to_save.save('compressed_image.jpg')
    if gif_create['reverse_animation'] is not None:
        create_gif(quadtree, depth, 'compress_image.gif', show_lines=gif_create['line_option'],
                   revers=gif_create['reverse_animation'])

    end_time = time.perf_counter()  # Конец отсчета времени
    elapsed_time = end_time - start_time
    print(f"Function executed in {elapsed_time:.4f} seconds")


if __name__ == "__main__":
    main()
