import os
import cv2
import shutil
import numpy as np


def filter_pads(
    folder,
    trash_folder="trash",
    margin=10,
    mask_prefix="mask",
    object_prefix="object",
):
    os.makedirs(os.path.join(folder, trash_folder), exist_ok=True)

    files = os.listdir(folder)

    mask_files = sorted([
        f for f in files
        if f.startswith(mask_prefix)
    ])

    for mask_name in mask_files:
        suffix = mask_name[len(mask_prefix):]
        object_name = object_prefix + suffix

        mask_path = os.path.join(folder, mask_name)
        object_path = os.path.join(folder, object_name)

        # если парного object нет — пропускаем
        if not os.path.exists(object_path):
            continue

        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            continue

        h, w = mask.shape
        ys, xs = np.where(mask > 0)

        # 1. пустая маска
        if len(xs) == 0:
            invalid = True
        else:
            xmin, xmax = xs.min(), xs.max()
            ymin, ymax = ys.min(), ys.max()

            # 2. касание границы
            touches_border = (
                xmin == 0 or ymin == 0 or
                xmax == w - 1 or ymax == h - 1
            )

            # 3. попадание в margin-зону
            touches_margin = (
                xmin < margin or ymin < margin or
                xmax > w - margin - 1 or
                ymax > h - margin - 1
            )

            invalid = touches_border or touches_margin

        if invalid:
            print(f"[REMOVE] {mask_name}")

            shutil.move(
                mask_path,
                os.path.join(folder, trash_folder, mask_name)
            )
            shutil.move(
                object_path,
                os.path.join(folder, trash_folder, object_name)
            )


def split_trace_instances(
    folder,
    trash_folder="trash",
    mask_prefix="mask",
    object_prefix="object",
    min_area=20,
):
    os.makedirs(os.path.join(folder, trash_folder), exist_ok=True)

    files = os.listdir(folder)
    mask_files = sorted([
        f for f in files
        if f.startswith(mask_prefix)
    ])

    for mask_name in mask_files:
        suffix = mask_name[len(mask_prefix):]
        object_name = object_prefix + suffix

        mask_path = os.path.join(folder, mask_name)
        object_path = os.path.join(folder, object_name)

        if not os.path.exists(object_path):
            continue

        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        obj = cv2.imread(object_path, cv2.IMREAD_COLOR)

        if mask is None or obj is None:
            continue

        # бинаризация на всякий случай
        bin_mask = (mask > 0).astype(np.uint8)

        num_labels, labels = cv2.connectedComponents(
            bin_mask,
            connectivity=8
        )

        # фон = label 0
        if num_labels <= 2:
            comp_mask = (labels == 1).astype(np.uint8) * 255
            area = comp_mask.sum() // 255

            if area < min_area:
                shutil.move(mask_path, os.path.join(folder, trash_folder, mask_name))
                shutil.move(object_path, os.path.join(folder, trash_folder, object_name))
            continue

        base_name, ext = os.path.splitext(mask_name)

        comp_idx = 0
        for label in range(1, num_labels):
            comp_mask = (labels == label).astype(np.uint8) * 255
            area = comp_mask.sum() // 255

            if area < min_area:
                continue

            # bounding box (чтобы не сохранять пустоты)
            ys, xs = np.where(comp_mask > 0)
            ymin, ymax = ys.min(), ys.max()
            xmin, xmax = xs.min(), xs.max()

            comp_mask_cropped = comp_mask[ymin:ymax+1, xmin:xmax+1]
            comp_obj_cropped = obj[ymin:ymax+1, xmin:xmax+1]

            # применяем маску
            comp_obj_cropped = cv2.bitwise_and(
                comp_obj_cropped,
                comp_obj_cropped,
                mask=comp_mask_cropped
            )

            new_suffix = suffix.replace(
                ext, f"_{comp_idx}{ext}"
            )

            cv2.imwrite(
                os.path.join(folder, f"{mask_prefix}{new_suffix}"),
                comp_mask_cropped
            )
            cv2.imwrite(
                os.path.join(folder, f"{object_prefix}{new_suffix}"),
                comp_obj_cropped
            )

            comp_idx += 1

        # переносим исходные файлы в trash
        shutil.move(mask_path, os.path.join(folder, trash_folder, mask_name))
        shutil.move(object_path, os.path.join(folder, trash_folder, object_name))
