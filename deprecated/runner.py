import os
import statistics
import tifffile
import numpy as np
from skimage import filters
from glob import glob
from skimage.io import imsave, imread
from skimage import img_as_ubyte
from tqdm import tqdm
from commons import (crop_image_box,
                     crop_image_reduce_errors,
                     find_center_mask)


def runner(arr_images,
           bg_image,
           image_name,
           path_out_images,
           fn_segmentation,
           crop_margin_pixel=80):

    path_images = f'{path_out_images}/{image_name}/images'
    path_masks = f'{path_out_images}/{image_name}/masks'

    try:
        os.makedirs(path_images)
        os.makedirs(path_masks)
    except Exception:
        print('Pastas images e masks já existem!')

    for index, frame in enumerate(tqdm(arr_images)):
        image = crop_image_reduce_errors(frame)

        try:
            mask = fn_segmentation(image, bg_image)
        except Exception:
            continue

        mask_center = find_center_mask(mask)

        crop_mask = crop_image_box(image=mask,
                                   shape=mask_center,
                                   margin_pixel=crop_margin_pixel)

        crop_image = crop_image_box(image=image,
                                    shape=mask_center,
                                    margin_pixel=crop_margin_pixel)

        if crop_mask.size == 0:
            continue

        if not statistics.mode(crop_mask.flatten()):

            imsave(f'{path_images}/{str(index)}.tif', crop_image)
            imsave(f'{path_masks}/{str(index)}.tif', img_as_ubyte(crop_mask))


def rule_of_three_percent_pixels(arr):
    """
        Essa função calcula a porcentagem de pixels Pretos e Brancos
        Essa informação é importante para selecionar imagens válidas para
         processamentos
    """

    def co_occurrence(arr):
        unique, counts = np.unique(arr, return_counts=True)

        return dict(zip(unique, counts))

    def ternary(value):
        return 0 if value is None else value

    def binarize_image(arr):
        return arr > filters.threshold_minimum(arr)

    image_bin = binarize_image(arr)
    image_coo = co_occurrence(image_bin)

    true_value = ternary(image_coo.get(True))
    false_value = ternary(image_coo.get(False))

    _100 = false_value + true_value

    return dict({
        'true_pixels': int((true_value * 100) / _100),
        'false_pixels': int((false_value * 100) / _100)
    })


def check_colision_border(mask):
    """
        Pós processamento
    """
    x, *_ = mask.shape

    left = mask[:1, ].flatten()
    right = mask[x - 1: x, ].flatten()
    top = mask[:, : 1].flatten()
    bottom = mask[:, x - 1: x].flatten()

    borders_flatten = [left, right, top, bottom]
    if np.concatenate(borders_flatten).sum():
        return True

    return False


def runner_v2(bg_image_path,
              img_name,
              path_out_folder,
              path_images_in_folder,
              fn_segmentation):

    path_images = f'{path_out_folder}/{img_name}/images'
    path_masks = f'{path_out_folder}/{img_name}/masks'

    try:
        os.makedirs(path_out_folder)
    except Exception:
        pass

    try:
        os.makedirs(path_images)
        os.makedirs(path_masks)
    except Exception:
        pass

    arr_images = glob(path_images_in_folder + '/*')
    background = crop_image_reduce_errors(imread(bg_image_path)[:, :, 0])

    for index, frame_path in enumerate(tqdm(arr_images)):

        image_loaded = tifffile.imread(frame_path)[:, :, 0]
        image = crop_image_reduce_errors(image_loaded)

        try:
            percents = rule_of_three_percent_pixels(image)
        except Exception:
            continue

        is_image_valid = percents['true_pixels'] > percents['false_pixels']

        # Ignora se a imagem for predominantemente preta
        if not is_image_valid:
            continue

        # Ignora qualquer error de segmentação e parte para o proximo frame
        try:
            image_cropped, mask = fn_segmentation(image, background)
        except Exception:
            continue

        # Ignora se a mascara colidir com as bordas
        if check_colision_border(mask):
            continue

        imsave(f'{path_images}/{str(index)}.tif', image_cropped)
        imsave(f'{path_masks}/{str(index)}.tif', img_as_ubyte(mask))
