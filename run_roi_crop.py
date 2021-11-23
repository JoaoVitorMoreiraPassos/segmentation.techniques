import os
import numpy as np
import statistics
from tqdm import tqdm
from glob import glob
from skimage.io import imsave
from skimage import img_as_ubyte

from commons import (build_volume_from_directory,
                     check_colision_border,
                     crop_image_box,
                     find_broiler_roi,
                     find_center_mask,
                     read_image_frames_seq,
                     rescale_arr,
                     rule_of_three_percent_pixels)


if __name__ == "__main__":
    pixel_size = 100

    for path in glob('base/images/*'):

        image_name = path.split('/')[2]

        print(f'[{image_name}] Imagem.')

        try:
            os.makedirs(f'tests2/{image_name}')
        except Exception:
            pass

        seq_frames = read_image_frames_seq(path)

        frames = build_volume_from_directory(seq_frames)

        background = rescale_arr(np.median(frames, axis=0))

        for index, frame in enumerate(tqdm(frames)):

            try:

                mask = find_broiler_roi(frame, background)

                if statistics.mode(mask.flatten()):
                    mask = np.invert(mask)

                if check_colision_border(mask):
                    continue

                center_roi = find_center_mask(mask)

                crop_mask = crop_image_box(image=mask,
                                           shape=center_roi,
                                           margin_pixel=100)

                crop_frame = crop_image_box(image=frame,
                                            shape=center_roi,
                                            margin_pixel=100)

                if not crop_frame.size:
                    continue
                    
                if crop_frame.shape != (pixel_size * 2, pixel_size * 2):
                    continue

                pixel_percent = rule_of_three_percent_pixels(crop_frame)

                if pixel_percent['true_pixels'] >= 98:
                    continue

                imsave(f'tests2/{image_name}/{index}_mask.tif',
                       img_as_ubyte(crop_mask))
                imsave(f'tests2/{image_name}/{index}.tif',
                       img_as_ubyte(crop_frame))

            except Exception:
                continue
