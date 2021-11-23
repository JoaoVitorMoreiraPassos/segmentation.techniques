import numpy as np
import statistics
import tifffile

from sklearn.cluster import KMeans
from glob import glob
from matplotlib import pyplot as plt
from skimage import (filters,
                     measure,
                     exposure,
                     morphology)

from scipy import ndimage as ndi
from skimage.color import rgb2gray
from mpl_toolkits.axes_grid1 import ImageGrid


def apply_kmeans(img, k_clusters=3):
        
    img = np.array(img, dtype=np.float64) / 255

    img_reshaped = img.reshape((-1, 1))
    
    kmeans = KMeans(random_state=0, n_clusters=k_clusters).fit(img_reshaped)
    
    return kmeans.labels_.reshape(img.shape)

def binarize_image(arr):
    return arr > filters.threshold_triangle(arr)


def find_bighest_cluster_area(clusters):
    regions = measure.regionprops(clusters)

    def area(item): return item.area

    return max(map(area, regions))


def find_best_larger_cluster(image_mask):

    clusters = image_mask.copy()

    if statistics.mode(clusters.flatten()):
        clusters = np.invert(clusters)

    clusters = measure.label(clusters, background=0)

    cluster_size = find_bighest_cluster_area(clusters)

    return morphology.remove_small_objects(
        clusters.astype(dtype=bool),
        min_size=(cluster_size-1),
        connectivity=8
    )


def plot(arr_images=[], grid=(2, 2)):

    fig = plt.figure(figsize=(20, 10))

    grid = ImageGrid(fig, 111,
                     nrows_ncols=grid,
                     axes_pad=0.1)

    for ax, img in zip(grid, arr_images):
        ax.imshow(img, cmap='inferno')
        ax.axis('off')

    plt.show()


def read_image_frames_seq(path):
    def parser_image_name(image_name):

        *_, name = image_name.split("/")
        name, *_ = name.split(".")

        return int(name)

    arr = []

    for index in glob(path + "/*"):
        try:
            image_name = parser_image_name(index)

            arr.append(image_name)

        except Exception:
            continue

    image_path_sorted = sorted(arr)

    def image_unique_name(x): return f"{path}/{x}.tif"

    return list(map(image_unique_name, image_path_sorted))


def build_volume_from_directory(arr_paths, is_gray=False):
    arr_images = []

    if is_gray:
        for img_path in arr_paths:

            frame = rgb2gray(tifffile.imread(img_path))

            if not statistics.mode(binarize_image(frame).flatten()):
                continue

            arr_images.append(frame)
    else:
        for img_path in arr_paths:
            
            frame = tifffile.imread(img_path)
            
            if not statistics.mode(binarize_image(frame).flatten()):
                continue
                
            arr_images.append(frame)

    return np.asarray(arr_images)


def check_colision_border(mask):

    x, *_ = mask.shape

    left = mask[:1, ].flatten()
    right = mask[x - 1: x, ].flatten()
    top = mask[:, : 1].flatten()
    bottom = mask[:, x - 1: x].flatten()

    borders_flatten = [left, right, top, bottom]

    if np.concatenate(borders_flatten).sum():
        return True

    return False


def find_broiler_roi(frame, background):
    def second_tecnique(frame):

        binary_frame = binarize_image(exposure.equalize_hist(frame))

        best_cluster = find_best_larger_cluster(binary_frame)

        merged = binarize_image(
            binary_frame.astype("uint8") + best_cluster.astype("uint8")
        )

        best_cluster = find_best_larger_cluster(merged)

        return binarize_image(best_cluster)

    def first_tecnique(frame, background):

        mask_bin = binarize_image(
            np.subtract(
                exposure.equalize_hist(background),
                exposure.equalize_hist(frame),
            )
        )

        best_cluster = find_best_larger_cluster(mask_bin)

        return binarize_image(best_cluster)

    mask_1 = second_tecnique(frame)
    mask_2 = first_tecnique(frame, background)
    mask = (mask_1 + mask_2).astype(dtype=bool)

    if statistics.mode(mask.flatten()):
        mask = np.invert(mask)

    # Arremata

    mask = find_best_larger_cluster(mask)
    mask = morphology.closing(mask, morphology.disk(5))
    mask = ndi.binary_fill_holes(mask)
    mask = filters.gaussian(mask, sigma=1.5)
    mask = binarize_image(mask)

    return mask.astype(dtype=bool)


def rescale_arr(arr, scale=255):
    return (arr * scale).astype('uint8')


def crop_image_box(image=None, shape=(100, 100), margin_pixel=30):

    x, y = shape

    return image[x - margin_pixel:
                 x + margin_pixel,
                 y - margin_pixel:
                 y + margin_pixel]


def find_center_mask(image_bin):

    props, *_ = measure.regionprops(
        measure.label(image_bin)
    )

    x, y = props.centroid

    return int(x), int(y)


def rule_of_three_percent_pixels(arr):

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
