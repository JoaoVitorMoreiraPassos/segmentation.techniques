import math
import tifffile
import numpy as np
import matplotlib.pyplot as plt

from glob import glob
from mpl_toolkits.axes_grid1 import ImageGrid
from skimage import measure, img_as_ubyte
from skimage import filters


# Recorte de imagens

def crop_image_reduce_errors(image):
    # Números mágicos
    return image[130:380, 50:400]


def crop_image_box(image=None, shape=(100, 100), margin_pixel=30):
    """
        Essa função a partir do centro da imagem tenta fazer o crop da galinha
        Não é perfeita, mas serve :D
    """
    x, y = shape

    return image[x - margin_pixel:
                 x + margin_pixel,
                 y - margin_pixel:
                 y + margin_pixel]


def find_center_mask(image_bin):
    """
        Retona o centro da imagem
    """

    props, *_ = measure.regionprops(
        measure.label(image_bin)
    )

    x, y = props.centroid

    return int(x), int(y)


def build_volume_from_directory(path_folder, with_rgb=False):
    """
        Ler todas as imagens do diretório e cria um bloco de imagens
    """
    if with_rgb:
        return np.asarray([
            crop_image_reduce_errors(tifffile.imread(img))
            for img in glob(path_folder + '/*')
        ])

    return np.asarray([
        crop_image_reduce_errors(tifffile.imread(img)[:, :, 0])
        for img in glob(path_folder + '/*')
    ])

def binarize_image(image_array):
    """
        Binariza imagens utilizando algoritmo OTSU
    """
    return image_array > filters.threshold_otsu(image_array)


def rule_of_three(arr):
    """
        Essa função calcula a porcentagem de pixels Pretos e Brancos
        Essa informação é importante para selecionar imagens válidas para
        processamentos
    """

    def co_occurrence(arr):

        unique, counts = np.unique(arr, return_counts=True)

        return dict(zip(unique, counts))

    def ternary(value):
        if value is None:
            return 0
        return value

    image_bin = binarize_image(arr)
    image_coo = co_occurrence(image_bin)

    true_value = ternary(image_coo.get(True))
    false_value = ternary(image_coo.get(False))

    _100 = false_value + true_value

    false = int((false_value * 100) / _100)

    true = int((true_value * 100) / _100)

    return {
        'true': true,
        'false': false
    }


# Visualização

def plot_box_mask(image_bin):
    """
        Legalizinha até, faz um plot da mascara com uma box ao redor
        e marcando o centro da imagem
    """
    regions = measure.regionprops(
        img_as_ubyte(
            image_bin
        )
    )

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.imshow(image_bin, cmap='inferno')

    for props in regions:

        y0, x0 = props.centroid
        orientation = props.orientation

        x1 = x0 + math.cos(orientation) * 0.5 * props.minor_axis_length
        y1 = y0 - math.sin(orientation) * 0.5 * props.minor_axis_length
        x2 = x0 - math.sin(orientation) * 0.5 * props.major_axis_length
        y2 = y0 - math.cos(orientation) * 0.5 * props.major_axis_length

        ax.plot((x0, x1), (y0, y1), '-r', linewidth=2.5)
        ax.plot((x0, x2), (y0, y2), '-r', linewidth=2.5)
        ax.plot(x0, y0, '.g', markersize=15)

        minr, minc, maxr, maxc = props.bbox
        bx = (minc, maxc, maxc, minc, minc)
        by = (minr, minr, maxr, maxr, minr)
        ax.plot(bx, by, 'lightgreen', linewidth=2.5)

    plt.show()


def plot_images(images, color='gray', names=[]):
    """
        Função para plotar array de imagens, essa função não é perfeita
        mas serve bem...
    """

    if len(names) == 0:
        names = [""] * len(images)

    if len(images) == 1:
        plt.figure(figsize=(10, 8))
        plt.imshow(images[0], color)

        return plt.show()

    fig, ax = plt.subplots(1,
                           len(images),
                           figsize=(15, 20))

    for index, arr in enumerate(images):
        ax[index].imshow(arr, cmap=color)
        ax[index].set_title(names[index])

    plt.show()


def plot_grid_images(arr_images=[], grid=(2, 2)):
    """
        Essa função é massa, renderiza um grid de imagens xD
    """

    fig = plt.figure(figsize=(20, 10))

    grid = ImageGrid(fig, 111,
                     nrows_ncols=grid,
                     axes_pad=0.1)

    for ax, img in zip(grid, arr_images):
        ax.imshow(img, cmap='inferno')
        ax.axis('off')

    plt.show()


def plot_image_with_contour(original_image, mask_image):
    """
        Função massa, retorna a imagem com o contorno da mascara segmentada
    """

    contours = measure.find_contours(mask_image, 0.8)

    _, ax = plt.subplots(figsize=(10, 8))
    ax.imshow(original_image, cmap='inferno')

    for contour in contours:
        ax.plot(contour[:, 1],
                contour[:, 0],
                color='w',
                linewidth=3)

    ax.set_xticks([])
    ax.set_yticks([])
    plt.show()
