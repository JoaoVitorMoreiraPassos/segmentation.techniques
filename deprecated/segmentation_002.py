import numpy as np
import statistics

from skimage import feature
from scipy import ndimage as ndi
from sklearn.cluster import KMeans
from skimage import (filters,
                     morphology,
                     exposure,
                     measure)

from commons import crop_image_box, binarize_image


def apply_kmeans(img, k_clusters=3):
    return KMeans(random_state=1,
                  n_clusters=k_clusters,
                  init='k-means++'
                  ).fit(img.reshape((-1, 1))).labels_.reshape(img.shape)


def find_center_mask(image_bin):
    """
        Retona o centro da imagem
    """

    props, *_ = measure.regionprops(
        measure.label(image_bin)
    )

    x, y = props.centroid

    return int(x), int(y)


def find_perfect_cluster(mask):
    """
        Função especial para o kmeans
    """
    max_area = 0
    max_area_index = 0

    for index, props in enumerate(measure.regionprops(mask)):
        if props.area >= max_area:
            max_area = props.area
            max_area_index = index + 1
    return max_area_index


def find_bighest_cluster_area(clusters):
    """
        Essa função deve receber uma imagem segmentada (Clusters)
        Retorna a área do maior cluster
    """
    regions = measure.regionprops(clusters)

    def area(item): return item.area

    return max(map(area, regions))


def find_best_larger_cluster(image_mask):

    clusters = image_mask.copy()

    if statistics.mode(clusters.flatten()):
        clusters = np.invert(clusters)

    clusters_labels = measure.label(clusters, background=0)

    cluster_size = find_bighest_cluster_area(clusters_labels)

    return morphology.remove_small_objects(
        clusters,
        min_size=(cluster_size-1),
        connectivity=8
    )


def segmentation_mask(image, background):

    frame_eq = np.subtract(exposure.equalize_hist(image),
                           exposure.equalize_hist(background))

    # Aplica K-means na imagem
    segment_clusters = apply_kmeans(frame_eq, k_clusters=3)

    # Ecolhe melhor da segmentação Kmeans
    best_segment_cluster = find_perfect_cluster(segment_clusters)

    # Encontra maior cluster a mascara e remove as menores
    best_cluster = find_best_larger_cluster(
        (segment_clusters == best_segment_cluster)
    )

    # Detectando bordas
    canny_edges = feature.canny(best_cluster)
    sobel_edges = filters.sobel(canny_edges)

    # Conectando bordase preenchendo clusters
    closed_frame = morphology.closing(sobel_edges,
                                      morphology.disk(15))

    return ndi.binary_fill_holes(closed_frame)


def segmentation_roi(image):
    
    # Aplica K-means na imagem
    clusters = apply_kmeans(image, k_clusters=2)

    # Ecolhe melhor da segmentação Kmeans
    best_cluster = find_perfect_cluster(clusters)

    # Encontra maior cluster a mascara e remove as menores
    return find_best_larger_cluster((clusters == best_cluster))

    
def segmentation_mask_v2(image, background):
    
    image_eq = np.subtract(exposure.equalize_hist(image),
                           exposure.equalize_hist(background))
    
    image_roi = segmentation_roi(image_eq)
    
    crop_image = crop_image_box(image=image,
                                shape=find_center_mask(image_roi),
                                margin_pixel=80)
    # Detectando bordas
    canny_edges = feature.canny(exposure.equalize_hist(crop_image))
    sobel_edges = filters.sobel(canny_edges)

    # Conectando bordase preenchendo clusters
    closed_frame = morphology.closing(sobel_edges, morphology.disk(15))

    closed_frame = binarize_image(closed_frame)

    closed_frame = find_best_larger_cluster(closed_frame)

    return crop_image, ndi.binary_fill_holes(closed_frame)
