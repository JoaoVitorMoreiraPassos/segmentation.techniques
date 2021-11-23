import statistics
import numpy as np

from scipy import ndimage as ndi
from skimage import (measure,
                     morphology,
                     exposure,
                     filters,
                     feature)


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

    """
        Verificação para saber se a mascara está correta c:
    """

    if statistics.mode(clusters.flatten()):
        clusters = np.invert(clusters)

    clusters = measure.label(clusters, background=0)

    cluster_size = find_bighest_cluster_area(clusters)

    return morphology.remove_small_objects(
        clusters,
        min_size=(cluster_size-1),
        connectivity=8
    )


def segmentation_mask(image, background):

    frame = np.subtract(exposure.equalize_hist(image),
                        exposure.equalize_hist(background))

    sobel_edges = filters.sobel(frame)

    edges = feature.canny(sobel_edges)

    fill = ndi.binary_fill_holes(edges)

    clusters = find_best_larger_cluster(fill)

    closed_frame = morphology.closing(clusters, morphology.disk(15))

    return ndi.binary_fill_holes(closed_frame)
