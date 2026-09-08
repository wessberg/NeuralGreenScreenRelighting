"""Conservative source-annotated material scoring, independent of predictions."""
import cv2
import numpy as np
from temporal_iris import footprint_valid


def region_masks(label, shape, boundary=1.5):
    masks = {}
    for name,polygons in label['regions'].items():
        raw = np.zeros(shape,np.uint8)
        for polygon in polygons:
            cv2.fillPoly(raw,[np.asarray(polygon,np.int32)],1)
        masks[name] = cv2.distanceTransform(raw,cv2.DIST_L2,cv2.DIST_MASK_PRECISE)>boundary
    if masks and (np.stack(list(masks.values())).sum(0)>1).any():
        raise ValueError('Overlapping material interiors')
    return masks


def classify_footprints(masks, xy):
    result = np.full(len(xy),'unknown',dtype='<U24')
    for name,mask in masks.items():
        result[footprint_valid(mask,xy)] = name
    return result
