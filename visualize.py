"""
Source: https://github.com/hendrycks/anomaly-seg/issues/15#issuecomment-890300278
"""
import numpy as np
from PIL import Image


COLORS = np.array([
    [  0,   0,   0],  # unlabeled    =   0,
    [ 70,  70,  70],  # building     =   1,
    [190, 153, 153],  # fence        =   2, 
    [250, 170, 160],  # other        =   3,
    [220,  20,  60],  # pedestrian   =   4, 
    [153, 153, 153],  # pole         =   5,
    [157, 234,  50],  # road line    =   6, 
    [128,  64, 128],  # road         =   7,
    [244,  35, 232],  # sidewalk     =   8,
    [107, 142,  35],  # vegetation   =   9, 
    [  0,   0, 142],  # car          =  10,
    [102, 102, 156],  # wall         =  11, 
    [220, 220,   0],  # traffic sign =  12,
    [ 60, 250, 240],  # anomaly      =  13,
]) 


def color(annot_path: str, colors: np.ndarray) -> Image.Image:
    img_pil = Image.open(annot_path)
    img_np = np.array(img_pil)
    img_new = np.zeros((720, 1280, 3))

    for index, color in enumerate(colors):
        img_new[img_np == index + 1] = color
    
    return Image.fromarray(img_new.astype("uint8"), "RGB")


if __name__ == "__main__":
    annot_path = "StreetHazards/train/annotations/t1-3/1.png"
    segm_map = color(annot_path, COLORS)
    segm_map.save("1.png")
