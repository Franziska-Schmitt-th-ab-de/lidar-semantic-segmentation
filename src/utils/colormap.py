import numpy as np

class_idx_to_color = {
    0: [255, 255, 255],  # Ignorierte Labels (schwarz), unlabeled + outlier
    1: [100, 150, 245],  # car
    2: [80, 30, 180],  # truck
    3: [102, 178, 255],  # forklift
    4: [255, 30, 30],  # person
    5: [255, 153, 255],  # bicyclist
    6: [255, 178, 100],  # object
    7: [153, 76, 0],  # pallet
    8: [102, 102, 0],  # terrain
    9: [0, 255, 255],  # driveable ground
    10: [175, 0, 75],  # other ground
    11: [245, 255, 0],  # lane marking
    12: [0, 175, 0],  # vegetation
    13: [102, 51, 0],  # trunk
    14: [255, 200, 0],  # building
    15: [252, 102, 38],  # static object
    16: [255, 0, 0],  # fence
    17: [255, 153, 153],  # rack
}


def create_custom_class_colormap():
    custom_colormap = np.zeros((256, 1, 3), dtype=np.uint8)
    for i in range(256):
        if i in class_idx_to_color:
            custom_colormap[i, 0, :] = class_idx_to_color[i]
        else:
            custom_colormap[i, 0, :] = [0, 0, 0]
    custom_colormap = custom_colormap[..., ::-1]  # OpenCV erwartet BGR-Format
    return custom_colormap


# Color Map für Labels definieren
# color_map = {
#     0: [0, 0, 0],  # 0 unlabeled: schwarz
#     1: [102, 0, 204],  # 1 outlier: dunkellila
#     2: [100, 150, 245],  # 2 car: mittelblau
#     3: [80, 30, 180],  # 3 truck: dunkelblau
#     4: [102, 178, 255],  # 4 forklift: hellblau
#     5: [255, 30, 30],  # 5 person: pink (rot hä)
#     6: [255, 153, 255],  # 6 bicyclist: schweinchenrosa
#     7: [255, 178, 100],  # 7 object: hellorange
#     8: [153, 76, 0],  # 8 pallet: mittelbraun
#     9: [102, 102, 0],  # 9 terrain: khaki
#     10: [0, 255, 255],  # 10 driveable ground: cyan
#     11: [175, 0, 75],  # 11 other ground: dunkelrot
#     12: [245, 255, 0],  # 12 lane marking: neongelb
#     13: [0, 175, 0],  # 13 vegetation: grün
#     14: [102, 51, 0],  # 14 trunk: dunkelbraun
#     15: [255, 200, 0],  # 15 building: senfgelb
#     16: [252, 102, 38],  # 16 static object: dunkelorange
#     17: [255, 0, 0],  # 17 fence: rot
#     18: [255, 153, 153],  # 18 rack: hellrot (lachs)
# }
