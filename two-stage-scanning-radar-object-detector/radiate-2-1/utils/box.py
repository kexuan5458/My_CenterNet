import numpy as np


def get_annotation_with_angle(ann):
    new_ann = None
    # get width and height
    if 'bbox' in ann and len(ann['bbox']) == 5:
        # using rotated bounding box datasets. 5 = [cx,cy,w,h,angle]
        # assert len(ann['rbbox']) == 5, 'Unknown bbox format'  # x,y,w,h,a
        new_ann = np.array(ann['bbox'], dtype=np.float32)

        # sungting: transfrom ang from 0-360 to -90-90
        ang = new_ann[4]
        if (90 <= ang) and (ang < 180): ang -= 180
        elif (180 <= ang) and (ang < 270): ang -= 180
        elif (270 <= ang) and (ang < 360): ang -= 360
        new_ann[4] = -ang

        # sungting: change x1, y1 to cx, cy
        new_ann[0] = new_ann[0] + new_ann[2] / 2
        new_ann[1] = new_ann[1] + new_ann[3] / 2

        if new_ann[2] > new_ann[3]:
            new_ann[2], new_ann[3] = new_ann[3], new_ann[2]
            new_ann[4] -= 90 if new_ann[4] > 0 else -90
    elif not 'rbbox' in ann:
        bbox = np.array(ann['bbox'], dtype=np.float32)
        # convert COCO format: x1,y1,w,h to cx,cy,w,h
        bbox[0] = bbox[0] + bbox[2] / 2
        bbox[1] = bbox[1] + bbox[3] / 2
        bbox.append(0)
        if bbox[2] > bbox[3]:
            bbox[2], ann['bbox'][3] = bbox[3], bbox[2]
            bbox[4] -= 90
        new_ann = bbox
    else:
        # using rotated bounding box datasets. 5 = [cx,cy,w,h,angle]
        assert len(ann['rbbox']) == 5, 'Unknown bbox format'  # x,y,w,h,a
        new_ann = np.array(ann['rbbox'], dtype=np.float32)
        if new_ann[2] > new_ann[3]:
            new_ann[2], new_ann[3] = new_ann[3], new_ann[2]
            new_ann[4] -= 90 if new_ann[4] > 0 else -90

    if new_ann[2] == new_ann[3]:
        new_ann[3] += 1  # force that w < h

    if new_ann[4] == 90:
        new_ann[4] = -90

    new_ann[4] = np.clip(new_ann[4], -90, 90 - 1e-5)
    

    assert new_ann[2] < new_ann[3], "width not smaller than height"
    assert (new_ann[4] >= -90 and
            new_ann[4] < 90), f"{new_ann[4]} not in interval [-90, 90)"

    # override original bounding box with rotated bounding box
    return new_ann


def rotate_bbox(x, y, w, h, angle):
    c, s = np.cos(np.radians(angle)), np.sin(np.radians(angle))
    R = np.asarray([[c, s], [-s, c]])
    pts = np.asarray(
        [[-w / 2, -h / 2],
         [w / 2, -h / 2],
         [w / 2, h / 2],
         [-w / 2, h / 2]])
    rot_pts = []
    for pt in pts:
        rot_pts.append(([x, y] + pt @ R).astype(int))
    return rot_pts
