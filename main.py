import numpy as np
import os
import json
import cv2
from tqdm import tqdm


def ccw(a, b, c):
    """
    Compute direction from A to B to C.
    :param a: 2-long array describing point A
    :param b: 2-long array describing point B
    :param c: 2-long array describing point C
    :return: 1 if it's a right-turn, -1 if it's a left-turn, and 0 if it's a straight line
    """
    s = (a[0] - b[0]) * (c[1] - b[1]) - (a[1] - b[1]) * (c[0] - b[0])
    if s > 0:
        return 1
    elif s < 0:
        return -1
    else:
        return 0


def intersect(p1, p2, q1, q2):
    """
    Check if segment (p1, p2) intersects with (q1, q2)
    :return: True/False
    """
    if ccw(p1, p2, q1) * ccw(p1, p2, q2) == 1:
        return False
    if ccw(q1, q2, p1) * ccw(q1, q2, p2) == 1:
        return False
    return True


def intersect_with_polygon(polygon, ps, pt):
    for i in range(len(polygon)):
        qs = polygon[i]
        qt = polygon[(i + 1) % len(polygon)]
        if intersect(ps, pt, qs, qt):
            return True
    return False


def intersect_with_roi(roi, x1, y1, x2, y2):
    if intersect_with_polygon(roi, [x1, y1], [x2, y1]):
        return True
    if intersect_with_polygon(roi, [x1, y1], [x1, y2]):
        return True
    if intersect_with_polygon(roi, [x2, y1], [x2, y2]):
        return True
    if intersect_with_polygon(roi, [x1, y2], [x2, y2]):
        return True
    return False


def dist(x1, y1, x2, y2, x3, y3):
    px = x2-x1
    py = y2-y1

    norm = px*px + py*py

    u = ((x3 - x1) * px + (y3 - y1) * py) / float(norm)

    if u > 1:
        u = 1
    elif u < 0:
        u = 0

    x = x1 + u * px
    y = y1 + u * py

    dx = x - x3
    dy = y - y3

    return (dx*dx + dy*dy)**.5


def min_distance_to_polygon(polygon, point):
    ans = 1e9
    for i in range(len(polygon)):
        ps = polygon[i]
        pt = polygon[(i + 1) % len(polygon)]
        ans = min(ans, dist(*ps, *pt, *point))
    return ans


def is_outside_of_image(x, y, width, height):
    return x < 0 or x >= width or y < 0 or y >= height


def solve(occurrence_list, moi_list, roi, height, width, moi_weight):
    ans = np.zeros((height, width))
    path = []
    mark = []
    for fid, x1, y1, x2, y2 in occurrence_list:
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        path.append((int(cx), int(cy)))
        mark.append(intersect_with_roi(roi, x1, y1, x2, y2))

    for i in range(len(path) - 1):
        ps = path[i]
        pt = path[i + 1]
        ans = cv2.line(ans, ps, pt, (255, 255, 255), 20)

    best_match = -1e9
    best_moi = 0
    for idx, moi in enumerate(moi_list):
        overlapped = np.sum(np.minimum(moi, ans) * moi_weight) / 255
        left = np.sum(np.minimum(255 - moi, ans) * moi_weight) / 255

        weight = overlapped - left

        if best_match < weight:
            best_match = weight
            best_moi = idx

    for i in range(len(path) - 1, -1, -1):
        x1, y1, x2, y2 = occurrence_list[i][1:]
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        if min_distance_to_polygon(roi, (cx, cy)) > 50:
            continue
        if not mark[i]:
            return best_moi + 1, occurrence_list[i][0], occurrence_list[i][1:]

    return -1, -1, [0, 0, 0, 0]


def load_moi_list(cam_id):
    ans = []
    movement_id = 0
    while True:
        movement_id = movement_id + 1
        npy_path = os.path.join("moi", cam_id, "movement_%d.npy" % movement_id)
        if os.path.exists(npy_path):
            moi = np.load(npy_path).astype(np.float) * 255
            ans.append(moi)
        else:
            break
    return ans


def process(tracking_path, cam_id, video_id, video_path):
    print("Processing %s" % video_path)
    video = cv2.VideoCapture(video_path)

    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cv2.CAP_PROP_FRAME_COUNT)

    moi_list = load_moi_list(cam_id)
    with open(os.path.join("roi", "%s_new.json" % cam_id), "r") as f:
        roi = json.load(f)
    moi_weight = 255 / (255 + sum(moi_list))

    x = np.load(tracking_path, allow_pickle=True)
    x = x[0:300]

    num_vehicle = 0
    class_ids = {}
    for frame_id in range(x.shape[0]):
        for track_id in range(x[frame_id].shape[0]):
            class_id = int(x[frame_id][track_id][0])
            object_id = int(x[frame_id][track_id][3])
            num_vehicle = max(num_vehicle, object_id)
            class_ids[object_id] = class_id
    num_vehicle = num_vehicle + 1

    ans = [[] for i in range(x.shape[0])]
    for vehicle_id in tqdm(range(num_vehicle)):
        occurrence_list = []
        for frame_id in range(x.shape[0]):
            for track_id in range(x[frame_id].shape[0]):
                _, _, _, object_id, x_min, y_min, x_max, y_max = x[frame_id][track_id]
                if object_id == vehicle_id:
                    # print(class_id, confident_score, x_min, y_min, x_max, y_max)
                    x_min = max(1, x_min)
                    y_min = max(1, y_min)
                    x_max = min(width, x_max)
                    y_max = min(height, y_max)
                    occurrence_list.append([frame_id, x_min, y_min, x_max, y_max])
        if len(occurrence_list) <= 1:
            continue
        moi_id, frame_id, bbox = solve(occurrence_list, moi_list, roi, height, width, moi_weight)
        if moi_id != -1:
            ans[frame_id].append([moi_id, vehicle_id, class_ids[vehicle_id], bbox])

    print("Creating video")
    output = cv2.VideoWriter("output_%s.mp4" % video_id, cv2.VideoWriter_fourcc("m", "p", "4", "v"), fps, (width, height))
    for frame_id in tqdm(range(x.shape[0])):
        _, frame = video.read()
        for moi_id, v_id, v_type, [x1, y1, x2, y2] in ans[frame_id]:
            frame = cv2.putText(
                frame,
                ("C" if v_type == 1 else "T") + " " + str(v_id) + " " + str(moi_id),
                (int(min(width - 300, x1)), int(min(height - 300, y1))),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.5,
                (0, 0, 255),
                5
            )
        output.write(frame)
    video.release()
    output.release()
    print("")


if __name__ == "__main__":
    # for filename in os.listdir("videos"):
    #     video_id, ext = os.path.splitext(filename)
    #     if ext != ".mp4":
    #         continue
    #
    #     cam_id = "_".join(video_id.split("_")[0:2])
    #     process(
    #         os.path.join("tracking-results", "info_%s.mp4.npy" % video_id),
    #         cam_id,
    #         video_id,
    #         os.path.join("videos", "%s.mp4" % video_id)
    #     )
    process(
        os.path.join("tracking-results", "info_%s.mp4.npy" % "cam_7"),
        "cam_7",
        "cam_7",
        os.path.join("videos", "cam_7.mp4")
    )
