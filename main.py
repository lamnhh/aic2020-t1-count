import numpy as np
import math
import os
import json
import cv2
from tqdm import tqdm
import matplotlib.path as mplPath


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


def inside_roi(roi, point):
    pol = mplPath.Path(roi)
    return pol.contains_point(point, radius=1)


# def dist(x1, y1, x2, y2, x3, y3):
#     px = x2-x1
#     py = y2-y1
#
#     norm = px*px + py*py
#
#     u = ((x3 - x1) * px + (y3 - y1) * py) / float(norm)
#
#     if u > 1:
#         u = 1
#     elif u < 0:
#         u = 0
#
#     x = x1 + u * px
#     y = y1 + u * py
#
#     dx = x - x3
#     dy = y - y3
#
#     return (dx*dx + dy*dy)**.5
#
#
# def min_distance_to_polygon(polygon, point):
#     ans = 1e9
#     for i in range(len(polygon)):
#         ps = polygon[i]
#         pt = polygon[(i + 1) % len(polygon)]
#         ans = min(ans, dist(*ps, *pt, *point))
#     return ans


def line_intersection(line1, line2):
    xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
    ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])

    def det(a, b):
        return a[0] * b[1] - a[1] * b[0]

    div = det(xdiff, ydiff)
    if div == 0:
       raise Exception('lines do not intersect')

    d = (det(*line1), det(*line2))
    x = det(d, xdiff) / div
    y = det(d, ydiff) / div
    return x, y


def dist(x1, y1, x2, y2):
    return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


def almost_touch_roi(roi, x1, y1, x2, y2, velocity, moi):
    point = ((x1 + x2) / 2, (y1 + y2) / 2)
    lowest_dist = 1e9
    endpoint = (point[0] + velocity[0] * 10, point[1] + velocity[1] * 10)
    for i in range(len(roi)):
        p = roi[i]
        q = roi[(i + 1) % len(roi)]
        if intersect(point, endpoint, p, q):
            try:
                x, y = line_intersection([point, endpoint], [p, q])
                if moi[int(y)][int(x)] > 0:
                    lowest_dist = min(lowest_dist, dist(point[0], point[1], x, y))
            except:
                pass
    return lowest_dist <= max(max(abs(y2 - y1), abs(x2 - x1)) * 0.8, 100)


def is_outside_of_image(x, y, width, height):
    return x < 0 or x >= width or y < 0 or y >= height


def solve(occurrence_list, moi_list, roi, height, width, moi_weight):
    ans = np.zeros((height, width))
    path = []
    for fid, x1, y1, x2, y2 in occurrence_list:
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        path.append((int(cx), int(cy)))

    if not inside_roi(roi, path[-1]):
        tmp = 0
        while len(path) > 0 and not inside_roi(roi, path[-1]):
            tmp = path[-1]
            path = path[:-1]
        if tmp == 0:
            return -1, -1, [0, 0, 0, 0]
        path.append(tmp)

    if len(path) <= 1:
        return -1, -1, [0, 0, 0, 0]
    occurrence_list = occurrence_list[:len(path)]

    path = [(2 * path[0][0] - path[1][0], 2 * path[0][1] - path[1][1])] + path
    for i in range(len(path) - 1):
        ps = path[i]
        pt = path[i + 1]
        ans = cv2.line(ans, ps, pt, (255, 255, 255), 50)
    path = path[1:]

    # Check if path is inside ANY MOI. If not, return immediately
    found = False
    for moi in moi_list:
        overlapped = np.sum(np.minimum(moi, ans))
        if overlapped > 0:
            found = True
            break
    if not found:
        return -1, -1, [0, 0, 0, 0]

    best_match = -1e12
    best_moi = 0
    for idx, moi in enumerate(moi_list):
        overlapped = np.sum(np.minimum(moi, ans) * moi_weight) / 255
        left = np.sum(np.minimum(255 - moi, ans) * moi_weight) / 255

        weight = overlapped - 10000 * left

        if best_match < weight:
            best_match = weight
            best_moi = idx

    if best_match < -4e8:
        return -1, -1, [0, 0, 0, 0]

    prev = max(0, len(path) - 10)
    vx = (path[-1][0] - path[prev][0]) / (occurrence_list[-1][0] - occurrence_list[prev][0])
    vy = (path[-1][1] - path[prev][1]) / (occurrence_list[-1][0] - occurrence_list[prev][0])
    for i in range(len(path) - 1, -1, -1):
        if inside_roi(roi, path[i]) and almost_touch_roi(roi, *occurrence_list[i][1:], (vx, vy), moi_list[best_moi]):
            return best_moi + 1, occurrence_list[i][0], occurrence_list[i][1:]
        if i > 0 and not inside_roi(roi, path[i]) and inside_roi(roi, path[i - 1]):
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
    moi_weight = math.e ** (255 / (255 + sum(moi_list)))

    x = np.load(tracking_path, allow_pickle=True)

    num_frame = 0
    num_vehicle = 0
    class_ids = {}
    for class_id, frame_id, _, object_id, _, _, _, _ in x:
        num_vehicle = max(num_vehicle, object_id)
        num_frame = max(num_frame, frame_id)
        class_ids[object_id] = class_id
    num_vehicle = int(num_vehicle + 1)
    num_frame = int(num_frame + 1)

    ans = [[] for i in range(num_frame)]
    for vehicle_id in tqdm(range(num_vehicle)):
    # for vehicle_id in range(39, 40):
        occurrence_list = []
        for class_id, frame_id, _, object_id, x_min, y_min, x_max, y_max in x:
            if object_id == vehicle_id:
                # print(class_id, confident_score, x_min, y_min, x_max, y_max)
                x_min = max(1, x_min)
                y_min = max(1, y_min)
                x_max = min(width, x_max)
                y_max = min(height, y_max)
                occurrence_list.append([int(frame_id), x_min, y_min, x_max, y_max])
        if len(occurrence_list) <= 1:
            continue
        moi_id, frame_id, bbox = solve(occurrence_list, moi_list, roi, height, width, moi_weight)
        if moi_id != -1:
            ans[frame_id].append([moi_id, vehicle_id, class_ids[vehicle_id], bbox])

    print("Writing results to %s.txt" % video_id)
    with open("%s.txt" % video_id, "w") as f:
        for frame_id in tqdm(range(num_frame)):
            for moi_id, _, v_type, _ in ans[frame_id]:
                f.write(f"{video_ids[video_id]},{frame_id + 1},{moi_id},{v_type}\n")

    print("Creating video")
    output = cv2.VideoWriter("output_%s.mp4" % video_id, cv2.VideoWriter_fourcc("m", "p", "4", "v"), fps, (width, height))
    for frame_id in tqdm(range(num_frame)):
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
    video_ids = {}
    with open(os.path.join("videos", "list_video_id.txt"), "r") as f:
        lines = f.read().splitlines()
        for line in lines:
            video_id, name = line.split(" ")
            video_ids[name.split(".")[0]] = video_id

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

    cam_id = "cam_11"
    process(
        os.path.join("tracking-results", "info_%s.mp4.npy" % cam_id),
        cam_id,
        cam_id,
        os.path.join("videos", "%s.mp4" % cam_id)
    )