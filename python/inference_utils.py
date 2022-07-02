import cv2
import numpy as np
import random


def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def tanh(x):
    return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))

def letterbox(img, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[1], new_shape[1] / shape[0])
    if not scaleup:  # only scale down, do not scale up (for better test mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))

    dw, dh = new_shape[0] - new_unpad[0], new_shape[1] - new_unpad[1]  # wh padding


    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))


    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return img, ratio, (left, top)


def filter_boxes(boxes, box_confidences, box_class_probs=0.5, conf_thres=0.3):

    # print('boxes.shape:', boxes.shape)
    box_scores = box_confidences * box_class_probs  # 条件概率， 在该cell存在物体的概率的基础上是某个类别的概率
    # print('box_scores.shape:', box_scores.shape)

    box_classes = np.argmax(box_scores, axis=-1)  # 找出概率最大的类别索引

    # print('box_classes.shape:', box_classes.shape)
    box_class_scores = np.max(box_scores, axis=-1)  # 最大类别对应的概率值

    # print('box_class_scores.shape:', box_class_scores.shape)
    pos = np.where(box_class_scores >= conf_thres)  # 找出概率大于阈值的item

    boxes = boxes[pos]
    classes = box_classes[pos]
    scores = box_class_scores[pos]
    return boxes, classes, scores


def draw(image, boxes, scores, classes, label_names):
    """Draw the boxes on the image.

    # Argument:
        image: original image.
        boxes: ndarray, boxes of objects.
        classes: ndarray, classes of objects.
        scores: ndarray, scores of objects.
        all_classes: all classes name.
    """
    for box, score, cl in zip(boxes, scores, classes):
        top, left, right, bottom = box
        # print('class: {}, score: {}'.format(CLASSES[cl], score))
        print('class: %s, score: %11.6f' % (label_names[cl], score))
        # print('box coordinate left,top,right,down: [{}, {}, {}, {}]'.format(top, left, right, bottom))
        print('box coordinate left,top,right,down: [%11.6f, %11.6f, %11.6f, %11.6f]' % (top, left, right, bottom))
        top = int(top)
        left = int(left)
        right = int(right)
        bottom = int(bottom)

        cv2.rectangle(image, (top, left), (right, bottom), (255, 0, 0), 2)
        cv2.putText(image, '{0} {1:.2f}'.format(label_names[cl], score),
                    (top, left - 6),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 0, 255), 2)


# ======================================================================================================================
def polygon_plot_one_box(c, img, color=(0, 0, 255), label=None, line_thickness=None):
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
    color = color or [random.randint(0, 255) for _ in range(3)]

    cv2.polylines(img, pts=[c], isClosed=True, color=color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c1 = (int(c[:, 0, 0].mean()), int(c[:, 0, 1].mean()))
        c2 = (int(c1[0] + t_size[0]), int(c1[1] - t_size[1] - 3))
        im_origin = img.copy()
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)
        alpha = 0.5    # opacity of 0.5
        cv2.addWeighted(img, alpha, im_origin, 1 - alpha, 0, img)



# ======================================================================================================================


def xywh2xyxy(x):
    # Convert [x, y, w, h] to [x1, y1, x2, y2]
    y = np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y

def process(input, anchors, masks, net_input_size, pad_left, pad_top, ratio):

    anchors = [anchors[i] for i in masks]
    grid_h, grid_w = map(int, input.shape[0:2])
    # print("h:",grid_h)
    # print("w:",grid_w)
    box_xy = sigmoid(input[..., :2])
    box_wh = sigmoid(input[..., 2:4])
    box_confidence = sigmoid(input[..., 4])
    box_confidence = np.expand_dims(box_confidence, axis=-1)
    box_class_probs = sigmoid(input[..., 5:]) * box_confidence


    col = np.tile(np.arange(0, grid_w), grid_h).reshape(-1, grid_w)
    row = np.tile(np.arange(0, grid_h).reshape(-1, 1), grid_w)

    # 因为每个 head 有对应多组 anchor 所以, grid 也需要 repeat 每组 anchors 的个数次
    col = col.reshape((grid_h, grid_w, 1, 1)).repeat(len(anchors), axis=-2)
    row = row.reshape((grid_h, grid_w, 1, 1)).repeat(len(anchors), axis=-2)

    grid = np.concatenate((col, row), axis=-1)

    # 计算得到 center_x center_y 在 -0.5~1.5 范围之间，并加上偏移
    box_xy = (box_xy * 2 - 0.5 + grid)
    box_xy /= (grid_w, grid_h)  # 计算原尺寸的中心
    box_xy *= net_input_size
    box_xy -= (pad_left, pad_top)
    box_xy /= ratio
    
    # 计算得到 box_w box_h 在0~4的范围内 并乘以对应的 anchor 尺寸
    box_wh = (box_wh * 2) ** 2 * anchors
    box_wh /= ratio  # 计算原尺寸的宽高

    box = np.concatenate((box_xy, box_wh), axis=-1)

    return box, box_confidence, box_class_probs


def nms_boxes(boxes, scores, iou_thres):
    """Suppress non-maximal boxes.

    # Arguments
        boxes: ndarray, boxes of objects.
        scores: ndarray, scores of objects.

    # Returns
        keep: ndarray, index of effective boxes.
    """
    x = boxes[:, 0]
    y = boxes[:, 1]
    w = boxes[:, 2] - boxes[:, 0]
    h = boxes[:, 3] - boxes[:, 1]

    areas = w * h
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)

        xx1 = np.maximum(x[i], x[order[1:]])
        yy1 = np.maximum(y[i], y[order[1:]])
        xx2 = np.minimum(x[i] + w[i], x[order[1:]] + w[order[1:]])
        yy2 = np.minimum(y[i] + h[i], y[order[1:]] + h[order[1:]])

        w1 = np.maximum(0.0, xx2 - xx1 + 0.00001)
        h1 = np.maximum(0.0, yy2 - yy1 + 0.00001)
        inter = w1 * h1

        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        inds = np.where(ovr <= iou_thres)[0]
        order = order[inds + 1]
    keep = np.array(keep)
    return keep


def yolov5_post_process(outputs, anchors, masks, net_input_size, pad_left, pad_top, ratio, conf_thres=0.5, iou_thres=0.3):

    boxes, classes, scores = [], [], []
    for input, mask in zip(outputs, masks):
        b, c, s = process(input, anchors, mask, net_input_size, pad_left, pad_top, ratio)
        b, c, s = filter_boxes(b, c, s, conf_thres)
        boxes.append(b)
        classes.append(c)
        scores.append(s)

    boxes = np.concatenate(boxes)
    boxes = xywh2xyxy(boxes)
    classes = np.concatenate(classes)
    scores = np.concatenate(scores)

    nboxes, nclasses, nscores = [], [], []
    for c in set(classes):
        inds = np.where(classes == c)
        b = boxes[inds]
        c = classes[inds]
        s = scores[inds]

        keep = nms_boxes(b, s, iou_thres)

        nboxes.append(b[keep])
        nclasses.append(c[keep])
        nscores.append(s[keep])

    if not nclasses and not nscores:
        return None, None, None

    boxes = np.concatenate(nboxes)
    classes = np.concatenate(nclasses)
    scores = np.concatenate(nscores)

    return boxes, classes, scores