from pickletools import uint8
import cv2
import numpy as np
import time
import os


from inference_utils import polygon_plot_one_box, letterbox, yolov5_post_process

def split_mul_input(img, boundary):
    img0 = img[:,:, 0:img.shape[2] // 2 + boundary, 0:img.shape[3] // 2 + boundary]
    img1 = img[:,:, 0:img.shape[2] // 2 + boundary, img.shape[3] // 2 - boundary:]
    img2 = img[:,:, img.shape[2] // 2 - boundary:, 0:img.shape[3] // 2 + boundary]
    img3 = img[:,:, img.shape[2] // 2 - boundary:, img.shape[3] // 2 - boundary:]
    return [img0, img1, img2, img3]

def get_model_info():
    
    conf_thres = 0.2
    iou_thres = 0.3

    label_names = ["person", "bicycle", "car", "motorbike ", "aeroplane ", "bus ", "train", "truck ", "boat", "traffic light",
    "fire hydrant", "stop sign ", "parking meter", "bench", "bird", "cat", "dog ", "horse ", "sheep", "cow", "elephant",
    "bear", "zebra ", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
    "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife ",
    "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza ", "donut", "cake", "chair", "sofa",
    "pottedplant", "bed", "diningtable", "toilet ", "tvmonitor", "laptop	", "mouse	", "remote ", "keyboard ", "cell phone", "microwave ",
    "oven ", "toaster", "sink", "refrigerator ", "book", "clock", "vase", "scissors ", "teddy bear ", "hair drier", "toothbrush "]
    anchors = [[10, 13],  [16, 30],  [33,23],
                [30, 61],  [62, 62],  [45,59],
                [116,90],  [156,198], [373,326]]
    masks = [[0, 1, 2], [3, 4, 5], [6, 7, 8]]

    return conf_thres, iou_thres, label_names, masks, anchors


def tool_inference(tool, model, input, mul_inputs, preprocess=False):

    if mul_inputs:
        # [1HWC] - > [1CWH] -> [4 * 1Cwh] -> [4 * 1hwC]
        input = np.transpose(input, (0, 3, 1, 2))
        input = split_mul_input(input, 16)
        input = [np.transpose(i, (0, 2, 3, 1)) for i in input]
    else:
        input = [input]

    if tool == 'onnx':
        if not preprocess:
            # 0.0~255.0 NHWC -> 0.0~1.0 NCWH
            input = [np.transpose(i, (0, 3, 1, 2)) / 255. for i in input]
        input = {model.get_inputs()[i].name: input[i] for i in range(len(input))}
        outputs = model.run(None, input)
    else:
        input = [i.astype(np.uint8) for i in input]
        # for idx, in_data in enumerate(input):
        #     cv2.imwrite("py_crop_%d.bmp" % idx, in_data[0])
        #     in_data.tofile('in_{}.tensor'.format(idx), '\n')
        # 0~255 NHWC
        inputs_pass_through = [preprocess] * len(input)
        outputs = model.inference(input, inputs_pass_through = inputs_pass_through)
    return outputs


def yolov5_model_inference(tool, model, net_input_size, mul_inputs = False, preprocess=True,
                      image_path = '../images/',
                      res_path = '../result/',
                      model_class=2):

    SIMPLE_NET = True

    conf_thres, iou_thres, label_names, masks, anchors = get_model_info()

    post_process = yolov5_post_process

    if image_path is None:
        image_path = '../images/'
    if res_path is None:
        res_path = '../result/'

    image_list = os.listdir(image_path)
    if not os.path.exists(res_path):
        os.makedirs(res_path)

    for i, image_name in enumerate(image_list):

        print("="*120)
        print(image_path + image_name)

        img0 = cv2.imread(image_path + image_name)

        # print("img0.shape: ", img0.shape)
        img, ratio, (pad_left, pad_top) = letterbox(img0, net_input_size)
        if net_input_size[2] == 1:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = np.expand_dims(img, 2)
        
        # print("img.shape: ", img.shape)
        img = np.expand_dims(img, 0).astype(np.float32)

        t0 = time.time()
        outputs = tool_inference(tool, model, img, mul_inputs, preprocess=preprocess)
        print("infer time: ", time.time() - t0)

        outputs_data = list()
        for output in outputs:
            if SIMPLE_NET:
                # [255 20 20] -> [3 85 20 20] -> [20 20 3 85]
                output = output.reshape([len(masks[0]), -1]+list(output.shape[-2:]))
                output = np.transpose(output, (2, 3, 0, 1))
            else:
                # [3 20 20 85] -> [20 20 3 85]
                output = np.transpose(output, (1, 2, 0, 3))
            outputs_data.append(output)
        boxes, classes, scores = post_process(outputs_data, anchors, masks, net_input_size[:2], pad_left, pad_top, ratio, conf_thres, iou_thres)
        if boxes is None:
            continue
        for box, score, cl in zip(boxes, scores, classes):
            # 计算预测框 4 个坐标点在原图中的具体位置
            # print(box)
            if len(box == 4):
                box =  np.array([box[0], box[1], box[2], box[1], box[2], box[3], box[0], box[3]])

            box = box.reshape(-1, 1, 2).astype(np.int32)

            # label = (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
            label=label_names[cl] + ':%.2f' % score 
            polygon_plot_one_box(box, img0, label=label)
            print(label)

        cv2.imwrite(res_path + image_name, img0)
        print('inference reuslt save->', res_path + image_name)
