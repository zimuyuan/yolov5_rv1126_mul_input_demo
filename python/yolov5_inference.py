import argparse


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--format', type=str, default='onnx', help='model format')
    parser.add_argument('--tool', type=str, default='rknn1', help='rknn toolkit')
    parser.add_argument('--chip_index', type=int, default=0, help='select target chip')
    parser.add_argument('--preprocess', action='store_true', default=False, help='add image preprocess layer, benefit for decreasing rknn_input_set time-cost')
    parser.add_argument('--size_index', type=int, default=1, help='select infer model input size')
    parser.add_argument('--img_ch', type=int, default=3, help='input image channel')
    parser.add_argument('--model_path', type=str, default='../model/mul_input/', help='model path')
    parser.add_argument('--mul_input', action='store_true', default=False, help='multiply input model')
    args = parser.parse_args()

    net_input_size = [[320, 224], [640, 448], [1280, 896], [1920, 1344], [2784, 1856]]
    net_input_size = net_input_size[args.size_index]

    targe_chip = ['rv1126', 'rk3399pro'] if args.tool == 'rknn1' else ['rk3566', 'rk3588', 'rv1106']
    
    model_path = args.model_path + '/%s/' % args.tool

    model = "yolov5n_%d_%d." % (net_input_size[0], net_input_size[1]) + args.format
    net_input_size = [net_input_size[0], net_input_size[1], args.img_ch]
    if args.format == 'onnx':
        import onnxruntime
        from yolov5_inference_utils import yolov5_model_inference
        model_path += model
        print('--> Loading model:', model_path)
        onnx_session = onnxruntime.InferenceSession(model_path)
        yolov5_model_inference('onnx', onnx_session, net_input_size, args.mul_input, preprocess=args.preprocess)
    else:
        from transform_utils import RKNN_Model
        model = '/%s/' % targe_chip[args.chip_index] + model
        model_path += model
        # Create RKNN object
        rknn = RKNN_Model(model_path=model_path, preprocess=args.preprocess)
        rknn.infer(net_input_size, mul_inputs=args.mul_input)
        rknn.release()
