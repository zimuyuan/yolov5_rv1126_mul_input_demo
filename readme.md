
## 编译环境 
- demo 参照 1.7.1中的 examples 将次工程放到 examples下
- 修改 CMakeLists.txt 中 opencv 路径
- ./build.sh
```
include_directories("/home/lichangyuan/soft_tool/vbox_env/opt_env/armhf/opencv/include/opencv4")
set(OpenCV_DIR /home/lichangyuan/soft_tool/vbox_env/opt_env/armhf/opencv/lib/cmake/opencv4)
find_package(OpenCV REQUIRED)
```

## C API 运行
### rv1126 中 C API 推理多输入模型
- `cd rknn_yolov5_demo `
- 命令 `./rknn_yolov5_demo <model> <images> <inputs_pass_through>`
- model 表示所要推理的模型
- image 表示所要推理的图片
- inputs_pass_through 为 0 或 1 ,表示推理的模型添加预处理层时为 1 其余为 0

- 推理单输入模型  
`./rknn_yolov5_demo model/single_input/rknn1/rv1126/yolov5n_640_448.rknn  images/bus_640_448.bmp 0`

- 推理多输入模型  
`./rknn_yolov5_demo model/mul_input/rknn1/rv1126/yolov5n_640_448.rknn  images/bus_640_448.bmp 0`

- 推理单输入加预处理模型  
`./rknn_yolov5_demo model/preprocess_single_input/rknn1/rv1126/yolov5n_640_448.rknn  images/bus_640_448.bmp 1`

- 推理多输入加预处理模型  
`./rknn_yolov5_demo model/preprocess_mul_input/rknn1/rv1126/yolov5n_640_448.rknn  images/bus_640_448.bmp 1`


## python API 在 PC 模推理模型
- 到 pthon 目录下 `cd python`
- 运行 yolov5_inference.py 脚本
参数
--format 表示推理的模型类型可以为 rknn 或 onnx
--mul_input 推理多输入时加上这个参数即可
--model_path 所推理模型路径
--preprocess 推理添加预处理层的模型时加上这个参数

### 用 rknn_toolkit1 在 PC 模拟器推理 rknn 模型 

- 单输入输入的 rknn 模型
`python yolov5_inference.py --format=rknn --model_path=../model/single_input/`

- 多输入的 rknn 模型
`python yolov5_inference.py --format=rknn --model_path=../model/mul_input/ --mul_input`

- 单输入加预处理层的 rknn 模型
`python yolov5_inference.py --format=rknn --model_path=../model/preprocess_single_input/ --preprocess`

- 多输入加预处理层的 rknn 模型
`python yolov5_inference.py --format=rknn --model_path=../model/preprocess_mul_input/ --preprocess --mul_input`


### onnxruntime 推理 onnx 模型

- 单输入的 onnx 模型
`python yolov5_inference.py --format=onnx  --model_path=../model/single_input/`
- 多输入的 onnx 模型
`python yolov5_inference.py --format=onnx --model_path=../model/mul_input/ --mul_input`
- 单输入加预处理层的 onnx 模型
`python yolov5_inference.py --format=onnx  --model_path=../model/preprocess_single_input/ --preprocess`
- 多输入加预处理层的 onnx 模型
`python yolov5_inference.py --format=onnx --model_path=../model/preprocess_mul_input/ --preprocess --mul_input`
