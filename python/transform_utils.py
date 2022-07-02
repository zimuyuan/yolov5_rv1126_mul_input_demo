from rknn.api import RKNN

# to run '$ python *.py' files in subdirectories

from yolov5_inference_utils import yolov5_model_inference

class RKNN_Model:
    def __init__(self, model_path, preprocess=False, verbose=False) -> None:
        self.model = model_path
        self.preprocess = preprocess
        self.rknn = RKNN(verbose=verbose, verbose_file="./verbose.log")
        self.load_mode()
    
    def load_mode(self):
        # Load model
        print('--> Loading model: %s' % self.model)
        if self.model.endswith('onnx'):
            ret = self.rknn.load_onnx(self.model)
        elif self.model.endswith('rknn'):
            ret = self.rknn.load_rknn(self.model)
        if ret != 0:
            print('load model failed!')
            exit(ret)

    def init_runtime(self):
        print('--> Init runtime environment...')
        ret = self.rknn.init_runtime()
        if ret != 0:
            print('Init runtime environment failed!')
            exit(ret)

    def infer(self, net_input_size, mul_inputs, images_path=None, result_path=None, infer=True):
        if infer:
            self.init_runtime()
            yolov5_model_inference(tool='rknn', model=self.rknn, net_input_size=net_input_size, mul_inputs=mul_inputs,
                              image_path=images_path, res_path=result_path, preprocess=self.preprocess)

    def release(self):
        self.rknn.release()


class RKNN_Exporter(RKNN_Model):
    def __init__(self, model_path, quant_dataset, mean_values, std_values,
                 pre_compile=False, quant=True, preprocess=True, verbose=False):
        self.pre_compile = pre_compile 
        self.quant = quant
        self.dataset = quant_dataset
        self.mean = mean_values
        self.std = std_values
        self.model = model_path
        self.preprocess = preprocess
        self.rknn = RKNN(verbose=verbose, verbose_file="./verbose.log")
        self.rknn.inference

    def rknn1_config(self, chip, preprocess=True, out_opt=True):
        if preprocess:
            self.rknn.config(target_platform=chip, batch_size=8, quantize_input_node=True, output_optimize=out_opt)
        else:
            self.rknn.config(mean_values=self.mean, std_values=self.std, 
                            target_platform=chip, batch_size=8, quantize_input_node=True, output_optimize=out_opt)


    def rknn1_build(self):
        # Build model
        print('--> Building model...')
        ret = self.rknn.build(do_quantization=self.quant, dataset=self.dataset, pre_compile=self.pre_compile)
        if ret != 0:
            print('build model failed!')
            exit(ret)


    def rknn2_config(self, chip, preprocess=False):

        if preprocess:
            self.rknn.config( target_platform=chip, quantized_method='channel')
        else:
            self.rknn.config(mean_values=self.mean, std_values=self.std, 
                            target_platform=chip, quantized_method='channel')


    def rknn2_build(self):
        print('--> Building model...')
        ret = self.rknn.build(do_quantization=self.quant, dataset=self.dataset)
        if ret != 0:
            print('build model failed!')
            exit(ret)


    def export_model(self, save_path, export=True):
        # Export rknn model
        if export:
            print('--> Export rknn model: %s' % save_path)
            ret = self.rknn.export_rknn(save_path)
            if ret != 0:
                print('Export rknn model failed!')
                exit(ret)


    def analysis_model(self, analysis_dataset, outpur_dir='./snapshot', analysis=False):
        if analysis:
            print('--> Analysis rknn model...')
            ret = self.rknn.accuracy_analysis(inputs=analysis_dataset, output_dir=outpur_dir)
            if ret != 0:
                print('Accuracy analysis failed!')
                exit(ret)


    def rknn1_export_model(self, export_path, chip, out_opt=True):
        self.rknn1_config(chip, out_opt)
        self.load_mode()
        self.rknn1_build()
        self.export_model(export_path)


    def rknn2_export_model(self, export_path, chip):
        self.rknn2_config(chip)
        self.load_mode()
        self.rknn2_build()
        self.export_model(export_path)
