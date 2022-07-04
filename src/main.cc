// Copyright (c) 2021 by Rockchip Electronics Co., Ltd. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

/*-------------------------------------------
                Includes
-------------------------------------------*/
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <dlfcn.h>
#include <vector>


#include "opencv2/core/core.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"

#define _BASETSD_H

#define STB_IMAGE_IMPLEMENTATION
#include "stb/stb_image.h"
#define STB_IMAGE_RESIZE_IMPLEMENTATION
#include <stb/stb_image_resize.h>

#undef cimg_display
#define cimg_display 0
#include "CImg/CImg.h"

#include "drm_func.h"
#include "rga_func.h"
#include "rknn_api.h"
#include "postprocess.h"

#define PERF_WITH_POST 0

using namespace cimg_library;
/*-------------------------------------------
                  Functions
-------------------------------------------*/

inline const char* get_type_string(rknn_tensor_type type)
{
    switch(type) {
    case RKNN_TENSOR_FLOAT32: return "FP32";
    case RKNN_TENSOR_FLOAT16: return "FP16";
    case RKNN_TENSOR_INT8: return "INT8";
    case RKNN_TENSOR_UINT8: return "UINT8";
    case RKNN_TENSOR_INT16: return "INT16";
    default: return "UNKNOW";
    }
}

inline const char* get_qnt_type_string(rknn_tensor_qnt_type type)
{
    switch(type) {
    case RKNN_TENSOR_QNT_NONE: return "NONE";
    case RKNN_TENSOR_QNT_DFP: return "DFP";
    case RKNN_TENSOR_QNT_AFFINE_ASYMMETRIC: return "AFFINE";
    default: return "UNKNOW";
    }
}

inline const char* get_format_string(rknn_tensor_format fmt)
{
    switch(fmt) {
    case RKNN_TENSOR_NCHW: return "NCHW";
    case RKNN_TENSOR_NHWC: return "NHWC";
    default: return "UNKNOW";
    }
}

static void dump_tensor_attr(rknn_tensor_attr *attr)
{
    printf("  index=%d, name=%s, n_dims=%d, dims=[%d, %d, %d, %d], n_elems=%d, size=%d, fmt=%s, type=%s, qnt_type=%s, "
           "zp=%d, scale=%f\n",
           attr->index, attr->name, attr->n_dims, attr->dims[3], attr->dims[2], attr->dims[1], attr->dims[0],
           attr->n_elems, attr->size, get_format_string(attr->fmt), get_type_string(attr->type),
           get_qnt_type_string(attr->qnt_type), attr->zp, attr->scale);
}

double __get_us(struct timeval t) { return (t.tv_sec * 1000000 + t.tv_usec); }

static unsigned char *load_data(FILE *fp, size_t ofst, size_t sz)
{
    unsigned char *data;
    int ret;

    data = NULL;

    if (NULL == fp)
    {
        return NULL;
    }

    ret = fseek(fp, ofst, SEEK_SET);
    if (ret != 0)
    {
        printf("blob seek failure.\n");
        return NULL;
    }

    data = (unsigned char *)malloc(sz);
    if (data == NULL)
    {
        printf("buffer malloc failure.\n");
        return NULL;
    }
    ret = fread(data, 1, sz, fp);
    return data;
}

static unsigned char *load_model(const char *filename, int *model_size)
{

    FILE *fp;
    unsigned char *data;

    fp = fopen(filename, "rb");
    if (NULL == fp)
    {
        printf("Open file %s failed.\n", filename);
        return NULL;
    }

    fseek(fp, 0, SEEK_END);
    int size = ftell(fp);

    data = load_data(fp, 0, size);

    fclose(fp);

    *model_size = size;
    return data;
}

static int saveFloat(const char *file_name, float *output, int element_size)
{
    FILE *fp;
    fp = fopen(file_name, "w");
    for (int i = 0; i < element_size; i++)
    {
        fprintf(fp, "%.6f\n", output[i]);
    }
    fclose(fp);
    return 0;
}

static int saveuint8(const char *file_name, u_int8_t* savebuf, int element_size)
{
    FILE *fp;
    fp = fopen(file_name, "w");
    for (int i = 0; i < element_size; i++)
    {
        fprintf(fp, "%hhu\n", savebuf[i]);
    }
    fclose(fp);
    return 0;
}


static uint8_t *ReadTensorData
    (
    const char *name,
    uint32_t width,
    uint32_t height,
    uint32_t channels
    )
{
    uint8_t *tensorData;
    FILE *tensorFile;
    int ival = 0.0;
    uint32_t   sz;

    tensorData = NULL;
    tensorFile = fopen(name, "rb");

    sz = width * height * channels;

    tensorData = (uint8_t *)malloc(sz * sizeof(uint8_t));
    memset(tensorData, 0, sz);

    for(int i = 0; i < sz; i++)
    {
        if(fscanf( tensorFile, "%d ", &ival ) != 1)
        {
            printf("Read tensor file fail.\n");
            printf("Please check file lines or if the file contains illegal characters\n");
            goto final;
        }
        // memcpy(tensorData + i, (uint8_t)(ival), 1);
        tensorData[i] = (uint8_t)ival;
    }

    if(tensorFile)fclose(tensorFile);
    return tensorData;
final:
    if(tensorFile)fclose(tensorFile);
    return NULL;
}


/*-------------------------------------------
                  Main Functions
-------------------------------------------*/
int main(int argc, char **argv)
{
    int status = 0;
    char *model_name = NULL;
    rknn_context ctx;
    void *drm_buf = NULL;
    int drm_fd = -1;
    int buf_fd = -1; // converted from buffer handle
    unsigned int handle;
    size_t actual_size = 0;
    int img_width = 0;
    int img_height = 0;
    int img_channel = 0;
    const float nms_threshold = NMS_THRESH;
    const float box_conf_threshold = BOX_THRESH;
    struct timeval start_time, stop_time;
    int ret;

    int input_width = 640;
    int input_height =448; 
    int input_channel = 3;


    if (argc != 4)
    {
        printf("Usage: %s <rknn model> <bmp> <1 ro 0 preprcess_layer>\n", argv[0]);
        return -1;
    }

    printf("post process config: box_conf_threshold = %.2f, nms_threshold = %.2f\n",
           box_conf_threshold, nms_threshold);

    model_name = (char *)argv[1];
    char *image_name = argv[2];
    int preprcess_layer = atoi(argv[3]);

    printf("Read %s ...\n", image_name);
    cv::Mat orig_img = cv::imread(image_name, cv::IMREAD_COLOR);
    if (!orig_img.data)
    {
        printf("cv::imread %s fail!\n", image_name);
        return -1;
    }
    cv::Mat img = cv::Mat(orig_img);

    // cv::cvtColor(orig_img, img, cv::COLOR_BGR2RGB);
    img_width = img.cols;
    img_height = img.rows;
    printf("img width = %d, height = %d, channel=%d\n", img_width, img_height, img.channels());


    /* Create the neural network */
    printf("Loading mode...\n");
    int model_data_size = 0;
    unsigned char *model_data = load_model(model_name, &model_data_size);
    ret = rknn_init(&ctx, model_data, model_data_size, 0);
    if (ret < 0)
    {
        printf("rknn_init error ret=%d\n", ret);
        return -1;
    }

    rknn_sdk_version version;
    ret = rknn_query(ctx, RKNN_QUERY_SDK_VERSION, &version, sizeof(rknn_sdk_version));
    if (ret < 0)
    {
        printf("rknn query sdk version error ret=%d\n", ret);
        return -1;
    }
    printf("sdk version: %s driver version: %s\n", version.api_version, version.drv_version);

    rknn_input_output_num io_num;
    ret = rknn_query(ctx, RKNN_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num));
    if (ret < 0)
    {
        printf("rknn query io num error ret=%d\n", ret);
        return -1;
    }
    printf("model input num: %d, output num: %d\n", io_num.n_input, io_num.n_output);

    rknn_tensor_attr input_attrs[io_num.n_input];
    memset(input_attrs, 0, sizeof(input_attrs));
    for (int i = 0; i < io_num.n_input; i++)
    {
        input_attrs[i].index = i;
        ret = rknn_query(ctx, RKNN_QUERY_INPUT_ATTR, &(input_attrs[i]), sizeof(rknn_tensor_attr));
        if (ret < 0)
        {
            printf("rknn query ininput attrs error ret=%d\n", ret);
            return -1;
        }
        dump_tensor_attr(&(input_attrs[i]));
    }

    rknn_tensor_attr output_attrs[io_num.n_output];
    memset(output_attrs, 0, sizeof(output_attrs));
    for (int i = 0; i < io_num.n_output; i++)
    {
        output_attrs[i].index = i;
        ret = rknn_query(ctx, RKNN_QUERY_OUTPUT_ATTR, &(output_attrs[i]),
                         sizeof(rknn_tensor_attr));
        dump_tensor_attr(&(output_attrs[i]));
        if(output_attrs[i].qnt_type != RKNN_TENSOR_QNT_AFFINE_ASYMMETRIC || output_attrs[i].type != RKNN_TENSOR_UINT8)
        {
            fprintf(stderr,"The Demo required for a Affine asymmetric u8 quantized rknn model, but output quant type is %s, output data type is %s\n", 
                    get_qnt_type_string(output_attrs[i].qnt_type),get_type_string(output_attrs[i].type));
            return -1;
        }
    }

    int real_input_channel = 3;
    int real_input_width = 0;
    int real_input_height = 0;

    if (io_num.n_input == 4)
    {
        real_input_width = 336;
        real_input_height = 240;
    }
    else
    {
        real_input_width = 640;
        real_input_height = 448;
    }

    printf("model input height=%d, width=%d, channel=%d\n", real_input_height, real_input_width, real_input_channel);

    rknn_input inputs[io_num.n_input];
    for (int i = 0; i < io_num.n_input; i++)
    {
        memset(inputs + i, 0, sizeof(inputs[i]));
        inputs[i].index = i;
        inputs[i].type = RKNN_TENSOR_UINT8;
        inputs[i].size = real_input_width * real_input_height * 3;
        inputs[i].fmt = RKNN_TENSOR_NHWC;
        inputs[i].pass_through = preprcess_layer;
    }

    if (io_num.n_input == 4)
    {
        int w = real_input_width;
        int h = real_input_height;
        // // start_x start_y w h
        cv::Mat ROI0 = cv::Mat(img, cv::Rect(0, 0, w, h)).clone();
        cv::Mat ROI1 = cv::Mat(img, cv::Rect(w - 32, 0, w, h)).clone();
        cv::Mat ROI2 = cv::Mat(img, cv::Rect(0, h - 32, w, h)).clone();
        cv::Mat ROI3 = cv::Mat(img, cv::Rect(w - 32, h - 32, w, h)).clone();
        inputs[0].buf = ROI0.data;
        inputs[1].buf = ROI1.data;
        inputs[2].buf = ROI2.data;
        inputs[3].buf = ROI3.data;
        
        // saveuint8("images/cv_in_0.tensor", (u_int8_t*)inputs[0].buf, inputs[0].size);
        // saveuint8("images/cv_in_1.tensor", (u_int8_t*)inputs[1].buf, inputs[1].size);
        // saveuint8("images/cv_in_2.tensor", (u_int8_t*)inputs[2].buf, inputs[2].size);
        // saveuint8("images/cv_in_3.tensor", (u_int8_t*)inputs[3].buf, inputs[3].size);
        
        // cv::imwrite("images/roi_0.bmp", ROI0);
        // cv::imwrite("images/roi_1.bmp", ROI1);
        // cv::imwrite("images/roi_2.bmp", ROI2);
        // cv::imwrite("images/roi_3.bmp", ROI3);
        // printf("read data form tensor.\n");
        // inputs[0].buf = ReadTensorData("images/in_0.tensor", real_input_width, real_input_height, 3);
        // inputs[1].buf = ReadTensorData("images/in_1.tensor", real_input_width, real_input_height, 3);
        // inputs[2].buf = ReadTensorData("images/in_2.tensor", real_input_width, real_input_height, 3);
        // inputs[3].buf = ReadTensorData("images/in_3.tensor", real_input_width, real_input_height, 3);
    }
    else
    {
        inputs[0].buf = img.data;
    }

    gettimeofday(&start_time, NULL);
    rknn_inputs_set(ctx, io_num.n_input, inputs);

    rknn_output outputs[io_num.n_output];
    memset(outputs, 0, sizeof(outputs));
    for (int i = 0; i < io_num.n_output; i++)
    {
        outputs[i].want_float = 0;
    }

    ret = rknn_run(ctx, NULL);
    ret = rknn_outputs_get(ctx, io_num.n_output, outputs, NULL);
    gettimeofday(&stop_time, NULL);
    printf("once run use %f ms\n", (__get_us(stop_time) - __get_us(start_time)) / 1000);

    //post process
    float scale_w = (float)input_width / img_width;
    float scale_h = (float)input_height / img_height;

    detect_result_group_t detect_result_group;
    std::vector<float> out_scales;
    std::vector<uint32_t> out_zps;
    for (int i = 0; i < io_num.n_output; ++i)
    {
        out_scales.push_back(output_attrs[i].scale);
        out_zps.push_back(output_attrs[i].zp);
    }
    post_process((uint8_t *)outputs[0].buf, (uint8_t *)outputs[1].buf, (uint8_t *)outputs[2].buf, input_height, input_width,
                 box_conf_threshold, nms_threshold, scale_w, scale_h, out_zps, out_scales, &detect_result_group);

    // Draw Objects
    char text[256];
    const unsigned char blue[] = {0, 0, 255};
    const unsigned char white[] = {255, 255, 255};
    for (int i = 0; i < detect_result_group.count; i++)
    {
        detect_result_t *det_result = &(detect_result_group.results[i]);
        sprintf(text, "%s %.2f", det_result->name, det_result->prop);
        printf("%s @ (%d %d %d %d) %f\n",
               det_result->name,
               det_result->box.left, det_result->box.top, det_result->box.right, det_result->box.bottom,
               det_result->prop);
        int x1 = det_result->box.left;
        int y1 = det_result->box.top;
        int x2 = det_result->box.right;
        int y2 = det_result->box.bottom;
        //draw box
        rectangle(orig_img, cv::Point(x1, y1), cv::Point(x2, y2), cv::Scalar(255, 0, 0, 255), 3);
        putText(orig_img, text, cv::Point(x1, y1 + 12), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0));
    }
    if (io_num.n_input == 4)
        imwrite("./c_mul_input_out.jpg", orig_img);
    else
        imwrite("./c_single_out.jpg", orig_img);

    ret = rknn_outputs_release(ctx, io_num.n_output, outputs);

    // loop test
    int test_count = 10;
    gettimeofday(&start_time, NULL);
    for (int i = 0; i < test_count; ++i)
    {
        rknn_inputs_set(ctx, io_num.n_input, inputs);
        ret = rknn_run(ctx, NULL);
        ret = rknn_outputs_get(ctx, io_num.n_output, outputs, NULL);
#if PERF_WITH_POST
        post_process((uint8_t *)outputs[0].buf, (uint8_t *)outputs[1].buf, (uint8_t *)outputs[2].buf, input_height, input_width,
                     box_conf_threshold, nms_threshold, scale_w, scale_h, out_zps, out_scales, &detect_result_group);
#endif
        ret = rknn_outputs_release(ctx, io_num.n_output, outputs);
    }
    gettimeofday(&stop_time, NULL);
    printf("loop count = %d , average run  %f ms\n", test_count,
           (__get_us(stop_time) - __get_us(start_time)) / 1000.0 / test_count);

    // release
    ret = rknn_destroy(ctx);

    if (model_data)
    {
        free(model_data);
    }

    // for (int i = 0; i < 4; ++i)
    // {
    //     if (inputs[i].buf)
    //     {
    //         free(inputs[i].buf);
    //     }
    // }
    return 0;
}
