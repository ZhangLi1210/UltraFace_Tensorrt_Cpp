//
// Created by lizhang on 2023/10/29.
//

#ifndef ULTRAFACE_H
#define ULTRAFACE_H
#define clip(x, y) (x < 0 ? 0 : (x > y ? y : x))

#include "NvInferPlugin.h"
#include "common.h"
#include "fstream"


using namespace FaceDetect;


class UltraFace {
public:
    explicit UltraFace(const std::string& UltraFace_engine_file_path,int input_width, int input_length, float score_threshold_ = 0.7, float iou_threshold_ = 0.3, int topk_ = -1);
    ~UltraFace();

    void                 make_pipe();
    void                 infer(std::vector<void*> device_ptrs,std::vector<void*> host_ptrs,int num_inputs,int num_outputs,std::vector<Binding> output_bindings);
    void                 detect(cv::Mat im, std::vector<FaceInfo> &face_list);
    void                 generateBBox(std::vector<FaceInfo> &bbox_collection, float * scores, float *boxes, float score_threshold, int num_anchors);
    void                 nms(std::vector<FaceInfo> &input, std::vector<FaceInfo> &output, int type = blending_nms);
    cv::Mat              resizeAndPadingImage(cv::Mat src, cv::Size size, int padColor=0);
    int                  num_bindings;  //engine所有输入和输出张量的数量

    int                  num_inputs  = 0; //输入张量的数量
    int                  num_outputs = 0; //输出张量的数量

    std::vector<Binding> input_bindings; //所有输入张量的信息(结构体Binding)存成列表
    std::vector<Binding> output_bindings; //所有输出张量的信息存成列表


    std::vector<void*>   UltraFace_host_ptrs;  //模型的的CPU指针
    std::vector<void*>   UltraFace_device_ptrs; //模型的GPU指针  对于每一个输入, 都需要给它分配一个GPU的内存空间,用于将CPU数据传给GPU
                                      // 对于每一个输出, 需要给它分别分配一个Cpu内存和GPU内存

    float * output_bboxes;
    float * output_scores;



private:
    nvinfer1::IRuntime*          runtime = nullptr;

    nvinfer1::ICudaEngine*       engine_UltraFace  = nullptr;
    nvinfer1::IExecutionContext* context_UltraFace = nullptr;

    cudaStream_t                 stream  = nullptr;
    Logger                       gLogger{nvinfer1::ILogger::Severity::kERROR};

    float * UltraFace_input_blob;

    int num_thread;
    int image_w;
    int image_h;

    int in_w;
    int in_h;
    int num_anchors;

    int topk;
    float score_threshold;
    float iou_threshold;

    const float mean_vals[3] = {127, 127, 127};
    const float norm_vals[3] = {1.0 / 128, 1.0 / 128, 1.0 / 128};

    const float center_variance = 0.1;
    const float size_variance = 0.2;
    const std::vector<std::vector<float>> min_boxes = {
            {10.0f,  16.0f,  24.0f},
            {32.0f,  48.0f},
            {64.0f,  96.0f},
            {128.0f, 192.0f, 256.0f}};
    const std::vector<float> strides = {8.0, 16.0, 32.0, 64.0};
    std::vector<std::vector<float>> featuremap_size;
    std::vector<std::vector<float>> shrinkage_size;
    std::vector<int> w_h_list;

    std::vector<std::vector<float>> priors = {};

    int pad_x;
    int pad_y; //x y方向上填充的像素宽度 单边
    float ratio; //缩放系数
};

UltraFace::UltraFace(const std::string& UltraFace_engine_file_path,int input_width, int input_length,
                     float score_threshold_, float iou_threshold_, int topk_)
{
    topk = topk_;
    score_threshold = score_threshold_;
    iou_threshold = iou_threshold_;
    in_w = input_width;
    in_h = input_length;
    w_h_list = {in_w, in_h};
    std::cout<<in_w<<std::endl;

    for (auto size : w_h_list) {
        std::vector<float> fm_item;
        for (float stride : strides) {
            fm_item.push_back(ceil(size / stride));
        }
        featuremap_size.push_back(fm_item);
    }

    for (auto size : w_h_list) {
        shrinkage_size.push_back(strides);
    }

    /* generate prior anchors */
    for (int index = 0; index < num_featuremap; index++) {
        float scale_w = in_w / shrinkage_size[0][index];
        float scale_h = in_h / shrinkage_size[1][index];
        for (int j = 0; j < featuremap_size[1][index]; j++) {
            for (int i = 0; i < featuremap_size[0][index]; i++) {
                float x_center = (i + 0.5) / scale_w;
                float y_center = (j + 0.5) / scale_h;

                for (float k : min_boxes[index]) {
                    float w = k / in_w;
                    float h = k / in_h;
                    priors.push_back({clip(x_center, 1), clip(y_center, 1), clip(w, 1), clip(h, 1)});
                }
            }
        }
    }
    num_anchors = priors.size();
    std::cout<<"num_anchors"<<num_anchors<<std::endl;
    //UltraFace的输入张量为1x3x240x320
    this->UltraFace_input_blob = new float[3*240*320];

    ///打开trt文件
    //打开engine二进制文件，这是通过std::ifstream类完成的，文件以二进制模式打开。
    std::ifstream UltraFace_engine_file(UltraFace_engine_file_path, std::ios::binary);
    //检查是否成功打开
    assert(UltraFace_engine_file.good());

    //将文件指针移动到文件的末尾，以便获取文件的大小。
    UltraFace_engine_file.seekg(0, std::ios::end);
    //获取了文件大小 即size
    auto size_UltraFace_engine_file = UltraFace_engine_file.tellg();
    //将文件指针移动回文件的开头。
    UltraFace_engine_file.seekg(0, std::ios::beg);
    //分配一个名为trtModelStream的数组，其大小为前面获取的文件大小（size变量的值）。
    char* trtModelStream_UltraFace = new char[size_UltraFace_engine_file];

    //判断是否分配成功
    assert(trtModelStream_UltraFace);

    //将文件读入trtModelStream
    UltraFace_engine_file.read(trtModelStream_UltraFace,  size_UltraFace_engine_file);

    UltraFace_engine_file.close();
    ///打开trt文件

    //初始化推理插件库
    initLibNvInferPlugins(&this->gLogger, "");

    ///步骤一:创建推理运行时(InferRuntime)对象 该对象作用如下
    //1.该函数会创建一个 TensorRT InferRuntime对象，这个对象是 TensorRT 库的核心组成部分之一。
    // InferRuntime对象是用于在推理阶段执行深度学习模型的实例。它提供了一种有效的方式来执行模型的前向传播操作。
    //2. TensorRT InferRuntime对象还负责管理 GPU 资源，包括分配和释放 GPU 内存。这对于加速推理操作非常重要，因为它可以确保有效地利用 GPU 资源，同时减少 GPU 内存泄漏的风险。
    //3. 一旦创建了 TensorRT 推理运行时对象，你可以使用它来构建、配置和执行 TensorRT 模型引擎（nvinfer1::ICudaEngine）。
    // Engine模型引擎是一个已经优化过的深度学习模型，可以高效地在 GPU 上执行推理操作。
    this->runtime = nvinfer1::createInferRuntime(this->gLogger);
    assert(this->runtime != nullptr);

    ///步骤二: 通过构建好InferRuntime对象 反序列化加载引擎给到(nvinfer1::ICudaEngine对象)
    this->engine_UltraFace = this->runtime->deserializeCudaEngine(trtModelStream_UltraFace, size_UltraFace_engine_file);
    assert(this->engine_UltraFace != nullptr);

    delete[] trtModelStream_UltraFace;

    ///步骤三:使用engine创建一个执行上下文对象(nvinfer1::IExecutionContext*)
    //执行上下文对象作用如下(nvinfer1::IExecutionContext):
    //1. 模型推理执行：执行上下文是用于执行深度学习模型推理操作的实例。
    // 一旦你创建了执行上下文，你可以使用它来加载输入数据、运行模型的前向传播，然后获取输出结果。
    // 这允许你将模型应用于实际数据，以获得推理结果。
    //2. GPU 资源管理：执行上下文还负责管理 GPU 资源，包括内存分配和释放。
    // 这确保了在执行推理操作时有效地利用 GPU 资源，同时降低了 GPU 内存泄漏的风险。
    this->context_UltraFace = this->engine_UltraFace->createExecutionContext();
    assert(this->context_UltraFace != nullptr);

    //cudaStreamCreate() 是 NVIDIA CUDA 库中的函数，用于创建一个 CUDA 流（CUDA Stream）。CUDA 流是一种并行执行 CUDA 操作的方式，它允许将多个 CUDA 操作异步执行，
    // 从而提高GPU的利用率。每个CUDA流代表一个独立的任务队列，CUDA操作可以按照流的顺序在多个流之间并行执行。
    cudaStreamCreate(&this->stream);

    //engine->getNbBindings() 是用于获取 TensorRT 模型引擎（nvinfer1::ICudaEngine）绑定的输入和输出张量数量的函数。
    // 在使用TensorRT进行深度学习模型推理时，你需要知道模型引擎绑定的输入和输出张量的数量，以便为它们分配内存并正确配置推理上下文。
    //具体来说，engine->getNbBindings() 函数返回一个整数值，表示与该模型引擎相关的绑定张量的总数。这个值通常是输入张量的数量加上输出张量的数量
    this->num_bindings = this->engine_UltraFace->getNbBindings();

    ///为初始化模型赋予初值
    for (int i = 0; i < this->num_bindings; ++i) {
        //该结构体用于保存第i个输入或输出绑定的信息
        Binding            binding;

        //一个结构体,用于表示张量(输入输出和中间层数据)
        // dims.nbDims表示维度的数量1X3X255X255 时维度为4
        // dims.d 一个整数数组，包含每个维度的大小
        //一个形状为 (batch_size, channels, height, width) (1X3X255X255)
        // 的四维图像张量可以表示为 nvinfer1::Dims 对象，其中 nbDims 为 4，d[0] 表示批量大小，d[1] 表示通道数，d[2] 表示高度，d[3] 表示宽度。
        nvinfer1::Dims     dims;

        // 在使用 TensorRT 进行深度学习模型推理时，每个模型引擎都有输入绑定和输出绑定。这些绑定指定了模型的输入和输出张量的属性，包括数据类型、维度等。
        //这些信息都是可以被获取的
        //这里是获取第i个绑定的数据类型. i=0是为输入绑定,i=1时为输出绑定
        nvinfer1::DataType dtype = this->engine_UltraFace->getBindingDataType(i);
        //保存第i个绑定的张量的数据类型所对应的字节大小
        binding.dsize            = type_to_size(dtype);

        //获取第i个绑定的名称 i = 0 ,输入数据的名称  i = 1,输出数据的名称
        std::string        name  = this->engine_UltraFace->getBindingName(i);
        binding.name             = name;
        std::cout<<"第"<<i<<"个绑定的名称为:"<<name<<std::endl;

        //这个函数可以判断第i个绑定是输入绑定还是输出绑定
        bool IsInput = engine_UltraFace->bindingIsInput(i);
        if (IsInput) {
            //如果是输入绑定,将输入绑定的数量加1
            this->num_inputs += 1;
            //获取一个输入的图像张量,用dims进行保存
            dims         = this->engine_UltraFace->getProfileDimensions(i, 0, nvinfer1::OptProfileSelector::kMAX);
            //binding.size = get_size_by_dims(dims)计算该图像张量的所有元素的数量(1X3X255X255)
            //binding.dsize保存的元素的数据类型所需要的字节数,最终即可计算所需的总内存大小
            binding.size = get_size_by_dims(dims);
            //将该图像张量也保存进binding里面
            binding.dims = dims;
            this->input_bindings.push_back(binding);
            // set max opt shape
            // context对象用于执行推理,这里获取到输入图像张量之后,使用该函数设置推理时的输入张量维度
            this->context_UltraFace->setBindingDimensions(i, dims);
        }
        else {
            //获取输出的张量维度信息
            dims         = this->context_UltraFace->getBindingDimensions(i);
            binding.size = get_size_by_dims(dims);
            binding.dims = dims;
            this->output_bindings.push_back(binding);
            this->num_outputs += 1;
        }
    }

    this->make_pipe();
}


UltraFace::~UltraFace()
{
    this->context_UltraFace->destroy();
    this->engine_UltraFace->destroy();

    this->runtime->destroy();
    delete[] UltraFace_input_blob;

//    CHECK(cudaFree(init_output));

    cudaStreamDestroy(this->stream);

    for (auto& ptr : this->UltraFace_device_ptrs) {
        CHECK(cudaFree(ptr));
    }

    for (auto& ptr : this->UltraFace_host_ptrs) {
        CHECK(cudaFreeHost(ptr));
    }
}


void UltraFace::infer(std::vector<void*> device_ptrs,std::vector<void*> host_ptrs,int num_inputs,int num_outputs,std::vector<Binding> output_bindings)
{
    // this->context->enqueueV2：使用TensorRT执行上下文 this->context 对象执行推理操作。
    // 这是实际的推理步骤，它将输入数据传递给模型并执行模型的前向传播。
    // 具体来说，enqueueV2 接受以下参数：
    // this->device_ptrs.data()：包含了输入和输出数据的GPU内存指针数组。这些指针指向了经过分配的GPU内存中的输入数据和输出数据。
    // this->stream：CUDA流，用于异步执行推理操作。
    // nullptr：这里为了简化没有传递其他回调函数。
    this->context_UltraFace->enqueueV2(device_ptrs.data(), this->stream, nullptr);


    ///循环处理输出数据
    for (int i = 0; i < num_outputs; i++) {
        //对于每一个输出绑定,计算输出张量的大小,该大小为张量所有元素个数乘以每个元素所占用的字节数
        size_t osize = output_bindings[i].size * output_bindings[i].dsize;

        //使用 cudaMemcpyAsync 将模型的输出数据从GPU内存（this->device_ptrs[i + this->num_inputs]，其中 i 是当前输出绑定的索引）
        // 异步复制到CPU内存（this->host_ptrs[i]，其中 i 是当前输出绑定的索引）。
        //这个步骤用于将模型的输出数据从GPU内存传输到CPU内存，以便进一步处理和分析。
        CHECK(cudaMemcpyAsync(
                host_ptrs[i], device_ptrs[i + num_inputs], osize, cudaMemcpyDeviceToHost, this->stream));
    }
    //等待CUDA流中的所有操作完成,以确保在使用模型的输出数据之前，所有数据都已正确地从GPU传输到CPU。
    cudaStreamSynchronize(this->stream);
}

void UltraFace::make_pipe()
{
    ///为模型的每个绑定分配内存和指针
    ///device_ptrs中包含了输入和输出的GPU内存指针

    //对于输入绑定，使用 cudaMallocAsync 分配GPU内存，以便存储输入数据。
    // 它会为每个输入绑定分配一个相应大小的GPU内存块，并将指针添加到 this->device_ptrs 向量中。
    for (auto& bindings : this->input_bindings) {
        void* d_ptr;
        CHECK(cudaMallocAsync(&d_ptr, bindings.size * bindings.dsize, this->stream));
        this->UltraFace_device_ptrs.push_back(d_ptr);
    }

    //对于输出绑定，函数分配两个内存块：一个在GPU上分配，一个在CPU上分配。
    // cudaMallocAsync 用于分配GPU内存，
    // cudaHostAlloc 用于在CPU上分配内存。
    // 然后，将这两个内存块的指针添加到 this->device_ptrs 和 this->host_ptrs 向量中。
    for (auto& bindings : this->output_bindings) {
        void * d_ptr, *h_ptr;
        size_t size = bindings.size * bindings.dsize;
        CHECK(cudaMallocAsync(&d_ptr, size, this->stream));
        CHECK(cudaHostAlloc(&h_ptr, size, 0));
        this->UltraFace_device_ptrs.push_back(d_ptr);
        this->UltraFace_host_ptrs.push_back(h_ptr);
    }
}


void UltraFace::detect(cv::Mat im, std::vector<FaceInfo> &face_list)
{
    ///前处理
    this->image_w = im.cols;
    this->image_h = im.rows;
    //将图像进行按照模型输入大小320X240(wxh)进行缩放
    cv::Mat resizedImage;

    //cv::resize(im, resizedImage, cv::Size(in_w, in_h), 0, 0, cv::INTER_LINEAR);
    //将resize改成resizeandpading 因为有时候图像不一定和模型的输入成比例
    resizedImage = resizeAndPadingImage(im,cv::Size(in_w,in_h));

    cv::imshow("ss",resizedImage);
    cv::waitKey(5);
    //执行均值减法和归一化操作
    // 遍历图像的每个像素
    //std::cout<<resizedImage.rows<<std::endl;
    /**
    //行 240
    for (int y = 0; y < resizedImage.rows; y++) {
        //列 320
        for (int x = 0; x < resizedImage.cols; x++) {
            // 遍历图像的每个通道
            for (int c = 0; c < resizedImage.channels(); c++) {
                // 获取像素值
                float pixelValue = resizedImage.at<cv::Vec3b>(y, x)[c];
                //减均值并归一
                float pix_value_deal = (pixelValue - mean_vals[0])*norm_vals[0];
               // std::cout<<pix_value_deal<<std::endl;
                //赋予给UltraFace_input_blob
                UltraFace_input_blob[c*resizedImage.cols*resizedImage.rows + x + y*resizedImage.cols] = pix_value_deal;
            }
        }
    }
    **/

    //改进了循环遍历,减少了计算开销
    const int imageCols = resizedImage.cols;
    const int imageRows = resizedImage.rows;
    const int imageChannels = resizedImage.channels();
    const int imageSize = imageCols * imageRows;

    float* ultraFaceInputPtr = UltraFace_input_blob;

    for (int y = 0; y < imageRows; y++) {
        for (int x = 0; x < imageCols; x++) {
            for (int c = 0; c < imageChannels; c++) {
            	const int pixelIdx = c * imageSize + x + y * imageCols;
            	const float pixelValue = resizedImage.at<cv::Vec3b>(y, x)[c];
            	const float pixValueDeal = (pixelValue - mean_vals[c]) * norm_vals[c];
            	ultraFaceInputPtr[pixelIdx] = pixValueDeal;
        	}
    	}
    }

    ///将图像数据传入engine中
    //第一步将数据复制到给模型分配的GPU内存中,device_ptrs[0]是输入数据的GPU内存指针,device_ptrs[1] [2]是输出数据的GPU内存指针
    cudaMemcpy(UltraFace_device_ptrs[0], UltraFace_input_blob, input_bindings[0].size * input_bindings[0].dsize, cudaMemcpyHostToDevice);
    //第二步进行前向推理
    this->infer(UltraFace_device_ptrs,UltraFace_host_ptrs,num_inputs,num_outputs,output_bindings);
    //推理之后,数据已经被复制到cpu指针指向的内存中
    this->output_scores = static_cast<float*>(this->UltraFace_host_ptrs[0]);
    this->output_bboxes = static_cast<float*>(this->UltraFace_host_ptrs[1]);

    ///后处理
    std::vector<FaceInfo> bbox_collection;
    std::vector<FaceInfo> valid_input;

    generateBBox(bbox_collection, this->output_scores, this->output_bboxes, score_threshold, num_anchors);
    nms(bbox_collection,face_list);
}

void UltraFace::generateBBox(std::vector<FaceInfo> &bbox_collection, float * scores, float *boxes, float score_threshold, int num_anchors) {
    //score 1x4420x2  num_anchors = 4420 ,相当于有4420个2维数组,每个数组的第二个值对应一个分数.最終融合在了一个数组中
    //box   1x4420x4  相当于有4420个4维数组,每个4维数组对应一个坐标.
    for (int i = 0; i < num_anchors; i++) {
        if (scores[i * 2 + 1] > score_threshold) {
            FaceInfo rects;

            float x_center = ((boxes[i * 4] * center_variance * priors[i][2] + priors[i][0])*in_w - this->pad_x)*this->ratio;
            float y_center = ((boxes[i * 4 + 1] * center_variance * priors[i][3] + priors[i][1])*in_h-this->pad_y)*this->ratio;
            float w = (exp(boxes[i * 4 + 2] * size_variance) * priors[i][2])*in_w* this->ratio;
            float h = (exp(boxes[i * 4 + 3] * size_variance) * priors[i][3])*in_h*this->ratio;  //模型的输出是归一化之后的结果 所以要乘in_w in_h 320 240

            float x1 = x_center - w / 2.0;
            rects.x1 = clip(x1,image_w);

            float y1 = y_center - h / 2.0;
            rects.y1 = clip(y1,image_h);

            float x2 = x_center + w / 2.0;
            rects.x2 = clip(x2,image_w);

            float y2 = y_center + h / 2.0;
            rects.y2 = clip(y2,image_h);

//            rects.y1 = clip(y_center - h / 2.0,  1);
//            rects.x2 = clip(x_center + w / 2.0, 1);
//            rects.y2 = clip(y_center + h / 2.0, 1);

            rects.score = clip(scores[i * 2 + 1], 1);
            bbox_collection.push_back(rects);
        }
    }
}

///非极大值抑制
void UltraFace::nms(std::vector<FaceInfo> &input, std::vector<FaceInfo> &output, int type) {
    std::sort(input.begin(), input.end(), [](const FaceInfo &a, const FaceInfo &b) { return a.score > b.score; });

    int box_num = input.size();

    std::vector<int> merged(box_num, 0);

    for (int i = 0; i < box_num; i++) {
        if (merged[i])
            continue;
        std::vector<FaceInfo> buf;

        buf.push_back(input[i]);
        merged[i] = 1;

        float h0 = input[i].y2 - input[i].y1 + 1;
        float w0 = input[i].x2 - input[i].x1 + 1;

        float area0 = h0 * w0;

        for (int j = i + 1; j < box_num; j++) {
            if (merged[j])
                continue;

            float inner_x0 = input[i].x1 > input[j].x1 ? input[i].x1 : input[j].x1;
            float inner_y0 = input[i].y1 > input[j].y1 ? input[i].y1 : input[j].y1;

            float inner_x1 = input[i].x2 < input[j].x2 ? input[i].x2 : input[j].x2;
            float inner_y1 = input[i].y2 < input[j].y2 ? input[i].y2 : input[j].y2;

            float inner_h = inner_y1 - inner_y0 + 1;
            float inner_w = inner_x1 - inner_x0 + 1;

            if (inner_h <= 0 || inner_w <= 0)
                continue;

            float inner_area = inner_h * inner_w;

            float h1 = input[j].y2 - input[j].y1 + 1;
            float w1 = input[j].x2 - input[j].x1 + 1;

            float area1 = h1 * w1;

            float score;

            score = inner_area / (area0 + area1 - inner_area);

            if (score > iou_threshold) {
                merged[j] = 1;
                buf.push_back(input[j]);
            }
        }
        if (type == hard_nms) {
                output.push_back(buf[0]);
        }
        else if(type == blending_nms){
            float total = 0;
            for (int i = 0; i < buf.size(); i++) {
                total += exp(buf[i].score);
            }
            FaceInfo rects;
            memset(&rects, 0, sizeof(rects));
            for (int i = 0; i < buf.size(); i++) {
                float rate = exp(buf[i].score) / total;
                rects.x1 += buf[i].x1 * rate;
                rects.y1 += buf[i].y1 * rate;
                rects.x2 += buf[i].x2 * rate;
                rects.y2 += buf[i].y2 * rate;
                rects.score += buf[i].score * rate;
            }
            output.push_back(rects);
        }
    }
}

//输入前处理
cv::Mat UltraFace::resizeAndPadingImage(cv::Mat src, cv::Size size, int padColor) {
    cv::Mat img = src.clone();
    int h = img.rows;
    int w = img.cols;
    int sh = size.height;
    int sw = size.width;

    int interp = (h > sh || w > sw) ? cv::INTER_AREA : cv::INTER_CUBIC;

    double aspect = static_cast<double>(w) / h;
    int new_w, new_h, pad_left, pad_right, pad_top, pad_bot;

    if (aspect > 1) {
        new_w = sw;
        new_h = cvRound(new_w / aspect);
        double pad_vert = (sh - new_h) / 2;
        pad_top = static_cast<int>(std::floor(pad_vert));
        pad_bot = static_cast<int>(std::ceil(pad_vert));
        pad_left = 0;
        pad_right = 0;

        this->pad_x = 0;
        this->pad_y = int(pad_vert);
        this->ratio = float(w)/new_w;


    } else if (aspect < 1) {
        new_h = sh;
        new_w = cvRound(new_h * aspect);
        double pad_horz = (sw - new_w) / 2;
        pad_left = static_cast<int>(std::floor(pad_horz));
        pad_right = static_cast<int>(std::ceil(pad_horz));
        pad_top = 0;
        pad_bot = 0;
        this->pad_x = pad_horz;
        this->pad_y = 0;
        this->ratio = float(h)/new_h;
    } else {
        new_h = sh;
        new_w = sw;
        pad_left = 0;
        pad_right = 0;
        pad_top = 0;
        pad_bot = 0;

        this->pad_x = 0;
        this->pad_y = 0;
        this->ratio = float(h)/new_h;
    }

    cv::Scalar padColorScalar(padColor);
    cv::Mat scaledImg;
    cv::resize(img, scaledImg, cv::Size(new_w, new_h), 0, 0, interp);
    cv::copyMakeBorder(scaledImg, scaledImg, pad_top, pad_bot, pad_left, pad_right, cv::BORDER_CONSTANT, padColorScalar);

    return scaledImg;
}

#endif ULTRAFACE_H
