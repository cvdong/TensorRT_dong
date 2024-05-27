// resnet 18 推理代码

// @dong 整理归档 2024.5

// tensorRT include
// 编译用的头文件
#include <NvInfer.h>

// onnx解析器的头文件
#include <onnx-tensorrt/NvOnnxParser.h>

// 推理用的运行时头文件
#include <NvInferRuntime.h>

// cuda include
#include <cuda_runtime.h>

// system include
#include <stdio.h>
#include <math.h>

#include <iostream>
#include <fstream>
#include <vector>
#include <memory>
#include <functional>
#include <unistd.h>

#include <opencv2/opencv.hpp>

using namespace std;

#define checkRuntime(op)  __check_cuda_runtime((op), #op, __FILE__, __LINE__)


// 检查运行时错误
bool __check_cuda_runtime(cudaError_t code, const char* op, const char* file, int line){
    if(code != cudaSuccess){    
        const char* err_name = cudaGetErrorName(code);    
        const char* err_message = cudaGetErrorString(code);  
        printf("runtime error %s:%d  %s failed. \n  code = %s, message = %s\n", file, line, op, err_name, err_message);   
        return false;
    }
    return true;
}

// 打印日志
inline const char* severity_string(nvinfer1::ILogger::Severity t){
    switch(t){
        case nvinfer1::ILogger::Severity::kINTERNAL_ERROR: return "internal_error";
        case nvinfer1::ILogger::Severity::kERROR:   return "error";
        case nvinfer1::ILogger::Severity::kWARNING: return "warning";
        case nvinfer1::ILogger::Severity::kINFO:    return "info";
        case nvinfer1::ILogger::Severity::kVERBOSE: return "verbose";
        default: return "unknow";
    }
}

// 打印日志
class TRTLogger : public nvinfer1::ILogger{
public:
    virtual void log(Severity severity, nvinfer1::AsciiChar const* msg) noexcept override{
        if(severity <= Severity::kINFO){
            // 打印带颜色的字符，格式如下：
            // printf("\033[47;33m打印的文本\033[0m");
            // 其中 \033[ 是起始标记
            //      47    是背景颜色
            //      ;     分隔符
            //      33    文字颜色
            //      m     开始标记结束
            //      \033[0m 是终止标记
            // 其中背景颜色或者文字颜色可不写
            // 部分颜色代码 https://blog.csdn.net/ericbar/article/details/79652086
            if(severity == Severity::kWARNING){
                printf("\033[33m%s: %s\033[0m\n", severity_string(severity), msg);
            }
            else if(severity <= Severity::kERROR){
                printf("\033[31m%s: %s\033[0m\n", severity_string(severity), msg);
            }
            else{
                printf("%s: %s\n", severity_string(severity), msg);
            }
        }
    }
} logger;


// 通过智能指针管理nv返回的指针参数
// 内存自动释放，避免泄漏
template<typename _T>
shared_ptr<_T> make_nvshared(_T* ptr){
    return shared_ptr<_T>(ptr, [](_T* p){p->destroy();});
}

/*
template<typename _T>
shared_ptr<_T> make_nvshared(_T* ptr){
    return shared_ptr<_T>(ptr, [](_T* p){p->destroy();});
}

*/

// 判断文件是否存在
bool exists(const string& path){


#ifdef _WIN32
    return ::PathFileExistsA(path.c_str());
#else
    return access(path.c_str(), R_OK) == 0;
#endif
}

// general build code 
bool build_model(){

    if(exists("test.engine")){
        printf("test.engine has exists.\n");
        return true;
    }

    TRTLogger logger;

    // 基本组件
    auto builder = make_nvshared(nvinfer1::createInferBuilder(logger));
    auto config = make_nvshared(builder->createBuilderConfig());
    auto network = make_nvshared(builder->createNetworkV2(1));

    // 通过onnxparser解析器解析的结果会填充到network中
    auto parser = make_nvshared(nvonnxparser::createParser(*network, logger));

    if(!parser->parseFromFile("classifier.onnx", 1)){
        printf("Failed to parse classifier.onnx\n");

        return false;
    }
    

    int maxBatchSize = 10;
    printf("Workspace Size = %.2f MB\n", (1 << 28) / 1024.0f / 1024.0f);

    // 设置最大workspace大小
    config->setMaxWorkspaceSize(1 << 28);

    // 如果模型有多个输入，则必须多个profile
    auto profile = builder->createOptimizationProfile();
    auto input_tensor = network->getInput(0);
    auto input_dims = input_tensor->getDimensions();
    
    // 配置最小、最优、最大范围
    input_dims.d[0] = 1;
    profile->setDimensions(input_tensor->getName(), nvinfer1::OptProfileSelector::kMIN, input_dims);
    profile->setDimensions(input_tensor->getName(), nvinfer1::OptProfileSelector::kOPT, input_dims);

    input_dims.d[0] = maxBatchSize;

    profile->setDimensions(input_tensor->getName(), nvinfer1::OptProfileSelector::kMAX, input_dims);

    config->addOptimizationProfile(profile);

    // 构建engine
    // 构建engine时，会根据配置和网络结构生成engine
    auto engine = make_nvshared(builder->buildEngineWithConfig(*network, *config));

    if(engine == nullptr){
        printf("Build engine failed.\n");
        return false;
    }

    // 将模型序列化，并储存为文件
    auto model_data = make_nvshared(engine->serialize());
    FILE* f = fopen("test.engine", "wb");
    fwrite(model_data->data(), 1, model_data->size(), f);
    fclose(f);

    // 卸载顺序按照构建顺序倒序
    printf("Done.\n");
    return true;
}

///////////////////////////////////////////////////////////////////////////////////////////////////////

// 加载文件
vector<unsigned char> load_file(const string& file){
    ifstream in(file, ios::in | ios::binary);
    if (!in.is_open())
        return {};

    in.seekg(0, ios::end);
    size_t length = in.tellg();

    std::vector<uint8_t> data;

    if (length > 0){
        in.seekg(0, ios::beg);
        data.resize(length);

        in.read((char*)&data[0], length);
    }
    in.close();
    return data;
}


// 加载标签
vector<string> load_labels(const char* file){
    vector<string> lines;

    ifstream in(file, ios::in | ios::binary);
    if (!in.is_open()){
        printf("open %d failed.\n", file);
        return lines;
    }
    
    string line;
    while(getline(in, line)){
        lines.push_back(line);
    }
    in.close();
    return lines;
}


// infer  
void inference(){

    TRTLogger logger;

    // 加载engine
    auto engine_data = load_file("test.engine");

    auto runtime   = make_nvshared(nvinfer1::createInferRuntime(logger));

    // build 序列化 -> infer反序列化
    auto engine = make_nvshared(runtime->deserializeCudaEngine(engine_data.data(), engine_data.size()));

    if(engine == nullptr){
        printf("Deserialize cuda engine failed.\n");
        runtime->destroy();
        return;
    }

    cudaStream_t stream = nullptr;

    // 创建执行上下文
    // 执行上下文是执行网络的上下文，包括stream和execution context
    // 创建stream，并设置execution context
    // 创建execution context时，会根据配置和网络结构生成execution context
  
    checkRuntime(cudaStreamCreate(&stream));

    auto execution_context = make_nvshared(engine->createExecutionContext());


    // input
    int input_batch = 1;
    int input_channel = 3;
    int input_height = 224;
    int input_width = 224;


    int input_numel = input_batch * input_channel * input_height * input_width;
    float* input_data_host = nullptr;
    float* input_data_device = nullptr;

    /*
    cudaError_t cudaMallocHost(void **ptr, size_t size);

    ptr：指向分配内存的指针的地址
    size：要分配的内存大小（以字节为单位）

    cudaMallocHost 是 CUDA API 中用于分配可分页主机内存（也称为“页面锁定”内存或“固定”内存）的函数
    使用 cudaMallocHost 分配的内存可以提高主机与设备之间的数据传输性能，因为这类内存不会被操作系统分页到磁盘，因此可以实现更高效的 DMA（直接内存访问）传输
    */

    checkRuntime(cudaMallocHost(&input_data_host, input_numel * sizeof(float)));


    // cudaError_t cudaMalloc(void **devPtr, size_t size);
    
    checkRuntime(cudaMalloc(&input_data_device, input_numel * sizeof(float)));


    ///////////////////////////////////////////////////
    // image to float
    auto image = cv::imread("xxx.jpg");
    float mean[] = {0.406, 0.456, 0.485};
    float std[]  = {0.225, 0.224, 0.229};

    // 对应于pytorch的预处理部分
    cv::resize(image, image, cv::Size(input_width, input_height));
    int image_area = image.cols * image.rows;

    unsigned char* pimage = image.data;
    float* phost_b = input_data_host + image_area * 0;
    float* phost_g = input_data_host + image_area * 1;
    float* phost_r = input_data_host + image_area * 2;


    for(int i = 0; i < image_area; ++i, pimage += 3){
        // 注意这里的顺序rgb调换了
        *phost_r++ = (pimage[0] / 255.0f - mean[0]) / std[0];
        *phost_g++ = (pimage[1] / 255.0f - mean[1]) / std[1];
        *phost_b++ = (pimage[2] / 255.0f - mean[2]) / std[2];
    }

  /*
  cudaMemcpyAsync 是 CUDA API 中用于异步内存拷贝的函数。
  它允许在主机（CPU）和设备（GPU）之间以及设备内部进行非阻塞的数据传输。
  与 cudaMemcpy 不同，cudaMemcpyAsync 不会阻塞主机代码的执行，而是允许主机在数据传输的同时继续执行其他操作。

  cudaError_t cudaMemcpyAsync(void *dst, const void *src, size_t count, cudaMemcpyKind kind, cudaStream_t stream = 0);

  */
    // host->device
    checkRuntime(cudaMemcpyAsync(input_data_device, input_data_host, input_numel * sizeof(float), cudaMemcpyHostToDevice, stream));


    // 3x3输入，对应3x3输出
    const int num_classes = 1000;

    // output
    float output_data_host[num_classes];

    float* output_data_device = nullptr;
    checkRuntime(cudaMalloc(&output_data_device, sizeof(output_data_host)));

    // 明确当前推理时，使用的数据输入大小
    auto input_dims = execution_context->getBindingDimensions(0);

    input_dims.d[0] = input_batch;

    // 设置当前推理时，input大小
    execution_context->setBindingDimensions(0, input_dims);

    float* bindings[] = {input_data_device, output_data_device};


    // 执行推理
    bool success      = execution_context->enqueueV2((void**)bindings, stream, nullptr);
    // device->host
    checkRuntime(cudaMemcpyAsync(output_data_host, output_data_device, sizeof(output_data_host), cudaMemcpyDeviceToHost, stream));

    // cudaError_t cudaStreamSynchronize(cudaStream_t stream);

    checkRuntime(cudaStreamSynchronize(stream));

    float* prob = output_data_host;
    int predict_label = std::max_element(prob, prob + num_classes) - prob;  // 确定预测类别的下标

    auto labels = load_labels("labels.imagenet.txt");
    auto predict_name = labels[predict_label];

    float confidence  = prob[predict_label];    // 获得预测值的置信度
    printf("Predict: %s, confidence = %f, label = %d\n", predict_name.c_str(), confidence, predict_label);

    // 释放资源
    checkRuntime(cudaStreamDestroy(stream));
    checkRuntime(cudaFreeHost(input_data_host));
    checkRuntime(cudaFree(input_data_device));
    checkRuntime(cudaFree(output_data_device));
}



// main
int main(){
    if(!build_model()){
        return -1;
    }
    inference();
    return 0;
}