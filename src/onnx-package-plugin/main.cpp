/*
1. 对插件进行了封装，使得用起来更简单
2. 在onnx-tensorrt中添加了onnxplugin.cpp，实现对IPluginV2DynamicExt的封装
3. 在onnx-tensorrt/builtin_op_importers.cpp:5095行，添加了Plugin的解析支持
    - DEFINE_BUILTIN_OP_IMPORTER(Plugin)
    - 使得只要名字是Plugin的节点，都可以解释到该函数上
    - 在代码中，为通用插件提供了支持，使得使用者只需要继承简单的插件接口即可完成需求
4. 在gen-onnx.py导出时，symbolic函数返回时，g.op返回的永远都是Plugin这个名字，然后name_s指定为自己注册的插件名称，info_s则传递为json字符串，那么复合属性就可以轻易得到支持


# 封装后的插件实现
1. 导出onnx时，按照gen-onnx.py，在symbolic函数返回时，指定g.op的name为Plugin
2. 指定g.op中name_s属性为注册的插件名称，对应后续插件类的类名
3. 指定g.op中info_属性为需要读取的复合属性，字符串。通常可以传递json，使得属性再复杂都可以，避免使用官方的方式
4. 创建easy-plugin.cu文件，定义自己的类并继承自ONNXPlugin::TRTPlugin
5. 实现需要的函数
    - config_finish[非必要]：配置完成函数
        - 当插件配置完毕时调用，可以在其中拿到各种属性，例如info、weights等
    - new_config[非必要]：实例化一个配置对象
        - 可以自定义LayerConfig类并返回，也可以直接使用LayerConfig类
        - 这个函数最大的作用，是配置本插件支持的数据格式和类型。比如fp32和fp16的支持等
    - getOutputDimensions[非必要]，获取该插件输出的shape大小，默认取第一个输入的大小
        - 对应于原始插件的getOutputDimensions函数
    - enqueue[必要]，插件推理过程
        - 插件的实际推理过程，该函数可能在编译和推理阶段数次调用
6. 注册插件，使用RegisterPlugin宏
    - RegisterPlugin(MYSELU);
    - 格式是RegisterPlugin(类名);
7. 好了，可以使用插件了

*/
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

using namespace std;

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

bool build_model(){
    TRTLogger logger;

    // 这是基本需要的组件
    nvinfer1::IBuilder* builder = nvinfer1::createInferBuilder(logger);
    nvinfer1::IBuilderConfig* config = builder->createBuilderConfig();
    nvinfer1::INetworkDefinition* network = builder->createNetworkV2(1);

    // 通过onnxparser解析器解析的结果会填充到network中，类似addConv的方式添加进去
    nvonnxparser::IParser* parser = nvonnxparser::createParser(*network, logger);
    if(!parser->parseFromFile("demo.onnx", 1)){
        printf("Failed to parse demo.onnx\n");

        // 注意这里的几个指针还没有释放，是有内存泄漏的，后面考虑更优雅的解决
        return false;
    }
    
    int maxBatchSize = 10;
    printf("Workspace Size = %.2f MB\n", (1 << 28) / 1024.0f / 1024.0f);
    config->setMaxWorkspaceSize(1 << 28);

    // 如果模型有多个输入，则必须多个profile
    auto profile = builder->createOptimizationProfile();
    auto input_tensor = network->getInput(0);
    int input_channel = input_tensor->getDimensions().d[1];
    
    // 配置输入的最小、最优、最大的范围
    profile->setDimensions(input_tensor->getName(), nvinfer1::OptProfileSelector::kMIN, nvinfer1::Dims4(1, input_channel, 3, 3));
    profile->setDimensions(input_tensor->getName(), nvinfer1::OptProfileSelector::kOPT, nvinfer1::Dims4(1, input_channel, 3, 3));
    profile->setDimensions(input_tensor->getName(), nvinfer1::OptProfileSelector::kMAX, nvinfer1::Dims4(maxBatchSize, input_channel, 5, 5));
    config->addOptimizationProfile(profile);

    nvinfer1::ICudaEngine* engine = builder->buildEngineWithConfig(*network, *config);
    if(engine == nullptr){
        printf("Build engine failed.\n");
        return false;
    }

    // 将模型序列化，并储存为文件
    nvinfer1::IHostMemory* model_data = engine->serialize();
    FILE* f = fopen("engine.trtmodel", "wb");
    fwrite(model_data->data(), 1, model_data->size(), f);
    fclose(f);

    // 卸载顺序按照构建顺序倒序
    model_data->destroy();
    parser->destroy();
    engine->destroy();
    network->destroy();
    config->destroy();
    builder->destroy();
    printf("Done.\n");
    return true;
}

///////////////////////////////////////////////////////////////////////////////////////////////////////

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

void inference(){

    TRTLogger logger;
    auto engine_data = load_file("engine.trtmodel");
    nvinfer1::IRuntime* runtime   = nvinfer1::createInferRuntime(logger);
    nvinfer1::ICudaEngine* engine = runtime->deserializeCudaEngine(engine_data.data(), engine_data.size());
    if(engine == nullptr){
        printf("Deserialize cuda engine failed.\n");
        runtime->destroy();
        return;
    }

    nvinfer1::IExecutionContext* execution_context = engine->createExecutionContext();
    cudaStream_t stream = nullptr;
    cudaStreamCreate(&stream);

    float input_data_host[] = {
        // batch 0
        1,   1,   1,
        1,   1,   1,
        1,   1,   1,

        // batch 1
        -1,   1,   1,
        1,   0,   1,
        1,   1,   -1
    };
    float* input_data_device = nullptr;

    // 3x3输入，对应3x3输出
    int ib = 2;
    int iw = 3;
    int ih = 3;
    float output_data_host[ib * iw * ih];
    float* output_data_device = nullptr;
    cudaMalloc(&input_data_device, sizeof(input_data_host));
    cudaMalloc(&output_data_device, sizeof(output_data_host));
    cudaMemcpyAsync(input_data_device, input_data_host, sizeof(input_data_host), cudaMemcpyHostToDevice, stream);

    // 明确当前推理时，使用的数据输入大小
    execution_context->setBindingDimensions(0, nvinfer1::Dims4(ib, 1, ih, iw));
    float* bindings[] = {input_data_device, output_data_device};
    bool success      = execution_context->enqueueV2((void**)bindings, stream, nullptr);
    cudaMemcpyAsync(output_data_host, output_data_device, sizeof(output_data_host), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    for(int b = 0; b < ib; ++b){
        printf("batch %d. output_data_host = \n", b);
        for(int i = 0; i < iw * ih; ++i){
            printf("%f, ", output_data_host[b * iw * ih + i]);
            if((i + 1) % iw == 0)
                printf("\n");
        }
    }

    printf("Clean memory\n");
    cudaStreamDestroy(stream);
    cudaFree(input_data_device);
    cudaFree(output_data_device);
    execution_context->destroy();
    engine->destroy();
    runtime->destroy();
}

int main(){
    if(!build_model()){
        return -1;
    }
    inference();
    return 0;
}