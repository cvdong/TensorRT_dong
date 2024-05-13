// 插件主要是继承自IPluginV2DynamicExt后实现特定接口即可
/*

插件的实现
1. 导出onnx的时候，为module增加symbolic函数

   - 参照这里：https://pytorch.org/docs/1.10/onnx.html#torch-autograd-functions
   - g.op对应的名称，需要与下面解析器的名称对应
2. src/onnx-tensorrt-release-8.0/builtin_op_importers.cpp:5094行，添加对插件op的解析

   - DEFINE_BUILTIN_OP_IMPORTER(MYSELU)
   - 注意解析时采用的名称要匹配上src/myselu-plugin.cpp:15行
3. src/myselu-plugin.cpp:183行，创建MySELUPluginCreator，插件创建器

   - 实际注册时，注册的是创建器，交给tensorRT管理
   - REGISTER_TENSORRT_PLUGIN(MySELUPluginCreator);
   - src/myselu-plugin.cpp:23行
4. src/myselu-plugin.cpp:42行，定义插件类MySELUPlugin

   - Creator创建器来实例化MySELUPlugin类
5. 正常使用该onnx即可

插件的编译
   - 1. 通过MySELUPluginCreator::createPlugin创建plugin

   - 2. 期间会调用MySELUPlugin::clone克隆插件

   - 3. 调用MySELUPlugin::supportsFormatCombination判断该插件所支持的数据格式和类型

   - 在这里我们告诉引擎，本插件可以支持什么类型的推理
   - 可以支持多种，例如fp32、fp16、int8等等

   - 4. 调用MySELUPlugin::getOutputDimensions获取该层的输出维度是多少

   - 5. 调用MySELUPlugin::enqueue进行性能测试（不是一定会执行）

   - 如果支持多种，则会在多种里面进行实际测试，选择一个性能最好的配置

   - 6. 调用MySELUPlugin::configurePlugin配置插件格式

   - 告诉你目前这个层所采用的数据格式和类型

   - 7. 调用MySELUPlugin::serialize将该层的参数序列化储存为trtmodel文件

插件的推理

   - 1. 通过MySELUPluginCreator::deserializePlugin反序列化插件参数进行创建

   - 2. 期间会调用MySELUPlugin::clone克隆插件

   - 3. 调用MySELUPlugin::configurePlugin配置当前插件使用的数据类型和格式

   - 4. 调用MySELUPlugin::enqueue进行推理

*/

// make run

// @dong 整理归档 2024.5

// tensorRT include
// 编译用的头文件
#include <NvInfer.h>

// onnx解析器的头文件
#include <onnx-tensorrt-release-8.0/NvOnnxParser.h>

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
    FILE* f = fopen("test.engine", "wb");
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
    auto engine_data = load_file("test.engine");
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