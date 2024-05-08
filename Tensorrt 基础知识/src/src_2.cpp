// 本代码主要实现一个最简单的神经网络,并完成模型搭建->生成->推理
// @dong 整理归档 2024.5


// tensorRT include
#include <NvInfer.h>
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



class TRTLogger : public nvinfer1::ILogger{
public:
    virtual void log(Severity severity, nvinfer1::AsciiChar const* msg) noexcept override{
        if(severity <= Severity::kINFO){
            printf("%d: %s\n", severity, msg);
        }
    }
} logger;


nvinfer1::Weights make_weights(float* ptr, int n){
    nvinfer1::Weights w;
    w.count = n;
    w.type = nvinfer1::DataType::kFLOAT;
    w.values = ptr;
    return w;
}


// 生成engine
bool build_model(){
    TRTLogger logger;

    // 基本需要的组件
    nvinfer1::IBuilder* builder = nvinfer1::createInferBuilder(logger);
    nvinfer1::IBuilderConfig* config = builder->createBuilderConfig();
    nvinfer1::INetworkDefinition* network = builder->createNetworkV2(1);


    // 构建一个模型
    /*
        Network definition:

        image
          |
        linear (fully connected)  input = 3, output = 2, bias = True     w=[[1.0, 2.0, 0.5], [0.1, 0.2, 0.5]], b=[0.3, 0.8]
          |
        sigmoid
          |
        prob
    */

    // model 搭建
    const int num_input = 3;
    const int num_output = 2;
    float layer1_weight_values[] = {1.0, 2.0, 0.5, 0.1, 0.2, 0.5};
    float layer1_bias_values[]   = {0.3, 0.8};

    nvinfer1::ITensor* input = network->addInput("image", nvinfer1::DataType::kFLOAT, nvinfer1::Dims4(1, num_input, 1, 1));
    nvinfer1::Weights layer1_weight = make_weights(layer1_weight_values, 6);
    nvinfer1::Weights layer1_bias   = make_weights(layer1_bias_values, 2);
    auto layer1 = network->addFullyConnected(*input, num_output, layer1_weight, layer1_bias);
    auto prob = network->addActivation(*layer1->getOutput(0), nvinfer1::ActivationType::kSIGMOID);
    
    // 将我们需要的prob标记为输出
    network->markOutput(*prob->getOutput(0));

    printf("Workspace Size = %.2f MB\n", (1 << 28) / 1024.0f / 1024.0f);
    config->setMaxWorkspaceSize(1 << 28);
    builder->setMaxBatchSize(1);

    nvinfer1::ICudaEngine* engine = builder->buildEngineWithConfig(*network, *config);
    if(engine == nullptr){
        printf("Build engine failed.\n");
        return false;
    }


    // 编译
    // 将模型序列化，并储存为文件

    // 先序列化存储，然后反序列化解析
    // serialize --> deserializeCudaEngine

    nvinfer1::IHostMemory* model_data = engine->serialize();
    FILE* f = fopen("test.engine", "wb");
    fwrite(model_data->data(), 1, model_data->size(), f);
    fclose(f);

    // 卸载顺序按照构建顺序倒序
    model_data->destroy();
    engine->destroy();
    network->destroy();
    config->destroy();
    builder->destroy();
    printf("Done.\n");
    return true;
}


// load
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


// infer
void inference(){

    // ------------------------------ 1. 准备模型并加载   ----------------------------

    TRTLogger logger;
    auto engine_data = load_file("test.engine");


    // logger->runtime->engine->context->stream->enqueueV2

    // 执行推理前，需要创建一个推理的runtime接口实例。与builer一样，runtime需要logger

    nvinfer1::IRuntime* runtime   = nvinfer1::createInferRuntime(logger);

    // 将模型从读取到engine_data中，则可以对其进行反序列化以获得engine
    nvinfer1::ICudaEngine* engine = runtime->deserializeCudaEngine(engine_data.data(), engine_data.size());

    if(engine == nullptr){
        printf("Deserialize cuda engine failed.\n");
        runtime->destroy();
        return;
    }

    nvinfer1::IExecutionContext* execution_context = engine->createExecutionContext();

    cudaStream_t stream = nullptr;
    // 创建CUDA流，以确定这个batch的推理是独立的
    // cudaError_t cudaStreamCreate(cudaStream_t* pStream);

    cudaStreamCreate(&stream);

    /*
        Network definition:

        image
          |
        linear (fully connected)  input = 3, output = 2, bias = True     w=[[1.0, 2.0, 0.5], [0.1, 0.2, 0.5]], b=[0.3, 0.8]
          |
        sigmoid
          |
        prob
    */

    // ------------------------------ 2. 准备好要推理的数据并搬运到GPU   ----------------------------
    float input_data_host[] = {1, 2, 3};
    float* input_data_device = nullptr;

    float output_data_host[2];
    float* output_data_device = nullptr;

    // 申请显存
    // cudaError_t cudaMalloc(void** devPtr, size_t size);

    cudaMalloc(&input_data_device, sizeof(input_data_host));
    cudaMalloc(&output_data_device, sizeof(output_data_host));

    // cpu->gpu
    // cudaMemcpyAsync 异步内存拷贝函数，它允许在 CUDA 流中执行内存拷贝操作而无需等待拷贝完成

    // cudaError_t cudaMemcpyAsync(void* dst, const void* src, size_t count, cudaMemcpyKind kind, cudaStream_t stream = 0);

    cudaMemcpyAsync(input_data_device, input_data_host, sizeof(input_data_host), cudaMemcpyHostToDevice, stream);

    // 用一个指针数组指定input和output在gpu中的指针
    float* bindings[] = {input_data_device, output_data_device};

    // ------------------------------ 3. 推理并将结果搬运回CPU   ----------------------------
    bool success      = execution_context->enqueueV2((void**)bindings, stream, nullptr);
    
    // gpu->cpu
    cudaMemcpyAsync(output_data_host, output_data_device, sizeof(output_data_host), cudaMemcpyDeviceToHost, stream);

    // 阻塞host, 同步cuda流
    // CUDA 流是一系列异步执行的操作组成的序列，可以用于实现并行计算
    // cudaStreamSynchronize 函数用于等待指定的 CUDA 流上的所有操作完成，并且阻塞当前线程直到操作完成

    cudaStreamSynchronize(stream);

    printf("output_data_host = %f, %f\n", output_data_host[0], output_data_host[1]);

    // ------------------------------ 4. 释放内存 ----------------------------
    printf("Clean memory\n");


    cudaStreamDestroy(stream);

    execution_context->destroy();
    engine->destroy();
    runtime->destroy();

    // ------------------------------ 5. 手动推理进行验证 ----------------------------
    const int num_input = 3;
    const int num_output = 2;
    float layer1_weight_values[] = {1.0, 2.0, 0.5, 0.1, 0.2, 0.5};
    float layer1_bias_values[]   = {0.3, 0.8};

    printf("手动验证计算结果：\n");
    for(int io = 0; io < num_output; ++io){
        float output_host = layer1_bias_values[io];
        for(int ii = 0; ii < num_input; ++ii){
            output_host += layer1_weight_values[io * num_input + ii] * input_data_host[ii];
        }

        // sigmoid
        float prob = 1 / (1 + exp(-output_host));
        printf("output_prob[%d] = %f\n", io, prob);
    }
}



int main(){

    if(!build_model()){
        return -1;
    }
    inference();
    return 0;
}

/*
output_data_host = 
0.998887, 0.942673

手动验证计算结果：
output_prob[0] = 0.998887
output_prob[1] = 0.942676
*/

/*
执行推理的步骤：
  1. 准备模型并加载

  2. 创建runtime：`createInferRuntime(logger)`

  3. 使用运行时时，以下步骤：
     1. 反序列化创建engine, 得为engine提供数据：`runtime->deserializeCudaEngine(modelData, modelSize)`,其中`modelData`包含的是input和output的名字，形状，大小和数据类型
        ```cpp
        class ModelData(object):
        INPUT_NAME = "data"
        INPUT_SHAPE = (1, 1, 28, 28) // [B, C, H, W]
        OUTPUT_NAME = "prob"
        OUTPUT_SIZE = 10
        DTYPE = trt.float32
        ```

     2. 从engine创建执行上下文:`engine->createExecutionContext()`

  4. 创建CUDA流`cudaStreamCreate(&stream)`：
     1. CUDA编程流是组织异步工作的一种方式，创建流来确定batch推理的独立
     2. 为每个独立batch使用IExecutionContext(3.2中已经创建了)，并为每个独立批次使用cudaStreamCreate创建CUDA流。
     
  5. 数据准备：
     1. 在host上声明`input`数据和`output`数组大小，搬运到gpu上
     2. 要执行inference，必须用一个指针数组指定`input`和`output`在gpu中的指针。
     3. 推理并将`output`搬运回CPU

  6. 启动所有工作后，与所有流同步以等待结果:`cudaStreamSynchronize`
  
  7. 按照与创建相反的顺序释放内存
*/