#pragma once

#include <cublas_v2.h>

#include <memory>
#include <string>
#include <vector>

#include "../common/layer_base.h"
#include "NvInferPlugin.h"
namespace openbayes {
namespace plugin {
class CenteredLayer : public PluginDynamicBase {
    public:
        /*
        构造函数和析构函数,构造函数一般设置为三个。
        第一个用于在parse阶段，PluginCreator用于创建该插件时调用的构造函数，需要传递权重信息以及参数。
        第二个用于在clone阶段，复制这个plugin时会用到的构造函数。
        第三个用于在deserialize阶段，用于将序列化好的权重和参数传入该plugin并创建爱你哦。
        */
        CenteredLayer(const std::string &name);

        // CenteredLayer(int numInputs, const size_t* copySize);

        CenteredLayer(const std::string name,const void* data, size_t length);

        //注意需要把默认构造函数删掉
        CenteredLayer() = delete; 

        //获取plugin版本
        const char* getPluginVersion() const PLUGIN_NOEXCEPT override;

        nvinfer1::IPluginV2DynamicExt *clone() const PLUGIN_NOEXCEPT override;

        //TensorRT支持Dynamic-shape的时候，batch这一维度必须是explicit的，也就是说，TensorRT处理的维度从以往的三维[3,-1,-1]变成了[1,3,-1,-1]。而且这个batch维度在getOutputDimensions中是可以获取到的。
        nvinfer1::DimsExprs getOutputDimensions(
        int outputIndex, const nvinfer1::DimsExprs *inputs, int nbInputs,
        nvinfer1::IExprBuilder &exprBuilder) PLUGIN_NOEXCEPT override;

        /*
        TensorRT调用此方法以判断pos索引的输入/输出是否支持inOut[pos].format和inOut[pos].type指定的格式/数据类型。
        如果插件支持inOut[pos]处的格式/数据类型，则返回true。 
        如果是否支持取决于其他的输入/输出格式/数据类型，则插件可以使其结果取决于inOut[0..pos-1]中的格式/数据类型，该格式/数据类型将设置为插件支持的值。 
        这个函数不需要检查inOut[pos + 1..nbInputs + nbOutputs-1]，pos的决定必须仅基于inOut[0..pos]。
        */
        bool supportsFormatCombination(int pos,
                                    const nvinfer1::PluginTensorDesc *inOut,
                                    int nbInputs,
                                    int nbOutputs) PLUGIN_NOEXCEPT override;

        /*
        配置这个插件op，判断输入和输出类型数量是否正确。官方还提到通过这个配置信息可以告知TensorRT去选择合适的算法(algorithm)去调优这个模型。
        但自动调优目前还没有尝试过，我们一般自己写的plugin执行代码都是定死的，所谓的调优步骤可能更多地针对官方的op。
        */
        void configurePlugin(const nvinfer1::DynamicPluginTensorDesc *in,
                        int nbInputs,
                        const nvinfer1::DynamicPluginTensorDesc *out,
                        int nbOutputs) PLUGIN_NOEXCEPT override;

        /*
        这个函数需要返回这个插件op需要中间显存变量的实际数据大小(bytesize)，这个是通过TensorRT的接口去获取，是比较规范的方式。
        我们需要在这里确定这个op需要多大的显存空间去运行，在实际运行的时候就可以直接使用TensorRT开辟好的空间而不是自己去申请显存空间。
        */
        size_t getWorkspaceSize(const nvinfer1::PluginTensorDesc *inputs,
                            int nbInputs,
                            const nvinfer1::PluginTensorDesc *outputs,
                            int nbOutputs) const PLUGIN_NOEXCEPT override;

        //实际插件op的执行函数，我们自己实现的cuda操作就放到这里(当然C++写的op也可以放进来，不过因为是CPU执行，速度就比较慢了)，与往常一样接受输入inputs产生输出outputs，传给相应的指针就可以。
        int enqueue(const nvinfer1::PluginTensorDesc *inputDesc,
                    const nvinfer1::PluginTensorDesc *outputDesc,
                    const void *const *inputs, void *const *outputs, void *workspace,
                    cudaStream_t stream) PLUGIN_NOEXCEPT override;

        //返回结果的类型，一般来说我们插件op返回结果类型与输入类型一致：
        nvinfer1::DataType getOutputDataType(
            int index, const nvinfer1::DataType *inputTypes,
            int nbInputs) const PLUGIN_NOEXCEPT override;

        //获取plugin类型
        const char *getPluginType() const PLUGIN_NOEXCEPT override;
        
        //插件op返回多少个Tensor，比如MyCustomPlugin这个操作只输出一个Tensor(也就是一个output)，所以直接return 1：
        int getNbOutputs() const PLUGIN_NOEXCEPT override;

        //返回序列化时需要写多少字节到buffer中。
        size_t getSerializationSize() const PLUGIN_NOEXCEPT override;

        //把需要用的数据按照顺序序列化到buffer里头。
        void serialize(void *buffer) const PLUGIN_NOEXCEPT override;

        // /*
        // 初始化函数，在这个插件准备开始run之前执行。
        // 主要初始化一些提前开辟空间的参数，一般是一些cuda操作需要的参数(例如conv操作需要执行卷积操作，我们就需要提前开辟weight和bias的显存)，假如我们的算子需要这些参数，则在这里需要提前开辟显存。
        // 需要注意的是，如果插件算子需要开辟比较大的显存空间，不建议自己去申请显存空间，可以使用Tensorrt官方接口传过来的workspace指针来获取显存空间。
        // 因为如果这个插件被一个网络调用了很多次，而这个插件op需要开辟很多显存空间，那么TensorRT在构建network的时候会根据这个插件被调用的次数开辟很多显存，很容易导致显存溢出。
        // */
        // int initialize() noexcept override;

        // //析构函数则需要执行terminate，terminate函数就是释放这个op之前开辟的一些显存空间
        // void terminate() noexcept override;


        // // bool isOutputBroadcastAcrossBatch(int outputIndex, const bool* inputIsBroadcasted, int nbInputs) const noexcept override;

        // // bool canBroadcastInputAcrossBatch(int inputIndex) const noexcept override;


        // //从它的执行上下文中分离插件对象。当执行上下文被销毁或上下文资源从上下文中取消分配时，会为每个插件自动调用此函数。如果插件拥有 per-context 资源，则可以在此处发布。
        // void detachFromContext() noexcept override;


        // void destroy() noexcept override;

        // //如果这个op使用到了一些其他东西，例如cublas handle，可以直接借助TensorRT内部提供的cublas handle
        // void attachToContext(
        //     cudnnContext* cudnnContext, cublasContext* cublasContext, IGpuAllocator* gpuAllocator) noexcept override;


        // //设置plugin命名空间
        // void setPluginNamespace(const char* pluginNamespace) noexcept override;

        // ////获取plugin命名空间
        // const char* getPluginNamespace() const noexcept override;

};

class CenteredLayerPluginCreator : public PluginCreatorBase
{
public:
    //构造函数
    CenteredLayerPluginCreator();

    const char *getPluginName() const PLUGIN_NOEXCEPT override;

    const char *getPluginVersion() const PLUGIN_NOEXCEPT override;

    //这个成员函数作用是通过PluginFieldCollection去创建plugin，将op需要的权重和参数一个一个取出来，然后调用上文提到的第一个构造函数
    nvinfer1::IPluginV2 *createPlugin(const char *name,
                                    const nvinfer1::PluginFieldCollection *fc)
      PLUGIN_NOEXCEPT override;

    //这个函数会被onnx-tensorrt的一个叫做TRT_PluginV2的转换op调用，这个op会读取onnx模型的data数据将其反序列化到network中
    nvinfer1::IPluginV2 *deserializePlugin(
      const char *name, const void *serialData,
      size_t serialLength) PLUGIN_NOEXCEPT override;

};



} // namespace plugin
} // namespace nvinfer1


