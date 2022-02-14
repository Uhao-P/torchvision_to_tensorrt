
#include "CenteredLayerPlugin.h"

#include <assert.h>

#include <chrono>

#include "openbayesCommon.h"
#include "openbayes_cuda_util/common_util.h"
#include "common.h"
#include "centered_layer.h"
#include "serialize.hpp"


namespace openbayes {
namespace plugin {

namespace {
static const char *PLUGIN_VERSION{"1"};
static const char *PLUGIN_NAME{"CenteredLayerPlugin"};
}  // namespace


CenteredLayer::CenteredLayer(
    const std::string &name)
    : PluginDynamicBase(name){}

CenteredLayer::CenteredLayer(const std::string name,
                                        const void *data,
                                        size_t length)
    : PluginDynamicBase(name) {
}

nvinfer1::IPluginV2DynamicExt *CenteredLayer::clone() const
    PLUGIN_NOEXCEPT {
  CenteredLayer *plugin =
      new CenteredLayer(mLayerName);
  plugin->setPluginNamespace(getPluginNamespace());

  return plugin;
}

nvinfer1::DimsExprs CenteredLayer::getOutputDimensions(
    int outputIndex, const nvinfer1::DimsExprs *inputs, int nbInputs,
    nvinfer1::IExprBuilder &exprBuilder) PLUGIN_NOEXCEPT{
    return inputs[0];
}

bool CenteredLayer::supportsFormatCombination(
    int pos, const nvinfer1::PluginTensorDesc *inOut, int nbInputs,
    int nbOutputs) PLUGIN_NOEXCEPT {
  if (pos == 0) {
    return (inOut[0].type == nvinfer1::DataType::kFLOAT &&
            inOut[0].format == nvinfer1::TensorFormat::kLINEAR);
  } else {
    return inOut[pos].type == inOut[0].type &&
           inOut[pos].format == inOut[0].format;
  }

  return false;
}


void CenteredLayer::configurePlugin(
    const nvinfer1::DynamicPluginTensorDesc *inputs, int nbInputs,
    const nvinfer1::DynamicPluginTensorDesc *outputs,
    int nbOutputs) PLUGIN_NOEXCEPT {
  // Validate input arguments
  assert(nbOutputs == 1);
}

size_t CenteredLayer::getWorkspaceSize(
    const nvinfer1::PluginTensorDesc *inputs, int nbInputs,
    const nvinfer1::PluginTensorDesc *outputs,
    int nbOutputs) const PLUGIN_NOEXCEPT {
  size_t wordSize = samplesCommon::getElementSize(inputs[0].type);
  int batch_size = inputs[0].dims.d[0];
  int channel_size = inputs[0].dims.d[1];
  int width = inputs[0].dims.d[2];
  int height = inputs[0].dims.d[3];
  size_t final_size = openbayes::common::getAlignedSize(
      batch_size * channel_size * width * height * wordSize);
  return final_size + 100;
}

int CenteredLayer::enqueue(
    const nvinfer1::PluginTensorDesc *inputDesc,
    const nvinfer1::PluginTensorDesc *outputDesc, const void *const *inputs,
    void *const *outputs, void *workSpace,
    cudaStream_t stream) PLUGIN_NOEXCEPT {
  int batch_size = inputDesc[0].dims.d[0];
  int inputChannel = inputDesc[0].dims.d[1];
  int inputHeight = inputDesc[0].dims.d[2];
  int inputWidth = inputDesc[0].dims.d[3];

  auto data_type = inputDesc[0].type;

  switch (data_type) {
    case nvinfer1::DataType::kFLOAT:
      centered_layer<float>(
          (float *)outputs[0], (float *)inputs[0], batch_size,
          inputChannel, inputWidth * inputHeight, stream, workSpace);
      break;
    default:
      return 1;
      break;
  }

  return 0;
}

nvinfer1::DataType CenteredLayer::getOutputDataType(
    int index, const nvinfer1::DataType *inputTypes,
    int nbInputs) const PLUGIN_NOEXCEPT {
  return inputTypes[0];
}

const char *CenteredLayer::getPluginType() const PLUGIN_NOEXCEPT {
  return PLUGIN_NAME;
}

const char *CenteredLayer::getPluginVersion() const
    PLUGIN_NOEXCEPT {
  return PLUGIN_VERSION;
}

int CenteredLayer::getNbOutputs() const PLUGIN_NOEXCEPT {
  return 1;
}

size_t CenteredLayer::getSerializationSize() const PLUGIN_NOEXCEPT {
  return 0;
}

void CenteredLayer::serialize(void *buffer) const PLUGIN_NOEXCEPT {}


////////////////////// creator /////////////////////////////
CenteredLayerPluginCreator::CenteredLayerPluginCreator() {
  mPluginAttributes = std::vector<nvinfer1::PluginField>(
      {nvinfer1::PluginField("output_size")});
  mFC.nbFields = mPluginAttributes.size();
  mFC.fields = mPluginAttributes.data();
}

const char *CenteredLayerPluginCreator::getPluginName() const
    PLUGIN_NOEXCEPT {
  return PLUGIN_NAME;
}

const char *CenteredLayerPluginCreator::getPluginVersion() const
    PLUGIN_NOEXCEPT {
  return PLUGIN_VERSION;
}

IPluginV2 *CenteredLayerPluginCreator::createPlugin(
    const char *name, const PluginFieldCollection *fc) PLUGIN_NOEXCEPT {

  for (int i = 0; i < fc->nbFields; i++) {
    if (fc->fields[i].data == nullptr) {
      continue;
    }
    std::string field_name(fc->fields[i].name);


  }
  CenteredLayer *plugin =
      new CenteredLayer(name);
  plugin->setPluginNamespace(getPluginNamespace());
  return plugin;
}


IPluginV2 *CenteredLayerPluginCreator::deserializePlugin(
    const char *name, const void *serialData,
    size_t serialLength) PLUGIN_NOEXCEPT {
  auto plugin = new CenteredLayer(name, serialData, serialLength);
  plugin->setPluginNamespace(getPluginNamespace());
  return plugin;
}
} // namespace plugin
}// namespace openbayes