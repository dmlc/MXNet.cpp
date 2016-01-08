#ifndef _MXNETOP_H
#define _MXNETOP_H

#include "MxNetCpp.h"
namespace mxnet {
namespace cpp {

/*!
 * \brief Take absolute value of the src
 * \param src Source symbolic input to the function
 * \return new symbol
 */
Symbol abs(Symbol src) {
  return Operator("abs")
           .SetParam("src", src)
}

/*!
 * \brief Take sign value of the src
 * \param src Source symbolic input to the function
 * \return new symbol
 */
Symbol sign(Symbol src) {
  return Operator("sign")
           .SetParam("src", src)
}

/*!
 * \brief Take round value of the src
 * \param src Source symbolic input to the function
 * \return new symbol
 */
Symbol round(Symbol src) {
  return Operator("round")
           .SetParam("src", src)
}

/*!
 * \brief Take ceil value of the src
 * \param src Source symbolic input to the function
 * \return new symbol
 */
Symbol ceil(Symbol src) {
  return Operator("ceil")
           .SetParam("src", src)
}

/*!
 * \brief Take floor value of the src
 * \param src Source symbolic input to the function
 * \return new symbol
 */
Symbol floor(Symbol src) {
  return Operator("floor")
           .SetParam("src", src)
}

/*!
 * \brief Take square of the src
 * \param src Source symbolic input to the function
 * \return new symbol
 */
Symbol square(Symbol src) {
  return Operator("square")
           .SetParam("src", src)
}

/*!
 * \brief Take sqrt of the src
 * \param src Source symbolic input to the function
 * \return new symbol
 */
Symbol sqrt(Symbol src) {
  return Operator("sqrt")
           .SetParam("src", src)
}

/*!
 * \brief Take rsqrt of the src
 * \param src Source symbolic input to the function
 * \return new symbol
 */
Symbol rsqrt(Symbol src) {
  return Operator("rsqrt")
           .SetParam("src", src)
}

/*!
 * \brief Take exp of the src
 * \param src Source symbolic input to the function
 * \return new symbol
 */
Symbol exp(Symbol src) {
  return Operator("exp")
           .SetParam("src", src)
}

/*!
 * \brief Take log of the src
 * \param src Source symbolic input to the function
 * \return new symbol
 */
Symbol log(Symbol src) {
  return Operator("log")
           .SetParam("src", src)
}

/*!
 * \brief Take cos of the src
 * \param src Source symbolic input to the function
 * \return new symbol
 */
Symbol cos(Symbol src) {
  return Operator("cos")
           .SetParam("src", src)
}

/*!
 * \brief Take sin of the src
 * \param src Source symbolic input to the function
 * \return new symbol
 */
Symbol sin(Symbol src) {
  return Operator("sin")
           .SetParam("src", src)
}

/*! \brief Activation function to be applied.*/
enum ActivationActType {
  relu = 0,
  sigmoid = 1,
  softrelu = 2,
  tanh = 3
};

/*!
 * \brief Apply activation function to input.Softmax Activation is only available with CUDNN on GPUand will be computed at each location across channel if input is 4D.
 * \param data Input data to activation function.
 * \param act_type Activation function to be applied.
 * \return new symbol
 */
Symbol Activation(Symbol data,
                  ActivationActType act_type) {
  static const char *ActivationActTypeValues[] = {
    "relu",
    "sigmoid",
    "softrelu",
    "tanh"
  };
  return Operator("Activation")
           .SetParam("data", data)
           .SetParam("act_type", ActivationActTypeValues[int(act_type)])
}

/*!
 * \brief Apply batch normalization to input.
 * \param data Input data to batch normalization
 * \param eps Epsilon to prevent div 0
 * \param momentum Momentum for moving average
 * \param fix_gamma Fix gamma while training
 * \return new symbol
 */
Symbol BatchNorm(Symbol data,
                 mx_float eps = 0.001,
                 mx_float momentum = 0.9,
                 bool fix_gamma = True) {
  return Operator("BatchNorm")
           .SetParam("data", data)
           .SetParam("eps", eps)
           .SetParam("momentum", momentum)
           .SetParam("fix_gamma", fix_gamma)
}

/*!
 * \brief Get output from a symbol and pass 0 gradient back
 * \param data Input data.
 * \return new symbol
 */
Symbol BlockGrad(Symbol data) {
  return Operator("BlockGrad")
           .SetParam("data", data)
}

/*!
 * \brief Perform an feature concat on channel dim (dim 1) over all the inputs.
 * \param num_args Number of inputs to be concated.
 * \param dim the dimension to be concated.
 * \return new symbol
 */
Symbol Concat(int num_args,
              int dim = 1) {
  return Operator("Concat")
           .SetParam("num_args", num_args)
           .SetParam("dim", dim)
}

/*!
 * \brief Apply convolution to input then add a bias.
 * \param data Input data to the ConvolutionOp.
 * \param weight Weight matrix.
 * \param bias Bias parameter.
 * \param kernel convolution kernel size: (y, x)
 * \param num_filter convolution filter(channel) number
 * \param stride convolution stride: (y, x)
 * \param dilate convolution dilate: (y, x)
 * \param pad pad for convolution: (y, x)
 * \param num_group Number of groups partition. This option is not supported by CuDNN, you can use SliceChannel to num_group,apply convolution and concat instead to achieve the same need.
 * \param workspace Tmp workspace for convolution (MB).
 * \param no_bias Whether to disable bias parameter.
 * \return new symbol
 */
Symbol Convolution(Symbol data,
                   Symbol weight,
                   Symbol bias,
                   Shape kernel,
                   int num_filter,
                   Shape stride = (1, 1),
                   Shape dilate = (1, 1),
                   Shape pad = (0, 0),
                   int num_group = 1,
                   int64_t workspace = 512,
                   bool no_bias = False) {
  return Operator("Convolution")
           .SetParam("data", data)
           .SetParam("weight", weight)
           .SetParam("bias", bias)
           .SetParam("kernel", kernel)
           .SetParam("num_filter", num_filter)
           .SetParam("stride", stride)
           .SetParam("dilate", dilate)
           .SetParam("pad", pad)
           .SetParam("num_group", num_group)
           .SetParam("workspace", workspace)
           .SetParam("no_bias", no_bias)
}

/*!
 * \brief Crop the 2th and 3th dim of input data, with the corresponding size of w_h orwith widht and height of the second input symbol
 * \param num_args Number of inputs for crop, if equals one, then we will use the h_wfor crop heihgt and width, else if equals two, then we will use the heightand width of the second input symbol, we name crop_like here
 * \param offset corp offset coordinate: (y, x)
 * \param h_w corp height and weight: (h, w)
 * \param center_crop If set to true, then it will use be the center_crop,or it will crop using the shape of crop_like
 * \return new symbol
 */
Symbol Crop(int num_args,
            Shape offset = (0, 0),
            Shape h_w = (0, 0),
            bool center_crop = False) {
  return Operator("Crop")
           .SetParam("num_args", num_args)
           .SetParam("offset", offset)
           .SetParam("h_w", h_w)
           .SetParam("center_crop", center_crop)
}

/*!
 * \brief Apply deconvolution to input then add a bias.
 * \param data Input data to the DeconvolutionOp.
 * \param weight Weight matrix.
 * \param bias Bias parameter.
 * \param kernel deconvolution kernel size: (y, x)
 * \param num_filter deconvolution filter(channel) number
 * \param stride deconvolution stride: (y, x)
 * \param pad pad for deconvolution: (y, x)
 * \param num_group number of groups partition
 * \param workspace Tmp workspace for deconvolution (MB)
 * \param no_bias Whether to disable bias parameter.
 * \return new symbol
 */
Symbol Deconvolution(Symbol data,
                     Symbol weight,
                     Symbol bias,
                     Shape kernel,
                     int num_filter,
                     Shape stride = (1, 1),
                     Shape pad = (0, 0),
                     int num_group = 1,
                     int64_t workspace = 512,
                     bool no_bias = True) {
  return Operator("Deconvolution")
           .SetParam("data", data)
           .SetParam("weight", weight)
           .SetParam("bias", bias)
           .SetParam("kernel", kernel)
           .SetParam("num_filter", num_filter)
           .SetParam("stride", stride)
           .SetParam("pad", pad)
           .SetParam("num_group", num_group)
           .SetParam("workspace", workspace)
           .SetParam("no_bias", no_bias)
}

/*!
 * \brief Apply dropout to input
 * \param data Input data to dropout.
 * \param p Fraction of the input that gets dropped out at training time
 * \return new symbol
 */
Symbol Dropout(Symbol data,
               mx_float p = 0.5) {
  return Operator("Dropout")
           .SetParam("data", data)
           .SetParam("p", p)
}

/*!
 * \brief Perform an elementwise sum over all the inputs.
 * \param num_args Number of inputs to be sumed.
 * \return new symbol
 */
Symbol ElementWiseSum(int num_args) {
  return Operator("ElementWiseSum")
           .SetParam("num_args", num_args)
}

/*!
 * \brief Get embedding for one-hot input
 * \param data Input data to the EmbeddingOp.
 * \param weight Enbedding weight matrix.
 * \param input_dim input dim of one-hot encoding
 * \param output_dim output dim of embedding
 * \return new symbol
 */
Symbol Embedding(Symbol data,
                 Symbol weight,
                 int input_dim,
                 int output_dim) {
  return Operator("Embedding")
           .SetParam("data", data)
           .SetParam("weight", weight)
           .SetParam("input_dim", input_dim)
           .SetParam("output_dim", output_dim)
}

/*!
 * \brief Apply matrix multiplication to input then add a bias.
 * \param data Input data to the FullyConnectedOp.
 * \param weight Weight matrix.
 * \param bias Bias parameter.
 * \param num_hidden Number of hidden nodes of the output.
 * \param no_bias Whether to disable bias parameter.
 * \return new symbol
 */
Symbol FullyConnected(Symbol data,
                      Symbol weight,
                      Symbol bias,
                      int num_hidden,
                      bool no_bias = False) {
  return Operator("FullyConnected")
           .SetParam("data", data)
           .SetParam("weight", weight)
           .SetParam("bias", bias)
           .SetParam("num_hidden", num_hidden)
           .SetParam("no_bias", no_bias)
}

/*!
 * \brief Apply a sparse regularization to the output a sigmoid activation function.
 * \param data Input data.
 * \param sparseness_target The sparseness target
 * \param penalty The tradeoff parameter for the sparseness penalty
 * \param momentum The momentum for running average
 * \return new symbol
 */
Symbol IdentityAttachKLSparseReg(Symbol data,
                                 mx_float sparseness_target = 0.1,
                                 mx_float penalty = 0.001,
                                 mx_float momentum = 0.9) {
  return Operator("IdentityAttachKLSparseReg")
           .SetParam("data", data)
           .SetParam("sparseness_target", sparseness_target)
           .SetParam("penalty", penalty)
           .SetParam("momentum", momentum)
}

/*! \brief Activation function to be applied.*/
enum LeakyReLUActType {
  elu = 0,
  leaky = 1,
  prelu = 2,
  rrelu = 3
};

/*!
 * \brief Apply activation function to input.
 * \param data Input data to activation function.
 * \param act_type Activation function to be applied.
 * \param slope Init slope for the activation. (For leaky and elu only)
 * \param lower_bound Lower bound of random slope. (For rrelu only)
 * \param upper_bound Upper bound of random slope. (For rrelu only)
 * \return new symbol
 */
Symbol LeakyReLU(Symbol data,
                 LeakyReLUActType act_type = LeakyReLUActType::leaky,
                 mx_float slope = 0.25,
                 mx_float lower_bound = 0.125,
                 mx_float upper_bound = 0.334) {
  static const char *LeakyReLUActTypeValues[] = {
    "elu",
    "leaky",
    "prelu",
    "rrelu"
  };
  return Operator("LeakyReLU")
           .SetParam("data", data)
           .SetParam("act_type", LeakyReLUActTypeValues[int(act_type)])
           .SetParam("slope", slope)
           .SetParam("lower_bound", lower_bound)
           .SetParam("upper_bound", upper_bound)
}

/*!
 * \brief Apply convolution to input then add a bias.
 * \param data Input data to the ConvolutionOp.
 * \param nsize normalization window width in elements.
 * \param alpha value of the alpha variance scaling parameter in the normalization formula
 * \param beta value of the beta power parameter in the normalization formula
 * \param knorm value of the k parameter in normalization formula
 * \return new symbol
 */
Symbol LRN(Symbol data,
           int nsize,
           mx_float alpha = 0.0001,
           mx_float beta = 0.75,
           mx_float knorm = 2) {
  return Operator("LRN")
           .SetParam("data", data)
           .SetParam("nsize", nsize)
           .SetParam("alpha", alpha)
           .SetParam("beta", beta)
           .SetParam("knorm", knorm)
}

/*! \brief Pooling type to be applied.*/
enum PoolingPoolType {
  avg = 0,
  max = 1,
  sum = 2
};

/*!
 * \brief Perform spatial pooling on inputs.
 * \param data Input data to the pooling operator.
 * \param kernel pooling kernel size: (y, x)
 * \param pool_type Pooling type to be applied.
 * \param stride stride: for pooling (y, x)
 * \param pad pad for pooling: (y, x)
 * \return new symbol
 */
Symbol Pooling(Symbol data,
               Shape kernel,
               PoolingPoolType pool_type,
               Shape stride = (1, 1),
               Shape pad = (0, 0)) {
  static const char *PoolingPoolTypeValues[] = {
    "avg",
    "max",
    "sum"
  };
  return Operator("Pooling")
           .SetParam("data", data)
           .SetParam("kernel", kernel)
           .SetParam("pool_type", PoolingPoolTypeValues[int(pool_type)])
           .SetParam("stride", stride)
           .SetParam("pad", pad)
}

/*!
 * \brief Use linear regression for final output, this is used on final output of a net.
 * \param data Input data to function.
 * \param label Input label to function.
 * \param grad_scale Scale the gradient by a float factor
 * \return new symbol
 */
Symbol LinearRegressionOutput(Symbol data,
                              Symbol label,
                              mx_float grad_scale = 1) {
  return Operator("LinearRegressionOutput")
           .SetParam("data", data)
           .SetParam("label", label)
           .SetParam("grad_scale", grad_scale)
}

/*!
 * \brief Use mean absolute error regression for final output, this is used on final output of a net.
 * \param data Input data to function.
 * \param label Input label to function.
 * \param grad_scale Scale the gradient by a float factor
 * \return new symbol
 */
Symbol MAERegressionOutput(Symbol data,
                           Symbol label,
                           mx_float grad_scale = 1) {
  return Operator("MAERegressionOutput")
           .SetParam("data", data)
           .SetParam("label", label)
           .SetParam("grad_scale", grad_scale)
}

/*!
 * \brief Use Logistic regression for final output, this is used on final output of a net.
Logistic regression is suitable for binary classification or probability prediction tasks.
 * \param data Input data to function.
 * \param label Input label to function.
 * \param grad_scale Scale the gradient by a float factor
 * \return new symbol
 */
Symbol LogisticRegressionOutput(Symbol data,
                                Symbol label,
                                mx_float grad_scale = 1) {
  return Operator("LogisticRegressionOutput")
           .SetParam("data", data)
           .SetParam("label", label)
           .SetParam("grad_scale", grad_scale)
}

/*!
 * \brief Reshape input to target shape
 * \param data Input data to  reshape.
 * \param target_shape Target new shape. One and only one dim can be 0, in which case it will be infered from the rest of dims
 * \return new symbol
 */
Symbol Reshape(Symbol data,
               Shape target_shape) {
  return Operator("Reshape")
           .SetParam("data", data)
           .SetParam("target_shape", target_shape)
}

/*!
 * \brief Flatten input
 * \param data Input data to  flatten.
 * \return new symbol
 */
Symbol Flatten(Symbol data) {
  return Operator("Flatten")
           .SetParam("data", data)
}

/*!
 * \brief Slice channel into many outputs with equally divided channel
 * \param num_outputs Number of outputs to be sliced.
 * \return new symbol
 */
Symbol SliceChannel(int num_outputs) {
  return Operator("SliceChannel")
           .SetParam("num_outputs", num_outputs)
}

/*! \brief Softmax Mode. If set to instance, this operator will compute a softmax for each instance in the batch; this is the default mode. If set to channel, this operator will compute a num_channel-class softmax at each position of each instance; this can be used for fully convolutional network, image segmentation, etc.*/
enum SoftmaxActivationType {
  channel = 0,
  instance = 1
};

/*!
 * \brief Apply softmax activation to input. This is intended for internal layers. For output (loss layer) please use SoftmaxOutput. If type=instance, this operator will compute a softmax for each instance in the batch; this is the default mode. If type=channel, this operator will compute a num_channel-class softmax at each position of each instance; this can be used for fully convolutional network, image segmentation, etc.
 * \param data Input data to activation function.
 * \param type Softmax Mode. If set to instance, this operator will compute a softmax for each instance in the batch; this is the default mode. If set to channel, this operator will compute a num_channel-class softmax at each position of each instance; this can be used for fully convolutional network, image segmentation, etc.
 * \return new symbol
 */
Symbol SoftmaxActivation(Symbol data,
                         SoftmaxActivationType type = SoftmaxActivationType::instance) {
  static const char *SoftmaxActivationTypeValues[] = {
    "channel",
    "instance"
  };
  return Operator("SoftmaxActivation")
           .SetParam("data", data)
           .SetParam("type", SoftmaxActivationTypeValues[int(type)])
}

/*!
 * \brief Perform a softmax transformation on input, backprop with logloss.
 * \param data Input data to softmax.
 * \param grad_scale Scale the gradient by a float factor
 * \param ignore_label the ignore_label will not work in backward, and this onlybe used when multi_output=true
 * \param multi_output If set to true, for a (n,k,x_1,..,x_n) dimensionalinput tensor, softmax will generate n*x_1*...*x_n output, eachhas k classes
 * \param use_ignore If set to true, the ignore_label value will not contributorto the backward gradient
 * \return new symbol
 */
Symbol SoftmaxOutput(Symbol data,
                     mx_float grad_scale = 1,
                     mx_float ignore_label = -1,
                     bool multi_output = False,
                     bool use_ignore = False) {
  return Operator("SoftmaxOutput")
           .SetParam("data", data)
           .SetParam("grad_scale", grad_scale)
           .SetParam("ignore_label", ignore_label)
           .SetParam("multi_output", multi_output)
           .SetParam("use_ignore", use_ignore)
}

/*!
 * \brief DEPRECATED: Perform a softmax transformation on input. Please use SoftmaxOutput
 * \param data Input data to softmax.
 * \param grad_scale Scale the gradient by a float factor
 * \param ignore_label the ignore_label will not work in backward, and this onlybe used when multi_output=true
 * \param multi_output If set to true, for a (n,k,x_1,..,x_n) dimensionalinput tensor, softmax will generate n*x_1*...*x_n output, eachhas k classes
 * \param use_ignore If set to true, the ignore_label value will not contributorto the backward gradient
 * \return new symbol
 */
Symbol Softmax(Symbol data,
               mx_float grad_scale = 1,
               mx_float ignore_label = -1,
               bool multi_output = False,
               bool use_ignore = False) {
  return Operator("Softmax")
           .SetParam("data", data)
           .SetParam("grad_scale", grad_scale)
           .SetParam("ignore_label", ignore_label)
           .SetParam("multi_output", multi_output)
           .SetParam("use_ignore", use_ignore)
}

/*!
 * \brief Apply swapaxis to input.
 * \param data Input data to the SwapAxisOp.
 * \param dim1 the first axis to be swapped.
 * \param dim2 the second axis to be swapped.
 * \return new symbol
 */
Symbol SwapAxis(Symbol data,
                int dim1 = 0,
                int dim2 = 0) {
  return Operator("SwapAxis")
           .SetParam("data", data)
           .SetParam("dim1", dim1)
           .SetParam("dim2", dim2)
}

/*! \brief upsampling method*/
enum UpSamplingSampleType {
  bilinear = 0,
  nearest = 1
};

/*! \brief How to handle multiple input. concat means concatenate upsampled images along the channel dimension. sum means add all images together, only available for nearest neighbor upsampling.*/
enum UpSamplingMultiInputMode {
  concat = 0,
  sum = 1
};

/*!
 * \brief Perform nearest neighboor/bilinear up sampling to inputs
 * \param scale Up sampling scale
 * \param sample_type upsampling method
 * \param num_args Number of inputs to be upsampled. For nearest neighbor upsampling, this can be 1-N; the size of output will be(scale*h_0,scale*w_0) and all other inputs will be upsampled to thesame size. For bilinear upsampling this must be 2; 1 input and 1 weight.
 * \param num_filter Input filter. Only used by nearest sample_type.
 * \param multi_input_mode How to handle multiple input. concat means concatenate upsampled images along the channel dimension. sum means add all images together, only available for nearest neighbor upsampling.
 * \return new symbol
 */
Symbol UpSampling(int scale,
                  UpSamplingSampleType sample_type,
                  int num_args,
                  int num_filter = 0,
                  UpSamplingMultiInputMode multi_input_mode = UpSamplingMultiInputMode::concat) {
  static const char *UpSamplingSampleTypeValues[] = {
    "bilinear",
    "nearest"
  };
  static const char *UpSamplingMultiInputModeValues[] = {
    "concat",
    "sum"
  };
  return Operator("UpSampling")
           .SetParam("scale", scale)
           .SetParam("sample_type", UpSamplingSampleTypeValues[int(sample_type)])
           .SetParam("num_args", num_args)
           .SetParam("num_filter", num_filter)
           .SetParam("multi_input_mode", UpSamplingMultiInputModeValues[int(multi_input_mode)])
}

} //namespace cpp
} //namespace mxnet
