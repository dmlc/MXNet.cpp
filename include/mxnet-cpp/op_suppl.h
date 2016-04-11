/*!
*  Copyright (c) 2016 by Contributors
* \file op_suppl.h
* \brief A supplement and amendment of the operators from op.h
* \author Zhang Chen, zhubuntu
*/

#ifndef OP_SUPPL_H
#define OP_SUPPL_H

#include <string>
#include <vector>
#include "mxnet-cpp/base.h"
#include "mxnet-cpp/shape.h"
#include "mxnet-cpp/operator.h"
#include "mxnet-cpp/MxNetCpp.h"

namespace mxnet {
namespace cpp {

Symbol _Plus(Symbol lhs, Symbol rhs) {
  return Operator("_Plus")
           .SetInput("lhs", lhs)
           .SetInput("rhs", rhs)
           .CreateSymbol();
}
Symbol _Mul(Symbol lhs, Symbol rhs) {
  return Operator("_Mul")
           .SetInput("lhs", lhs)
           .SetInput("rhs", rhs)
           .CreateSymbol();
}
Symbol _Minus(Symbol lhs, Symbol rhs) {
  return Operator("_Minus")
           .SetInput("lhs", lhs)
           .SetInput("rhs", rhs)
           .CreateSymbol();
}
Symbol _Div(Symbol lhs, Symbol rhs) {
  return Operator("_Div")
           .SetInput("lhs", lhs)
           .SetInput("rhs", rhs)
           .CreateSymbol();
}
Symbol _Power(Symbol lhs, Symbol rhs) {
  return Operator("_Power")
           .SetInput("lhs", lhs)
           .SetInput("rhs", rhs)
           .CreateSymbol();
}
Symbol _Maximum(Symbol lhs, Symbol rhs) {
  return Operator("_Maximum")
           .SetInput("lhs", lhs)
           .SetInput("rhs", rhs)
           .CreateSymbol();
}
Symbol _Minimum(Symbol lhs, Symbol rhs) {
  return Operator("_Minimum")
           .SetInput("lhs", lhs)
           .SetInput("rhs", rhs)
           .CreateSymbol();
}
Symbol _PlusScalar(Symbol lhs, mx_float scalar, bool scalar_on_left) {
  return Operator("_PlusScalar")
           .SetParam("scalar", scalar)
           .SetParam("scalar_on_left", scalar_on_left)
           .SetInput("lhs", lhs)
           .CreateSymbol();
}
Symbol _MinusScalar(Symbol lhs, mx_float scalar, bool scalar_on_left) {
  return Operator("_MinusScalar")
           .SetParam("scalar", scalar)
           .SetParam("scalar_on_left", scalar_on_left)
           .SetInput("lhs", lhs)
           .CreateSymbol();
}
Symbol _MulScalar(Symbol lhs, mx_float scalar, bool scalar_on_left) {
  return Operator("_MulScalar")
           .SetParam("scalar", scalar)
           .SetParam("scalar_on_left", scalar_on_left)
           .SetInput("lhs", lhs)
           .CreateSymbol();
}
Symbol _DivScalar(Symbol lhs, mx_float scalar, bool scalar_on_left) {
  return Operator("_DivScalar")
           .SetParam("scalar", scalar)
           .SetParam("scalar_on_left", scalar_on_left)
           .SetInput("lhs", lhs)
           .CreateSymbol();
}
Symbol _PowerScalar(Symbol lhs, mx_float scalar, bool scalar_on_left) {
  return Operator("_PowerScalar")
           .SetParam("scalar", scalar)
           .SetParam("scalar_on_left", scalar_on_left)
           .SetInput("lhs", lhs)
           .CreateSymbol();
}
Symbol _MaximumScalar(Symbol lhs, mx_float scalar, bool scalar_on_left) {
  return Operator("_MaximumScalar")
           .SetParam("scalar", scalar)
           .SetParam("scalar_on_left", scalar_on_left)
           .SetInput("lhs", lhs)
           .CreateSymbol();
}
Symbol _MinimumScalar(Symbol lhs, mx_float scalar, bool scalar_on_left) {
  return Operator("_MinimumScalar")
           .SetParam("scalar", scalar)
           .SetParam("scalar_on_left", scalar_on_left)
           .SetInput("lhs", lhs)
           .CreateSymbol();
}
// TODO(zhangcheng-qinyinghua)
//  make crop function run in op.h
//  This function is due to [zhubuntu](https://github.com/zhubuntu)
inline Symbol Crop(const std::string& symbol_name,
    int num_args,
    Symbol data,
    Symbol crop_like,
    Shape offset = Shape(0, 0),
    Shape h_w = Shape(0, 0),
    bool center_crop = false) {
  return Operator("Crop")
    .SetParam("num_args", num_args)
    .SetParam("offset", offset)
    .SetParam("h_w", h_w)
    .SetParam("center_crop", center_crop)
    .SetInput("arg0", data)
    .SetInput("arg1", crop_like)
    .CreateSymbol(symbol_name);
}
}  // namespace cpp
}  // namespace mxnet


#endif /* end of include guard: OP_SUPPL_H */

