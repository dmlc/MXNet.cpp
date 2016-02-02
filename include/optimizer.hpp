#ifndef MXNETCPP_OPTIMIZER_HPP
#define MXNETCPP_OPTIMIZER_HPP

#include "optimizer.h"

namespace mxnet {
namespace cpp {

Optimizer::Optimizer(const std::string &opt_type) {
  MXOptimizerFindCreator(opt_type.c_str(), &creator_);
  init_ = false;
}
void Optimizer::Update(int index, NDArray weight, NDArray grad, mx_float lr) {
  if (!init_) {
    std::vector<const char *> param_keys;
    std::vector<const char *> param_values;
    for (const auto &k_v : params_) {
      param_keys.push_back(k_v.first.c_str());
      param_values.push_back(k_v.second.c_str());
    }
    MXOptimizerCreateOptimizer(creator_, params_.size(), param_keys.data(),
      param_values.data(), &handle_);
    init_ = true;
  }
  MXOptimizerUpdate(handle_, index, weight.GetHandle(), grad.GetHandle(), lr);
}

}
}

#endif // MXNETCPP_OPTIMIZER_HPP