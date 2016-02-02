/*!
 *  Copyright (c) 2016 by Contributors
 * \file MxNetCpp.hpp
 * \brief implementation of Mxnet, Context, Operator, Optimizer
 * \author Chuntao Hong, Zhang Chen
 */

#ifndef MXNETCPP_HPP_C6LTBYYW
#define MXNETCPP_HPP_C6LTBYYW

#include <vector>
#include <string>
#include <map>
#include "MxNetCpp.h"

namespace mxnet {
namespace cpp {
template <typename K, typename V>
inline bool HasKey(const std::map<K, V> &m, const K &key) {
  return m.find(key) != m.end();
}

Mxnet::Mxnet() {
  mx_uint num_symbol_creators = 0;
  AtomicSymbolCreator *symbol_creators = nullptr;
  int r =
      MXSymbolListAtomicSymbolCreators(&num_symbol_creators, &symbol_creators);

  CHECK_EQ(r, 0);
  for (mx_uint i = 0; i < num_symbol_creators; i++) {
    const char *name;
    const char *description;
    mx_uint num_args;
    const char **arg_names;
    const char **arg_type_infos;
    const char **arg_descriptions;
    const char *key_var_num_args;
    r = MXSymbolGetAtomicSymbolInfo(symbol_creators[i], &name, &description,
                                    &num_args, &arg_names, &arg_type_infos,
                                    &arg_descriptions, &key_var_num_args);
    CHECK_EQ(r, 0);
    std::string symbolInfo;
    symbolInfo = symbolInfo + name + "(";
    for (mx_uint j = 0; j < num_args; j++) {
      if (j != 0) {
        symbolInfo += " ,";
      }
      symbolInfo += arg_names[j];
    }
    symbolInfo += ")";
    // LOG(INFO) << symbolInfo;
    symbol_creators_[name] = symbol_creators[i];
  }
}

/*context*/
Context::Context(const DeviceType &type, int id) {
  type_ = type;
  id_ = id;
}

/*
 * Operator
 *
 * */
Operator::Operator(const std::string &operator_name) {
  handle_ = MxNet->GetSymbolCreator(operator_name);
}

Symbol Operator::CreateSymbol(const std::string &name) {
  const char *pname = name == "" ? nullptr : name.c_str();

  SymbolHandle symbol_handle;
  std::vector<const char *> input_keys;
  std::vector<const char *> param_keys;
  std::vector<const char *> param_values;

  for (auto &data : params_) {
    param_keys.push_back(data.first.c_str());
    param_values.push_back(data.second.c_str());
  }
  for (auto &data : this->input_keys) {
    input_keys.push_back(data.c_str());
  }
  const char **input_keys_p =
      (input_keys.size() > 0) ? input_keys.data() : nullptr;

  MXSymbolCreateAtomicSymbol(handle_, param_keys.size(), param_keys.data(),
                             param_values.data(), &symbol_handle);
  MXSymbolCompose(symbol_handle, pname, input_values.size(), input_keys_p,
                  input_values.data());
  return Symbol(symbol_handle);
}
Operator &Operator::SetInput(const std::string &name, Symbol symbol) {
  input_keys.push_back(name.c_str());
  input_values.push_back(symbol.GetHandle());
  return *this;
}

/*
 * Optimizer
 *
 * */
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
}  // namespace cpp
}  // namespace mxnet


#endif /* end of include guard: MXNETCPP_HPP_C6LTBYYW */
