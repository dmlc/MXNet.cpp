/*!
 * Copyright (c) 2015 by Contributors
 */

#ifndef MXNETCPP_H_
#define MXNETCPP_H_

#include <mxnet/base.h>
#include <mxnet/operator.h>
#include <mxnet/c_api.h>

#include <map>
#include <string>
#include <vector>

namespace mxnet {
namespace cpp {

inline std::vector<mx_uint> ShapeToMxuintVec(const mshadow::TShape &shape) {
  std::vector<mx_uint> ret;
  for (size_t i = 0; i < shape.ndim(); ++i) {
    ret.push_back(shape.data()[i]);
  }
  return ret;
}

using OpReqType = mxnet::OpReqType;
using DeviceType = mxnet::Context::DeviceType;
class Mxnet;
// Mxnet * Mxnet_Instance;

class Symbol;
class Operator;
class Executor;
class Optimizer;
class Context {
 public:
  Context(const DeviceType &type, int id);
  DeviceType GetDeviceType() const { return type_; }
  int GetDeviceId() const { return id_; }

 private:
  DeviceType type_;
  int id_;
};

struct NDBlob {
 public:
  NDBlob() : handle_(nullptr) {}
  explicit NDBlob(NDArrayHandle handle) : handle_(handle) {}
  ~NDBlob() { MXNDArrayFree(handle_); }
  NDArrayHandle handle_;

 private:
  NDBlob(const NDBlob &);
  NDBlob &operator=(const NDBlob &);
};

class NDArray {
 public:
  NDArray();
  explicit NDArray(const NDArrayHandle &handle);
  NDArray(const std::vector<mx_uint> &shape, const Context &context,
          bool delay_alloc = true);
  NDArray(mshadow::TShape shape, const Context &context,
          bool delay_alloc = true);
  NDArray(const mx_float *data, size_t size);
  explicit NDArray(const std::vector<mx_float> &data);
  NDArray operator+(real_t scalar);
  NDArray operator-(real_t scalar);
  NDArray operator*(real_t scalar);
  NDArray operator/(real_t scalar);
  NDArray operator+(const NDArray &);
  NDArray operator-(const NDArray &);
  NDArray operator*(const NDArray &);
  NDArray operator/(const NDArray &);
  NDArray &operator=(real_t scalar);
  NDArray &operator+=(real_t scalar);
  NDArray &operator-=(real_t scalar);
  NDArray &operator*=(real_t scalar);
  NDArray &operator/=(real_t scalar);
  NDArray &operator+=(const NDArray &);
  NDArray &operator-=(const NDArray &);
  NDArray &operator*=(const NDArray &);
  NDArray &operator/=(const NDArray &);
  void SyncCopyFromCPU(const mx_float *data, size_t size);
  void SyncCopyFromCPU(const std::vector<mx_float> &data);
  void SyncCopyToCPU(mx_float *data, size_t size);
  NDArray Copy(const Context &) const;
  NDArray Slice(mx_uint begin, mx_uint end) const;
  void WaitToRead();
  void WaitToWrite();
  static void WaitAll();
  static void SampleGaussian(real_t mu, real_t sigma, NDArray *out);
  static void SampleUniform(real_t begin, real_t end, NDArray *out);
  std::vector<mx_uint> GetShape() const;
  mx_float *GetData();
  Context GetContext() const;
  NDArrayHandle GetHandle() const { return blob_ptr_->handle_; }

 private:
  std::shared_ptr<NDBlob> blob_ptr_;
};

struct SymBlob {
 public:
  SymBlob() : handle_(nullptr) {}
  explicit SymBlob(SymbolHandle handle) : handle_(handle) {}
  ~SymBlob() { MXSymbolFree(handle_); }
  SymbolHandle handle_;

 private:
  SymBlob(const SymBlob &);
  SymBlob &operator=(const SymBlob &);
};
class Symbol {
 public:
  // TODO(zhangcheng-qinyinghua)
  // add more input in a single operator
  explicit Symbol(SymbolHandle handle);
  explicit Symbol(const std::string &name);
  Symbol operator+(const Symbol &rhs);
  Symbol operator-(const Symbol &rhs);
  Symbol operator*(const Symbol &rhs);
  Symbol operator/(const Symbol &rhs);
  Symbol Copy() const;
  static Symbol Variable(const std::string &name = "");

  SymbolHandle GetHandle() const {
    return blob_ptr_->handle_;
  }

  Symbol(const std::string &operator_name, const std::string &name,
         std::vector<const char *> input_keys,
         std::vector<SymbolHandle> input_values,
         std::vector<const char *> config_keys,
         std::vector<const char *> config_values);

  void InferShape(
      const std::map<std::string, std::vector<mx_uint> > &arg_shapes,
      std::vector<std::vector<mx_uint> > *in_shape,
      std::vector<std::vector<mx_uint> > *aux_shape,
      std::vector<std::vector<mx_uint> > *out_shape) const;

  std::vector<std::string> ListArguments() const;
  std::vector<std::string> ListOutputs() const;
  std::vector<std::string> ListAuxiliaryStates() const;

  Executor *SimpleBind(const Context &context,
                       const std::map<std::string, NDArray> &args_map,
                       const std::map<std::string, NDArray> &arg_grad_store =
                           std::map<std::string, NDArray>(),
                       const std::map<std::string, OpReqType> &grad_req_type =
                           std::map<std::string, OpReqType>());
  Executor *Bind(const Context &context, const std::vector<NDArray> &arg_arrays,
                 const std::vector<NDArray> &grad_arrays,
                 const std::vector<OpReqType> &grad_reqs,
                 const std::vector<NDArray> &aux_arrays);

 private:
  std::shared_ptr<SymBlob> blob_ptr_;
};

class Operator {
 public:
  explicit Operator(const std::string &operator_name);
  Operator &operator=(const Operator &rhs);

  template <typename T>
  Operator &SetParam(const std::string &name, const T &value) {
    std::string value_str;
    std::stringstream ss;
    ss << value;
    ss >> value_str;

    params_[name] = value_str;
    return *this;
  }
  Operator &SetInput(const std::string &name, Symbol symbol);
  void PushInput(const Symbol &symbol) {
    // input_values.push_back(symbol.GetHandle());
  }
  template <class T, class... Args>
  void PushInput(const Symbol &symbol, const T &t, Args... args) {
    // PushInput(symbol);
    PushInput(t, args...);
  }
  Operator &operator()() {
    return *this;
  }
  Operator &operator()(const Symbol &symbol) {
    input_values.push_back(symbol.GetHandle());
    return *this;
  }
  Operator &operator()(const Symbol &symbol1, const Symbol &symbol2) {
    input_values.push_back(symbol1.GetHandle());
    return *this;
  }
  Operator &operator()(const Symbol &symbol1, const Symbol &symbol2,
                       const Symbol &symbol3) {
    input_values.push_back(symbol2.GetHandle());
    return *this;
  }
  template <typename T, typename... Args>
  Operator &operator()(const Symbol &symbol, const T &t, Args... args) {
    PushInput(symbol, t, args...);
    // PushInput(t,args...);
    return *this;
  }
  // std::string & operator[](const std::string & param_name);
  Symbol CreateSymbol(const std::string &name = "");

 private:
  std::map<std::string, std::string> params_desc_;
  bool variable_params_ = false;
  std::map<std::string, std::string> params_;
  std::vector<SymbolHandle> input_values;
  std::vector<std::string> input_keys;
  AtomicSymbolCreator handle_;
};

class Executor {
 public:
  Executor(const Symbol &symbol, Context context,
           const std::vector<NDArray> &arg_arrays,
           const std::vector<NDArray> &grad_arrays,
           const std::vector<OpReqType> &grad_reqs,
           const std::vector<NDArray> &aux_arrays);
  explicit Executor(const ExecutorHandle &h) { handle_ = h; }
  void Forward(bool is_train) { MXExecutorForward(handle_, is_train ? 1 : 0); }
  void Backward(const std::vector<NDArray> &head_grads =
                    std::vector<NDArray>()) {
    std::vector<NDArrayHandle> head_grads_;
    for (auto d : head_grads) {
      head_grads_.push_back(d.GetHandle());
    }
    if (head_grads_.size() > 0) {
      MXExecutorBackward(handle_, head_grads_.size(), head_grads_.data());
    } else {
      MXExecutorBackward(handle_, 0, nullptr);
    }
  }
  void UpdateAll(Optimizer *opt, float lr, int arg_update_begin = 1,
                 int arg_update_end = -1);
  ~Executor() { MXExecutorFree(handle_); }
  std::vector<NDArray> arg_arrays;
  std::vector<NDArray> grad_arrays;
  std::vector<NDArray> aux_arrays;
  std::vector<NDArray> outputs;

 private:
  Executor(const Executor &e);
  Executor &operator=(const Executor &e);
  ExecutorHandle handle_;
};

class Optimizer {
 public:
  explicit Optimizer(const std::string &opt_type);
  ~Optimizer() {
    if (init_) MXOptimizerFree(handle_);
  }
  template <typename T>
  Optimizer &SetParam(const std::string &name, const T &value) {
    std::string value_str;
    std::stringstream ss;
    ss << value;
    ss >> value_str;

    params_[name] = value_str;
    return *this;
  }
  void Update(int index, NDArray weight, NDArray grad, real_t lr);

 private:
  bool init_;
  Optimizer(const Optimizer &);
  Optimizer &operator=(const Optimizer &);
  OptimizerHandle handle_;
  OptimizerCreator creator_;
  std::map<std::string, std::string> params_;
};

class Mxnet {
 public:
  Mxnet();
  Operator GetSOperator(const std::string &name);
  AtomicSymbolCreator GetSymbolCreator(const std::string &name) {
    return symbol_creators_[name];
  }

 private:
  std::map<std::string, AtomicSymbolCreator> symbol_creators_;
};
static Mxnet *MxNet = new Mxnet();
}  // namespace cpp
}  // namespace mxnet

#endif  // MXNETCPP_H_
