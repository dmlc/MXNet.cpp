/*!
 *  Copyright (c) 2016 by Contributors
 * \file MxNetCpp.h
 * \brief the main definations of MxNetCpp
 * \author Chuntao Hong, Zhang Chen
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
/*!
 * \brief transform TShape to mx_uint vector
 *
 * \param shape the original TShape
 * \return the vector of mx_uint, with shape.ndim elements
 */
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
class Symbol;
class Operator;
class Executor;
class Optimizer;

/*!
 * \brief Context interface
 */
class Context {
 public:
  /*!
   * \brief Context constructor
   * \param type type of the device
   * \param id id of the device
   */
  Context(const DeviceType &type, int id);
  /*!
   * \return the type of the device 
   */
  DeviceType GetDeviceType() const { return type_; }
  /*!
   * \return the id of the device 
   */
  int GetDeviceId() const { return id_; }

 private:
  DeviceType type_;
  int id_;
};

/*!
 * \brief struct to store NDArrayHandle
 */
struct NDBlob {
 public:
  /*!
   * \brief default constructor
   */
  NDBlob() : handle_(nullptr) {}
  /*!
   * \brief construct with a NDArrayHandle
   * \param handle NDArrayHandle to store
   */
  explicit NDBlob(NDArrayHandle handle) : handle_(handle) {}
  /*!
   * \brief destructor, free the NDArrayHandle
   */
  ~NDBlob() { MXNDArrayFree(handle_); }
  /*!
   * \brief the NDArrayHandle
   */
  NDArrayHandle handle_;

 private:
  NDBlob(const NDBlob &);
  NDBlob &operator=(const NDBlob &);
};

/*!
 * \brief NDArray interface
 */
class NDArray {
 public:
  /*!
   * \brief construct with a none handle
   */
  NDArray();
  /*!
   * \brief construct with a NDArrayHandle
   */
  explicit NDArray(const NDArrayHandle &handle);
  /*!
   * \brief construct a new dynamic NDArray
   * \param shape the shape of array
   * \param constext context of NDArray
   * \param delay_alloc whether delay the allocation
   */
  NDArray(const std::vector<mx_uint> &shape, const Context &context,
          bool delay_alloc = true);
  /*!
   * \brief construct a new dynamic NDArray
   * \param shape the shape of array
   * \param constext context of NDArray
   * \param delay_alloc whether delay the allocation
   */
  NDArray(mshadow::TShape shape, const Context &context,
          bool delay_alloc = true);
  NDArray(const mx_float *data, size_t size);
  explicit NDArray(const std::vector<mx_float> &data);
  // TODO(zhangcheng-qinyinghua)
  // implement all the operators
  NDArray operator+(real_t scalar);
  NDArray operator-(real_t scalar);
  NDArray operator*(real_t scalar);
  NDArray operator/(real_t scalar);
  NDArray operator+(const NDArray &);
  NDArray operator-(const NDArray &);
  NDArray operator*(const NDArray &);
  NDArray operator/(const NDArray &);
  /*!
   * \brief set all the elements in ndarray to be scalar
   * \param scalar the scalar to set
   * \return reference of self
   */
  NDArray &operator=(real_t scalar);
  /*!
   * \brief elementwise add to current space
   *  this mutate the current NDArray
   * \param scalar the data to add
   * \return reference of self
   */
  NDArray &operator+=(real_t scalar);
  /*!
   * \brief elementwise subtract from current ndarray
   * this mutate the current NDArray
   * \param scalar the data to substract
   * \return reference of self
   */
  NDArray &operator-=(real_t scalar);
  /*!
   * \brief elementwise multiplication to current ndarray
   *  this mutate the current NDArray
   * \param scalar the data to substract
   * \return reference of self
   */
  NDArray &operator*=(real_t scalar);
  /*!
   * \brief elementwise division from current ndarray
   *  this mutate the current NDArray
   * \param scalar the data to substract
   * \return reference of self
   */
  NDArray &operator/=(real_t scalar);
  /*!
   * \brief elementwise add to current space
   *  this mutate the current NDArray
   * \param src the data to add
   * \return reference of self
   */
  NDArray &operator+=(const NDArray & src);
  /*!
   * \brief elementwise subtract from current ndarray
   * this mutate the current NDArray
   * \param src the data to substract
   * \return reference of self
   */
  NDArray &operator-=(const NDArray & src);
  /*!
   * \brief elementwise multiplication to current ndarray
   *  this mutate the current NDArray
   * \param src the data to substract
   * \return reference of self
   */
  NDArray &operator*=(const NDArray & src);
  /*!
   * \brief elementwise division from current ndarray
   *  this mutate the current NDArray
   * \param src the data to substract
   * \return reference of self
   */
  NDArray &operator/=(const NDArray & src);
  /*!
   * \brief Do a synchronize copy from a continugous CPU memory region.
   *
   *  This function will call WaitToWrite before the copy is performed.
   *  This is useful to copy data from existing memory region that are
   *  not wrapped by NDArray(thus dependency not being tracked).
   *
   * \param data the data source to copy from.
   * \param size the memory size we want to copy from.
   */
  void SyncCopyFromCPU(const mx_float *data, size_t size);
  /*!
   * \brief Do a synchronize copy from a continugous CPU memory region.
   *
   *  This function will call WaitToWrite before the copy is performed.
   *  This is useful to copy data from existing memory region that are
   *  not wrapped by NDArray(thus dependency not being tracked).
   *
   * \param data the data source to copy from, int the form of mx_float vector
   */
  void SyncCopyFromCPU(const std::vector<mx_float> &data);
  /*!
   * \brief Do a synchronize copy to a continugous CPU memory region.
   *
   *  This function will call WaitToRead before the copy is performed.
   *  This is useful to copy data from existing memory region that are
   *  not wrapped by NDArray(thus dependency not being tracked).
   *
   * \param data the data source to copyinto.
   * \param size the memory size we want to copy into.
   */
  void SyncCopyToCPU(mx_float *data, size_t size);
  /*!
   * \brief return a new copy this NDArray
   * \param context the new context of this NDArray
   * \return the new copy
   */
  NDArray Copy(const Context &) const;
  /*!
   * \brief Slice a NDArray
   * \param begin begin index in first dim
   * \param end end index in first dim
   * \return sliced NDArray
   */
  NDArray Slice(mx_uint begin, mx_uint end) const;
  /*!
   * \brief Block until all the pending write operations with respect
   *    to current NDArray are finished, and read can be performed.
   */
  void WaitToRead() const;
  /*!
   * \brief Block until all the pending read/write operations with respect
   *    to current NDArray are finished, and write can be performed.
   */
  void WaitToWrite();
  /*!
   * \brief Block until all the pending read/write operations with respect
   *    to current NDArray are finished, and read/write can be performed.
   */
  static void WaitAll();
  /*!
   * \brief Sample gaussian distribution for each elements of out.
   * \param mu mean of gaussian distribution.
   * \param sigma standard deviation of gaussian distribution.
   * \param out output NDArray.
   */
  static void SampleGaussian(real_t mu, real_t sigma, NDArray *out);
  /*!
   * \brief Sample uniform distribution for each elements of out.
   * \param begin lower bound of distribution.
   * \param end upper bound of distribution.
   * \param out output NDArray.
   */
  static void SampleUniform(real_t begin, real_t end, NDArray *out);
  /*!
   * \return the shape of current NDArray, in the form of mx_uint vector
   */
  std::vector<mx_uint> GetShape() const;
  /*!
   * \return the data pointer to the current NDArray
   */
  mx_float *GetData();
  const mx_float *NDArray::GetData() const;
  /*!
   * \return the context of NDArray
   */
  Context GetContext() const;
  /*!
   * \return the NDArrayHandle of the current NDArray
   */
  NDArrayHandle GetHandle() const { return blob_ptr_->handle_; }

 private:
  std::shared_ptr<NDBlob> blob_ptr_;
};

/*!
 * \brief struct to store SymbolHandle
 */
struct SymBlob {
 public:
  /*!
   * \brief default constructor
   */
  SymBlob() : handle_(nullptr) {}
  /*!
   * \brief construct with SymbolHandle to store
   */
  explicit SymBlob(SymbolHandle handle) : handle_(handle) {}
  /*!
   * \brief destructor, free the SymbolHandle
   */
  ~SymBlob() { MXSymbolFree(handle_); }
  /*!
   * \brief the SymbolHandle to store
   */
  SymbolHandle handle_;

 private:
  SymBlob(const SymBlob &);
  SymBlob &operator=(const SymBlob &);
};

/*!
 * \brief Symbol interface
 */
class Symbol {
 public:
  // TODO(zhangcheng-qinyinghua)
  // add more input in a single operator
  //Symbol(){};
  /*!
   * \brief construct a Symbol with SymbolHandle
   * \param handle the given SymbolHandle
   */
  explicit Symbol(SymbolHandle handle);
  /*!
   * \brief construct a variable Symbol
   * \param name the name of the variable
   */
  explicit Symbol(const std::string &name);
  // TODO(zhangcheng-qinyinghua)
  // implement all the operators
  Symbol operator+(const Symbol &rhs);
  Symbol operator-(const Symbol &rhs);
  Symbol operator*(const Symbol &rhs);
  Symbol operator/(const Symbol &rhs);
  Symbol Copy() const;
  /*!
   * \brief construct a variable Symbol
   * \param name the name of the variable
   */
  static Symbol Variable(const std::string &name = "");
  /*!
   * \return the SymbolHandle
   */
  SymbolHandle GetHandle() const { return blob_ptr_->handle_; }
  /*!
   * \brief construct an operator Symbol, with given input Symbol and config
   * \param name the name of the Symbol
   * \param input_keys the vector of keys of the input
   * \param input_values the vector of the intput Symbols
   * \param config_keys the vector of keys of the config
   * \param config_values the vecotr of values of the config
   */
  Symbol(const std::string &operator_name, const std::string &name,
         std::vector<const char *> input_keys,
         std::vector<SymbolHandle> input_values,
         std::vector<const char *> config_keys,
         std::vector<const char *> config_values);
  /*!
   * \brief infer the shapes by providing shapes of known argument shapes.
   * \param arg_shapes map of argument name to shape of arguments with known shapes.
   * \param in_shapes used to store infered shapes of input arguments.
   * \param out_shapes used to store infered shapes of outputs.
   * \param aux_shapes use to store the infered shapes of auxiliary states
   */
  void InferShape(
      const std::map<std::string, std::vector<mx_uint> > &arg_shapes,
      std::vector<std::vector<mx_uint> > *in_shape,
      std::vector<std::vector<mx_uint> > *aux_shape,
      std::vector<std::vector<mx_uint> > *out_shape) const;
  /*!
   * \brief List the arguments names.
   *
   * The position of the returned list also corresponds to calling position in operator()
   * \return the arguments list of this symbol, they can be either named or unnamed (empty string).
   */
  std::vector<std::string> ListArguments() const;
  /*! \return get the descriptions of outputs for this symbol */
  std::vector<std::string> ListOutputs() const;
  /*! \return get the descriptions of auxiliary data for this symbol */
  std::vector<std::string> ListAuxiliaryStates() const;
  /*!
   * \brief infer and construct all the arrays to bind to executor by providing some known arrays.
   * \param context the context of all the infered arrays
   * \param arg_arrays infered input arguments arrays.
   * \param arad_arrays infered arrays to store the gradient output of the input arguments.
   * \param aux_arrays infered arrays that is used as internal state in op.
   * \param args_map map of some given arguments arrays.
   * \param args_grad_store map of some gradient given store arrays.
   * \param args_req_type map of some given type of gradient saving. Can only be in {kNullOp, kAddTo, kWriteTo}.
   */
  void InferExecutorArrays(
      const Context &context, std::vector<NDArray> *arg_arrays,
      std::vector<NDArray> *grad_arrays, std::vector<OpReqType> *grad_reqs,
      std::vector<NDArray> *aux_arrays,
      const std::map<std::string, NDArray> &args_map,
      const std::map<std::string, NDArray> &arg_grad_store =
          std::map<std::string, NDArray>(),
      const std::map<std::string, OpReqType> &grad_req_type =
          std::map<std::string, OpReqType>()) const;
  /*!
   * \brief infer and construct all the input arguments arrays to bind to executor by providing some known arguments arrays.
   * \param context the context of all the infered arrays.
   * \param args_map map of all the infered input arguments arrays.
   * \param known_args map of some given arguments arrays.
   */
  void InferArgsMap(const Context &context,
                    std::map<std::string, NDArray> *args_map,
                    const std::map<std::string, NDArray> &known_args) const;
  /*!
   * \brief Create an executor by bind symbol with context and arguments.
   *  If user do not want to compute the gradients of i-th argument, grad_req_type[i] can be kNullOp.
   *  The input arrays in the given maps should have the same name with the input symbol.
   *  Only need some of the necessary arrays, and the other arrays can be infered automatically.
   *
   * \param context the context of binding.
   * \param args_map the NDArray that stores the input arguments to the symbol.
   * \param arg_grad_store NDArray that is used to store the gradient output of the input arguments.
   * \param grad_req_type requirment type of gradient saving. Can only be in {kNullOp, kAddTo, kWriteTo}.
   * \return a new executor, which need to be free manually.
   */
  Executor *SimpleBind(const Context &context,
                       const std::map<std::string, NDArray> &args_map,
                       const std::map<std::string, NDArray> &arg_grad_store =
                           std::map<std::string, NDArray>(),
                       const std::map<std::string, OpReqType> &grad_req_type =
                           std::map<std::string, OpReqType>());
  /*!
   * \brief Create an executor by bind symbol with context and arguments.
   *  If user do not want to compute the gradients of i-th argument, grad_req_type[i] can be kNullOp.
   *
   * \param context the context of binding.
   * \param arg_arrays the NDArray that stores the input arguments to the symbol.
   * \param grad_arrays NDArray that is used to store the gradient output of the input arguments.
   * \param grad_reqs requirment type of gradient saving. Can only be in {kNullOp, kAddTo, kWriteTo}.
   * \param aux_arrays NDArray that is used as internal state in op
   * \return a new executor, which need to be free manually.
   */
  Executor *Bind(const Context &context, const std::vector<NDArray> &arg_arrays,
                 const std::vector<NDArray> &grad_arrays,
                 const std::vector<OpReqType> &grad_reqs,
                 const std::vector<NDArray> &aux_arrays);

 private:
  std::shared_ptr<SymBlob> blob_ptr_;
};

/*!
 * \brief Operator interface
 */
class Operator {
 public:
  /*!
   * \brief Operator constructor
   * \param operator_name type of the operator
   */
  explicit Operator(const std::string &operator_name);
  Operator &operator=(const Operator &rhs);
  /*!
   * \brief set config parameters
   * \param name name of the config parameter
   * \param value value of the config parameter
   * \return reference of self
   */
  template <typename T>
  Operator &SetParam(const std::string &name, const T &value) {
    std::string value_str;
    std::stringstream ss;
    ss << value;
    ss >> value_str;

    params_[name] = value_str;
    return *this;
  }
  /*!
   * \brief add an input symbol
   * \param name name of the input symbol
   * \param symbol the input symbol
   * \return reference of self
   */
  Operator &SetInput(const std::string &name, Symbol symbol);
  /*!
   * \brief add an input symbol
   * \param symbol the input symbol
   */
  void PushInput(const Symbol &symbol) {
    input_values.push_back(symbol.GetHandle());
  }
  /*!
   * \brief add input symbols
   */
  template <class T, class... Args>
  void PushInput(const Symbol &symbol, const T &t, Args... args) {
    PushInput(symbol);
    PushInput(t, args...);
  }
  /*!
   * \brief add input symbols
   * \return reference of self
   */
  Operator &operator()() { return *this; }
  /*!
   * \brief add input symbols
   * \param symbol the input symbol
   * \return reference of self
   */
  Operator &operator()(const Symbol &symbol) {
    input_values.push_back(symbol.GetHandle());
    return *this;
  }
  /*!
   * \brief add a list of input symbols
   * \param symbols the vector of the input symbols
   * \return reference of self
   */
  Operator &operator()(const std::vector<Symbol> &symbols) {
    for (auto &s : symbols) {
      input_values.push_back(s.GetHandle());
    }
    return *this;
  }
  /*!
   * \brief add input symbols
   * \return reference of self
   */
  template <typename T, typename... Args>
  Operator &operator()(const Symbol &symbol, const T &t, Args... args) {
    PushInput(symbol, t, args...);
    return *this;
  }
  /*!
   * \brief create a Symbol from the current operator
   * \param name the name of the operator
   * \return the operator Symbol
   */
  Symbol CreateSymbol(const std::string &name = "");

 private:
  std::map<std::string, std::string> params_desc_;
  bool variable_params_ = false;
  std::map<std::string, std::string> params_;
  std::vector<SymbolHandle> input_values;
  std::vector<std::string> input_keys;
  AtomicSymbolCreator handle_;
};

/*!
 * \brief Executor interface
 */
class Executor {
 public:
  Executor(const Symbol &symbol, Context context,
           const std::vector<NDArray> &arg_arrays,
           const std::vector<NDArray> &grad_arrays,
           const std::vector<OpReqType> &grad_reqs,
           const std::vector<NDArray> &aux_arrays);
  explicit Executor(const ExecutorHandle &h) { handle_ = h; }
  /*!
   * \brief Perform a Forward operation of Operator
   *  After this operation, user can get the result by using function head.
   */
  void Forward(bool is_train) {
    MXExecutorForward(handle_, is_train ? 1 : 0);
    mx_uint out_size;
    NDArrayHandle *out_array;
    CHECK_EQ(MXExecutorOutputs(handle_, &out_size, &out_array), 0);
    for (mx_uint i = 0; i < out_size; ++i) {
      outputs[i] = NDArray(out_array[i]);
    }
  }
  /*!
   * \brief Perform a Backward operation of the Operator.
   *  This must be called after Forward.
   *  After this operation, NDArrays specified by grad_in_args_store will be updated accordingly.
   *  User is allowed to pass in an empty Array if the head node is
   *  loss function and head gradeitn is not needed.
   *
   * \param head_grads the gradient of head nodes to be backproped.
   */
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
  /*!
   * \brief update the arguments with given learning rate and optimizer
   * \param opt the pointer to the optimizer
   * \param lr learning rate
   * \param arg_update_begin begin index of the arguments to be updated, it starts after the input data by default 
   * \param arg_update_end end index of the arguments to be updated, it ends before the label data by default
   */
  void UpdateAll(Optimizer *opt, float lr, int arg_update_begin = 1,
                 int arg_update_end = -1);
  /*!
   * \brief destructor, free the handle
   */
  ~Executor() { MXExecutorFree(handle_); }
  std::vector<NDArray> arg_arrays;
  std::vector<NDArray> grad_arrays;
  std::vector<NDArray> aux_arrays;
  /*!
   * \brief arrays store the outputs of forward
   */
  std::vector<NDArray> outputs;

 private:
  Executor(const Executor &e);
  Executor &operator=(const Executor &e);
  ExecutorHandle handle_;
};

/*!
 * \brief Optimizer interface
 */
class Optimizer {
 public:
  /*!
   * \brief Operator constructor, the optimizer is not initialized until the first Update
   * \param opt_type type of the optimizer
   */
  explicit Optimizer(const std::string &opt_type);
  /*!
   * \brief destructor, free the handle
   */
  ~Optimizer() {
    if (init_) MXOptimizerFree(handle_);
  }
  /*!
   * \brief set config parameters
   * \param name name of the config parameter
   * \param value value of the config parameter
   * \return reference of self
   */
  template <typename T>
  Optimizer &SetParam(const std::string &name, const T &value) {
    std::string value_str;
    std::stringstream ss;
    ss << value;
    ss >> value_str;

    params_[name] = value_str;
    return *this;
  }
  /*!
   *  \brief Update a weight with gradient.
   *  \param index the unique index for the weight.
   *  \param weight the weight to update.
   *  \param grad gradient for the weight.
   *  \param lr learning rate for this update.
   */
  void Update(int index, NDArray weight, NDArray grad, real_t lr);
  // TODO(zhangcheng-qinyinghua)
  // implement Update a list of arrays, maybe in the form of map
  //void Update(int index, std::vector<NDArray> weights, std::vector<NDArray> grad, real_t lr);

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

class KVStore {
public:
  inline KVStore(bool IsLocal = true);
  inline void Push(int key, const NDArray& val, int priority = 0);
  inline void Push(const std::vector<int>& keys, const std::vector<NDArray>& val, int priority = 0);
  inline NDArray Pull(int key, int priority = 0);
  inline std::vector<NDArray> Pull(const std::vector<int>& keys, int priority = 0);
  inline void SetOptimizer(const Optimizer& optimizer);
  inline std::string GetType() const;
  inline int GetRank() const;
  inline int GetNumWorkers() const;

private:
  KVStoreHandle handle_;
};
}  // namespace cpp
}  // namespace mxnet

#include "MxNetCpp.hpp"
#include "executor.hpp"
#include "symbol.hpp"
#include "ndarray.hpp"

#endif  // MXNETCPP_H_
