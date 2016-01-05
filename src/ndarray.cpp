/*!
 * Copyright (c) 2015 by Contributors
 */
#include <vector>
#include "MxNetCpp.h"
namespace mxnet {
namespace cpp {
NDArray::NDArray() {
  NDArrayHandle handle;
  CHECK_EQ(MXNDArrayCreateNone(&handle), 0);
  blob_ptr_ = std::make_shared<NDBlob>(handle);
}
NDArray::NDArray(const NDArrayHandle &handle) {
  blob_ptr_ = std::make_shared<NDBlob>(handle);
}
NDArray::NDArray(const std::vector<mx_uint> &shape, const Context &context,
                 bool delay_alloc) {
  NDArrayHandle handle;
  CHECK_EQ(MXNDArrayCreate(shape.data(), shape.size(), context.GetDeviceType(),
                           context.GetDeviceId(), delay_alloc, &handle),
           0);
  blob_ptr_ = std::make_shared<NDBlob>(handle);
}
NDArray::NDArray(mshadow::TShape shape, const Context &context,
                 bool delay_alloc) {
  NDArrayHandle handle;
  CHECK_EQ(MXNDArrayCreate(shape.data(), shape.ndim(), context.GetDeviceType(),
                           context.GetDeviceId(), delay_alloc, &handle),
           0);
  blob_ptr_ = std::make_shared<NDBlob>(handle);
}
NDArray::NDArray(const mx_float *data, size_t size) {
  NDArrayHandle handle;
  CHECK_EQ(MXNDArrayCreateNone(&handle), 0);
  MXNDArraySyncCopyFromCPU(handle, data, size);
  blob_ptr_ = std::make_shared<NDBlob>(handle);
}
NDArray::NDArray(const std::vector<mx_float> &data) {
  NDArrayHandle handle;
  CHECK_EQ(MXNDArrayCreateNone(&handle), 0);
  MXNDArraySyncCopyFromCPU(handle, data.data(), data.size());
  blob_ptr_ = std::make_shared<NDBlob>(handle);
}

NDArray NDArray::operator+(real_t scalar) {
  NDArray ret;
  FunctionHandle func_handle;
  MXGetFunction("_plus_scalar", &func_handle);
  CHECK_EQ(MXFuncInvoke(func_handle, &blob_ptr_->handle_, &scalar,
                        &ret.blob_ptr_->handle_),
           0);
  return ret;
}
NDArray NDArray::operator-(real_t scalar) {
  NDArray ret;
  FunctionHandle func_handle;
  MXGetFunction("_minus_scalar", &func_handle);
  CHECK_EQ(MXFuncInvoke(func_handle, &blob_ptr_->handle_, &scalar,
                        &ret.blob_ptr_->handle_),
           0);
  return ret;
}
NDArray NDArray::operator*(real_t scalar) {
  NDArray ret;
  FunctionHandle func_handle;
  MXGetFunction("_mul_scalar", &func_handle);
  CHECK_EQ(MXFuncInvoke(func_handle, &blob_ptr_->handle_, &scalar,
                        &ret.blob_ptr_->handle_),
           0);
  return ret;
}
NDArray NDArray::operator/(real_t scalar) {
  NDArray ret;
  FunctionHandle func_handle;
  MXGetFunction("_div_scalar", &func_handle);
  CHECK_EQ(MXFuncInvoke(func_handle, &blob_ptr_->handle_, &scalar,
                        &ret.blob_ptr_->handle_),
           0);
  return ret;
}
NDArray NDArray::operator+(const NDArray &rhs) {
  NDArray ret;
  FunctionHandle func_handle;
  MXGetFunction("_plus", &func_handle);
  NDArrayHandle input_handle[2];
  input_handle[0] = blob_ptr_->handle_;
  input_handle[1] = rhs.blob_ptr_->handle_;
  CHECK_EQ(
      MXFuncInvoke(func_handle, input_handle, nullptr, &ret.blob_ptr_->handle_),
      0);
  return ret;
}
NDArray NDArray::operator-(const NDArray &rhs) {
  NDArray ret;
  FunctionHandle func_handle;
  MXGetFunction("_minus", &func_handle);
  NDArrayHandle input_handle[2];
  input_handle[0] = blob_ptr_->handle_;
  input_handle[1] = rhs.blob_ptr_->handle_;
  CHECK_EQ(
      MXFuncInvoke(func_handle, input_handle, nullptr, &ret.blob_ptr_->handle_),
      0);
  return ret;
}
NDArray NDArray::operator*(const NDArray &rhs) {
  NDArray ret;
  FunctionHandle func_handle;
  MXGetFunction("_mul", &func_handle);
  NDArrayHandle input_handle[2];
  input_handle[0] = blob_ptr_->handle_;
  input_handle[1] = rhs.blob_ptr_->handle_;
  CHECK_EQ(
      MXFuncInvoke(func_handle, input_handle, nullptr, &ret.blob_ptr_->handle_),
      0);
  return ret;
}
NDArray NDArray::operator/(const NDArray &rhs) {
  NDArray ret;
  FunctionHandle func_handle;
  MXGetFunction("_div", &func_handle);
  NDArrayHandle input_handle[2];
  input_handle[0] = blob_ptr_->handle_;
  input_handle[1] = rhs.blob_ptr_->handle_;
  CHECK_EQ(
      MXFuncInvoke(func_handle, input_handle, nullptr, &ret.blob_ptr_->handle_),
      0);
  return ret;
}
NDArray &NDArray::operator=(real_t scalar) {
  FunctionHandle func_handle;
  MXGetFunction("_set_value", &func_handle);
  CHECK_EQ(MXFuncInvoke(func_handle, nullptr, &scalar, &blob_ptr_->handle_), 0);
  return *this;
}
NDArray &NDArray::operator+=(real_t scalar) {
  FunctionHandle func_handle;
  MXGetFunction("_plus_scalar", &func_handle);
  CHECK_EQ(MXFuncInvoke(func_handle, &blob_ptr_->handle_, &scalar,
                        &blob_ptr_->handle_),
           0);
  return *this;
}
NDArray &NDArray::operator-=(real_t scalar) {
  FunctionHandle func_handle;
  MXGetFunction("_minus_scalar", &func_handle);
  CHECK_EQ(MXFuncInvoke(func_handle, &blob_ptr_->handle_, &scalar,
                        &blob_ptr_->handle_),
           0);
  return *this;
}
NDArray &NDArray::operator*=(real_t scalar) {
  FunctionHandle func_handle;
  MXGetFunction("_mul_scalar", &func_handle);
  CHECK_EQ(MXFuncInvoke(func_handle, &blob_ptr_->handle_, &scalar,
                        &blob_ptr_->handle_),
           0);
  return *this;
}
NDArray &NDArray::operator/=(real_t scalar) {
  FunctionHandle func_handle;
  MXGetFunction("_div_scalar", &func_handle);
  CHECK_EQ(MXFuncInvoke(func_handle, &blob_ptr_->handle_, &scalar,
                        &blob_ptr_->handle_),
           0);
  return *this;
}
NDArray &NDArray::operator+=(const NDArray &rhs) {
  FunctionHandle func_handle;
  MXGetFunction("_plus", &func_handle);
  NDArrayHandle input_handle[2];
  input_handle[0] = blob_ptr_->handle_;
  input_handle[1] = rhs.blob_ptr_->handle_;
  CHECK_EQ(
      MXFuncInvoke(func_handle, input_handle, nullptr, &blob_ptr_->handle_), 0);
  return *this;
}
NDArray &NDArray::operator-=(const NDArray &rhs) {
  FunctionHandle func_handle;
  MXGetFunction("_minus", &func_handle);
  NDArrayHandle input_handle[2];
  input_handle[0] = blob_ptr_->handle_;
  input_handle[1] = rhs.blob_ptr_->handle_;
  CHECK_EQ(
      MXFuncInvoke(func_handle, input_handle, nullptr, &blob_ptr_->handle_), 0);
  return *this;
}
NDArray &NDArray::operator*=(const NDArray &rhs) {
  FunctionHandle func_handle;
  MXGetFunction("_mul", &func_handle);
  NDArrayHandle input_handle[2];
  input_handle[0] = blob_ptr_->handle_;
  input_handle[1] = rhs.blob_ptr_->handle_;
  CHECK_EQ(
      MXFuncInvoke(func_handle, input_handle, nullptr, &blob_ptr_->handle_), 0);
  return *this;
}
NDArray &NDArray::operator/=(const NDArray &rhs) {
  FunctionHandle func_handle;
  MXGetFunction("_div", &func_handle);
  NDArrayHandle input_handle[2];
  input_handle[0] = blob_ptr_->handle_;
  input_handle[1] = rhs.blob_ptr_->handle_;
  CHECK_EQ(
      MXFuncInvoke(func_handle, input_handle, nullptr, &blob_ptr_->handle_), 0);
  return *this;
}

void NDArray::SyncCopyFromCPU(const mx_float *data, size_t size) {
  MXNDArraySyncCopyFromCPU(blob_ptr_->handle_, data, size);
}
void NDArray::SyncCopyFromCPU(const std::vector<mx_float> &data) {
  MXNDArraySyncCopyFromCPU(blob_ptr_->handle_, data.data(), data.size());
}
void NDArray::SyncCopyToCPU(mx_float *data, size_t size) {
  MXNDArraySyncCopyToCPU(blob_ptr_->handle_, data, size);
}
NDArray NDArray::Copy(const Context &ctx) const {
  NDArray ret(GetShape(), ctx);
  FunctionHandle func_handle;
  MXGetFunction("_copyto", &func_handle);
  CHECK_EQ(MXFuncInvoke(func_handle, &blob_ptr_->handle_, nullptr,
                        &ret.blob_ptr_->handle_),
           0);
  return ret;
}

NDArray NDArray::Slice(mx_uint begin, mx_uint end) const {
  NDArrayHandle handle;
  CHECK_EQ(MXNDArraySlice(GetHandle(), begin, end, &handle), 0);
  return NDArray(handle);
}

void NDArray::WaitToRead() {
  CHECK_EQ(MXNDArrayWaitToRead(blob_ptr_->handle_), 0);
}
void NDArray::WaitToWrite() {
  CHECK_EQ(MXNDArrayWaitToWrite(blob_ptr_->handle_), 0);
}
void NDArray::WaitAll() { CHECK_EQ(MXNDArrayWaitAll(), 0); }
void NDArray::SampleGaussian(real_t mu, real_t sigma, NDArray *out) {
  FunctionHandle func_handle;
  MXGetFunction("_random_gaussian", &func_handle);
  real_t scalar[2] = {mu, sigma};
  CHECK_EQ(MXFuncInvoke(func_handle, nullptr, scalar, &out->blob_ptr_->handle_),
           0);
}
void NDArray::SampleUniform(real_t begin, real_t end, NDArray *out) {
  FunctionHandle func_handle;
  MXGetFunction("_random_uniform", &func_handle);
  real_t scalar[2] = {begin, end};
  CHECK_EQ(MXFuncInvoke(func_handle, nullptr, scalar, &out->blob_ptr_->handle_),
           0);
}

std::vector<mx_uint> NDArray::GetShape() const {
  const mx_uint *out_pdata;
  mx_uint out_dim;
  MXNDArrayGetShape(blob_ptr_->handle_, &out_dim, &out_pdata);
  std::vector<mx_uint> ret;
  for (mx_uint i = 0; i < out_dim; ++i) {
    ret.push_back(out_pdata[i]);
  }
  return ret;
}
mx_float *NDArray::GetData() {
  mx_float *ret;
  MXNDArrayGetData(blob_ptr_->handle_, &ret);
  return ret;
}
Context NDArray::GetContext() const {
  int out_dev_type;
  int out_dev_id;
  MXNDArrayGetContext(blob_ptr_->handle_, &out_dev_type, &out_dev_id);
  return Context((DeviceType)out_dev_type, out_dev_id);
}
}  // namespace cpp
}  // namespace mxnet
