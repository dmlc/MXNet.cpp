/*!
 *  Copyright (c) 2016 by Contributors
 * \file kvstore.hpp
 * \brief implementation of kvstore
 * \author Xin Li
 */

#include "MxNetCpp.h"

#ifndef KVSTORE_HPP
#define KVSTORE_HPP

namespace mxnet {
namespace cpp {

KVStore::KVStore(const std::string& name) {
  CHECK_EQ(MXKVStoreCreate(name.c_str(), &handle_), 0);
}

void KVStore::Init(int key, const NDArray& val) {
  NDArrayHandle val_handle = val.GetHandle();
  CHECK_EQ(MXKVStoreInit(handle_, 1, &key, &val_handle), 0);
}

void KVStore::Init(const std::vector<int>& keys, const std::vector<NDArray>& vals) {
  CHECK_EQ(keys.size(), vals.size());
  std::vector<NDArrayHandle> val_handles(vals.size());
  std::transform(vals.cbegin(), vals.cend(), val_handles.begin(),
      [](const NDArray& val) {
        return val.GetHandle();
      });

  CHECK_EQ(MXKVStoreInit(handle_, keys.size(), keys.data(),
      val_handles.data()), 0);
}

void KVStore::Push(int key, const NDArray& val, int priority) {
  NDArrayHandle val_handle = val.GetHandle();
  CHECK_EQ(MXKVStorePush(handle_, 1, &key, &val_handle, priority), 0);
}

void KVStore::Push(const std::vector<int>& keys,
                   const std::vector<NDArray>& vals,
                   int priority) {
  CHECK_EQ(keys.size(), vals.size());
  std::vector<NDArrayHandle> val_handles(vals.size());
  std::transform(vals.cbegin(), vals.cend(), val_handles.begin(),
      [](const NDArray& val) {
        return val.GetHandle();
      });

  CHECK_EQ(MXKVStorePush(handle_, keys.size(), keys.data(),
      val_handles.data(), priority), 0);
}

void KVStore::Pull(int key, NDArray& out, int priority) {
  NDArrayHandle out_handle = out.GetHandle();
  CHECK_EQ(MXKVStorePull(handle_, 1, &key, &out_handle, priority), 0);
}

void KVStore::Pull(const std::vector<int>& keys, std::vector<NDArray>& outs, int priority) {
  CHECK_EQ(keys.size(), outs.size());

  std::vector<NDArrayHandle> out_handles(keys.size());
  std::transform(outs.cbegin(), outs.cend(), out_handles.begin(),
      [](const NDArray& val) {
        return val.GetHandle();
      });

  CHECK_EQ(MXKVStorePull(handle_, keys.size(), keys.data(),
      out_handles.data(), priority), 0);
}

namespace private_ {
  real_t learning_rate;

  extern "C"
  void updater(int key, NDArrayHandle recv, NDArrayHandle local,
      void* handle_) {
    Optimizer *opt = static_cast<Optimizer*>(handle_);
    opt->Update(key, NDArray(local), NDArray(recv), learning_rate);
  }
}

void KVStore::SetOptimizer(Optimizer& optimizer, real_t lr) {
  private_::learning_rate = lr;
  CHECK_EQ(MXKVStoreSetUpdater(handle_, &private_::updater, &optimizer), 0);
}

std::string KVStore::GetType() const {
  const char *type;
  CHECK_EQ(MXKVStoreGetType(handle_, &type), 0);
  // type is managed by handle_, no need to free its memory.
  return type;
}

int KVStore::GetRank() const {
  int rank;
  CHECK_EQ(MXKVStoreGetRank(handle_, &rank), 0);
  return rank;
}

int KVStore::GetNumWorkers() const {
  int num_workers;
  CHECK_EQ(MXKVStoreGetGroupSize(handle_, &num_workers), 0);
  return num_workers;
}

}  // namespace cpp
}  // namespace mxnet

#endif /* end of include guard: KVSTORE_HPP */