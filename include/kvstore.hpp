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

KVStore::KVStore(bool IsLocal) {
  CHECK_EQ(MXKVStoreCreate(IsLocal ? "local" : "dist", &handle_), 0);
}

void KVStore::Push(int key, const NDArray& val, int priority) {
  NDArrayHandle val_handle = val.GetHandle();
  CHECK_EQ(MXKVStorePush(handle_, 1, &key, &val_handle, priority), 0);
}

void KVStore::Push(const std::vector<int>& keys,
                   const std::vector<NDArray>& vals,
                   int priority) {
  CHECK_EQ(keys.size(), vals.size());
  std::vector<NDArrayHandle> val_handles;
  std::transform(vals.cbegin(), vals.cend(), val_handles.begin(),
      [](const NDArray& val) {
        return val.GetHandle();
      });

  CHECK_EQ(MXKVStorePush(handle_, keys.size(), keys.data(),
      val_handles.data(), priority), 0);
}

NDArray KVStore::Pull(int key, int priority) {
  NDArray out;
  NDArrayHandle out_handle = out.GetHandle();
  CHECK_EQ(MXKVStorePull(handle_, 1, &key, &out_handle, priority), 0);
  return std::move(out);
}

std::vector<NDArray> KVStore::Pull(const std::vector<int>& keys,
                                    int priority) {
  std::vector<NDArray> out(keys.size());
  std::vector<NDArrayHandle> out_handles;
  std::transform(out.cbegin(), out.cend(), out_handles.begin(),
      [](const NDArray& val) {
        return val.GetHandle();
      });

  CHECK_EQ(MXKVStorePull(handle_, keys.size(), keys.data(),
      out_handles.data(), priority), 0);

  return std::move(out);
}

void KVStore::SetOptimizer(const Optimizer& optimizer) {
  // TODO
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