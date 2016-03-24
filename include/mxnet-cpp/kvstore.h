/*!
*  Copyright (c) 2016 by Contributors
* \file kvstore.h
* \brief definition of kvstore
* \author Chuntao Hong
*/

#ifndef MXNETCPP_KVSTORE_H
#define MXNETCPP_KVSTORE_H

#include <string>
#include <vector>
#include "mxnet-cpp/ndarray.h"

namespace mxnet {
namespace cpp {

class KVStore {
 public:
  explicit inline KVStore(const std::string& name = "local");
  inline void RunServer();
  inline void Init(int key, const NDArray& val);
  inline void Init(const std::vector<int>& keys, const std::vector<NDArray>& vals);
  inline void Push(int key, const NDArray& val, int priority = 0);
  inline void Push(const std::vector<int>& keys,
      const std::vector<NDArray>& vals, int priority = 0);
  inline void Pull(int key, NDArray* out, int priority = 0);
  inline void Pull(const std::vector<int>& keys, std::vector<NDArray>* outs, int priority = 0);
  // TODO(lx): put lr in optimizer or not?
  inline void SetOptimizer(std::unique_ptr<Optimizer> optimizer);
  inline std::string GetType() const;
  inline int GetRank() const;
  inline int GetNumWorkers() const;
  inline std::string GetRole() const;
  ~KVStore() { MXKVStoreFree(handle_); }

 private:
  KVStoreHandle handle_;
  std::unique_ptr<Optimizer> optimizer_;
};

}  // namespace cpp
}  // namespace mxnet

#endif  // MXNETCPP_KVSTORE_H
