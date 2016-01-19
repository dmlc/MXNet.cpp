#ifndef MXNETCPP_KVSTORE_H
#define MXNETCPP_KVSTORE_H

#include <string>
#include "ndarray.hpp"
#include "mxnet/c_api.h"

namespace mxnet {
namespace cpp {

class KVStore {
public:
  inline KVStore(bool IsLocal = true) {
  }

  inline void Push(int key, const NDArray& val);
  inline void Pull(int key, NDArray& val);
  inline void SetOptimizer();
  inline std::string GetType();
  inline int GetNumWorkers();

private:
  const char* TYPE_STRINGS[] = {"dist", "local"};
};


}
}


#endif // MXNETCPP_KVSTORE_H