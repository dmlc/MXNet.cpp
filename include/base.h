#ifndef MXNETCPP_BASE_H
#define MXNETCPP_BASE_H

#include "c_api.h"

namespace mxnet {
namespace cpp {
  
typedef unsigned index_t;

enum OpReqType {
  /*! \brief no operation, do not write anything */
  kNullOp,
  /*! \brief write gradient to provided space */
  kWriteTo,
  /*!
  * \brief perform an inplace write,
  * Target shares memory with one of input arguments.
  * This option only happen when
  */
  kWriteInplace,
  /*! \brief add to the provided space */
  kAddTo
};

}
}

#endif // MXNETCPP_BASE_H