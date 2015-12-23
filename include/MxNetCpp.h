#include <map>
#include <string>

#include <dmlc/logging.h>
#include <mxnet/c_api.h>
#include <mxnet/base.h>
#include <mxnet/operator.h>

namespace mxnet
{
  namespace cpp {
    using OpReqType = mxnet::OpReqType;;
    using DeviceType = mxnet::Context::DeviceType;
    class Mxnet;
    Mxnet * Mxnet_Instance;

    class Symbol;
    class Exeuctor;
    class Context {
    public:
      Context(const DeviceType & type, int id);
    private:
      DeviceType type_;
      int id_;
    };

    class NDArray {
    public:
      NDArray();
      NDArray(const std::vector<mx_uint> & shape, 
        const Context & context, 
        bool delay_alloc = true);
      NDArray(const mx_float * data, size_t size);
      void CopyData(mx_float * data, size_t size);
      void WaitToRead();
      void WaitToWrite();
      void WaitAll();
      std::vector<mx_uint> & GetShape();
      mx_float * GetData();
      Context GetContext();
    private:
      NDArrayHandle handle_;
    };

    class Operator {
    public:
      Operator(const std::string & name);
      Operator & operator = (const Operator & rhs);
      Operator & SetParam(const std::string & name, const std::string & value);
      std::string & operator[](const std::string & param_name);
      Symbol CreateSymbol();
    private:
      std::map<std::string, std::string> params_desc_;
      bool variable_params_ = false;
      std::map<std::string, std::string> params_;
      AtomicSymbolCreator handle_;
    };

    class Symbol {
    public:
      Symbol(const std::string & name);
      Symbol(const Symbol & rhs);
      Symbol & operator = (const Symbol & rhs);
      ~Symbol();
      Executor Bind(Context context,
        std::vector<NDArray> & in_args,
        std::vector<NDArray> & arg_grad_store,
        std::vector<OpReqType> & grad_req_type,
        std::vector<NDArray> & aux_states);
    private:
      SymbolHandle handle_;
    };

    class Executor {
    public:
      Executor(const Symbol & symbol,
        Context context,
        std::vector<NDArray> & in_args,
        std::vector<NDArray> & arg_grad_store,
        std::vector<OpReqType> & grad_req_type,
        std::vector<NDArray> & aux_states);
      Executor(const ExecutorHandle & h);
      void Forward();
      void Backward();
      ~Executor();
    private:
      Executor(const Executor & e) {}
      Executor & operator = (const Executor & e) {}
      ExecutorHandle handle_;
    };

    class Mxnet {
    public:
      Mxnet();
      Operator GetSOperator(const std::string & name);
    private:
      std::map<std::string, AtomicSymbolCreator> symbol_creators_;
    };

  }
}