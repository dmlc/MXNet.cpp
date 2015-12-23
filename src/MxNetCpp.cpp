#include "MxNetCpp.h"

namespace mxnet
{
  namespace cpp {

    Mxnet * MxNet = new Mxnet();

    template<typename K, typename V>
    inline bool HasKey(const std::map<K, V> & m, const K & key) {
      return m.find(key) != m.end();
    }

    Mxnet::Mxnet() {
      mx_uint num_symbol_creators = 0;
      AtomicSymbolCreator * symbol_creators = nullptr;
      int r = MXSymbolListAtomicSymbolCreators(&num_symbol_creators, &symbol_creators);
      CHECK(r == 0);
      for (mx_uint i = 0; i < num_symbol_creators; i++) {
        const char * name;
        const char * description;
        mx_uint num_args;
        const char ** arg_names;
        const char ** arg_type_infos;
        const char ** arg_descriptions;
        const char * key_var_num_args;
        r = MXSymbolGetAtomicSymbolInfo(symbol_creators[i],
          &name, &description, &num_args, &arg_names, &arg_type_infos,
          &arg_descriptions, &key_var_num_args);
        CHECK(r == 0);
        std::string symbolInfo;
        symbolInfo = symbolInfo + name + "(";
        for (mx_uint j = 0; j < num_args; j++) {
          if (j != 0) {
            symbolInfo += " ,";
          }
          symbolInfo += arg_names[j];
        }
        symbolInfo += ")";
        LOG(INFO) << symbolInfo;

      }
    }

    SymbolCreator Mxnet::GetSymbolCreator(const std::string & name) {

    }

  }
}