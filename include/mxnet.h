#ifndef MXNETCPP_MXNET_H_
#define MXNETCPP_MXNET_H_

namespace mxnet {
namespace cpp {

/*!
* \brief Mxnet instance holds a map of all the symbol creators so we can
*  get symbol creators by name.
*  This is used internally by Symbol and Operator.
*/
class Mxnet {
public:
  /*!
  * \brief Create an Mxnet instance
  */
  inline Mxnet() {
    mx_uint num_symbol_creators = 0;
    AtomicSymbolCreator *symbol_creators = nullptr;
    int r =
      MXSymbolListAtomicSymbolCreators(&num_symbol_creators, &symbol_creators);
    CHECK_EQ(r, 0);
    for (mx_uint i = 0; i < num_symbol_creators; i++) {
      const char *name;
      const char *description;
      mx_uint num_args;
      const char **arg_names;
      const char **arg_type_infos;
      const char **arg_descriptions;
      const char *key_var_num_args;
      r = MXSymbolGetAtomicSymbolInfo(symbol_creators[i], &name, &description,
        &num_args, &arg_names, &arg_type_infos,
        &arg_descriptions, &key_var_num_args);
      CHECK_EQ(r, 0);
      symbol_creators_[name] = symbol_creators[i];
    }
  }

  /*!
  * \brief Get a symbol creator with its name.
  *
  * \param name name of the symbol creator
  * \return handle to the symbol creator
  */
  inline AtomicSymbolCreator GetSymbolCreator(const std::string &name) {
    return symbol_creators_[name];
  }
private:
  std::map<std::string, AtomicSymbolCreator> symbol_creators_;
};

}
}

#endif // MXNETCPP_MXNET_H_