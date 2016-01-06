#include <iostream>
#include <fstream>
#include <string>
#include <map>
#include <set>
using namespace std;

#include <dmlc/base.h>
#include <mxnet/c_api.h>
#include <dmlc/logging.h>
#include "util.h"

const std::string HEADER_FILE = "op.h";
const std::string SOURCE_FILE = "op.cpp";

std::map<std::string, std::string> typeMapping;

void InitTypeMapping() {
	typeMapping["integer"] = "int";
	typeMapping["float"] = "mx_float";
	typeMapping["boolean"] = "bool";
	typeMapping["Symbol"] = "Symbol";
}

// parses an atomic symbol, generate corresponding program
void ParseAtomicSymbol(const char *name, 
                      const char **arg_type_infos,
                      const char **arg_descriptions,
                      const char *key_var_num_args, 
                      ostream & header, ostream & source) {
  // parse arguments
  // now start generating code
  // comments for this operator
  // operator declaration
  // operator implementation
}

// parses a parameter type
void ParseParameterType (const string & type, const char * OpName, const char * paramName, 
	string & outType, string & outTypeDefinition) {
	// types: int, boolean, float, string, enum
	// could have default value
	// if it is a enum type, we will name the enum type as OpNameParamName
	// for primitive types, outType will be the string of that type
	// for enum, outType will be the enum type name, and outTypeDefinition will be the definition of the type
}

int main() {
  mx_uint num_symbol_creators = 0;
  AtomicSymbolCreator *symbol_creators = nullptr;
  int r =
    MXSymbolListAtomicSymbolCreators(&num_symbol_creators, &symbol_creators);

  set<std::string> allArgTypes;

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
    std::string symbolInfo;
    symbolInfo = symbolInfo + name + "(";
    for (mx_uint j = 0; j < num_args; j++) {
      if (j != 0) {
        symbolInfo += ", ";
      }
      string t = arg_type_infos[j];
      allArgTypes.insert(t);
      symbolInfo += string() + "T=[" + t + "] name=[" + arg_names[j]
        + "] desc=[" + arg_descriptions[j] + "]";
    }
    if (key_var_num_args != nullptr) {
      symbolInfo += string() + ", " + key_var_num_args;
    }
    symbolInfo += ")";
    LOG(INFO) << symbolInfo;
  }

  LOG(INFO) << "\nAllArgTypes: ";
  for (auto t : allArgTypes) {
    LOG(INFO) << t;
  }

	return 0;
}