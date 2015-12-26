#include <iostream>
#include <fstream>
#include <string>
#include <map>
using namespace std;

const std::string HEADER_FILE = "op.h";
const std::string SOURCE_FILE = "op.cpp";

std::map<std::string, std::string> typeMapping;

void InitTypeMapping() {
	typeMapping["integer"] = "int";
	typeMapping["float"] = "float";
	typeMapping["boolean"] = "bool";
	typeMapping["Symbol"] = "Symbol";
}

// parses an atomic symbol, generate corresponding program
void ParseAtomicSymbol(ostream & header, ostream & source) {
}

// parses a parameter type
void ParseParameterType (const char * type, const char * OpName, const char * paramName, 
	string & outType, string & outTypeDefinition) {
	// types: int, boolean, float, string, enum
	// could have default value
	// if it is a enum type, we will name the enum type as OpNameParamName
	// for primitive types, outType will be the string of that type
	// for enum, outType will be the enum type name, and outTypeDefinition will be the definition of the type
}



int main() {
	return 0;
}