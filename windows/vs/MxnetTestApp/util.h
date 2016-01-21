#pragma once

#include <string>
//----------------------------
// command line
inline bool CmdOptionExists(int argc, char ** argv, const std::string& option)
{
  return std::find(argv, argv + argc, option) != argv + argc;
}

template<class T>
bool GetCmdOption(int argc, char ** argv, const std::string& option, T& value)
{
  char ** op = std::find(argv, argv + argc, option);
  size_t pos = op - argv;
  if (pos == argc || pos + 1 == argc)
  {
    return false;
  }
  std::istringstream iss(argv[pos + 1]);
  T v;
  iss >> v;
  if (!iss.fail())
  {
    value = v;
    return true;
  }
  else
  {
    return false;
  }
}

class OptionParser
{
public:
  OptionParser(int argc_, char ** argv_) : argc(argc_), argv(argv_) {}

  template<typename T>
  void GetOption(const std::string & option, T & value)
  {
    GetCmdOption(argc, argv, option, value);
  }

  template<typename T>
  void GetOption(const std::string & option, T & value, std::function<void(void)> printHelp)
  {
    if (!GetCmdOption(argc, argv, option, value))
    {
      printHelp();
      exit(1);
    }
  }

  template<typename T>
  void GetOption(const std::string & option, T & value,
    const std::string & errMsg, std::function<void(void)> printHelp)
  {
    if (!GetCmdOption(argc, argv, option, value))
    {
      cout << errMsg << endl;
      printHelp();
      exit(1);
    }
  }
private:
  int argc;
  char ** argv;
};