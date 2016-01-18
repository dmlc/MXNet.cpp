/*!
* Copyright (c) 2015 by Contributors
*/
#include <condition_variable>
#include <iostream>
#include <fstream>
#include <map>
#include <mutex>
#include <string>
#include <vector>
#include <thread>

#include "MxNetCpp.h"
#include "MxNetOp.h"
#include "util.h"
using namespace std;
using namespace mxnet::cpp;

class DataReader {
public:
  DataReader(const std::string & dataDir,
    int recordSize,
    int batchSize)
    : dataDir_(dataDir),
    recordSize_(recordSize),
    batchSize_(batchSize),
    reset_(false),
    eof_(false),
    exit_(false) {

  }
  ~DataReader() {
    lock_guard<mutex> l(mutex_);
    exit_ = true;
  }

  bool Eof() {
    lock_guard<mutex> l(mutex_);
    return eof_;
  }

  void Reset() {
    lock_guard<mutex> l(mutex_);
    reset_ = true;
    if (!buffer_.empty()) buffer_.clear();
    condEmpty_.notify_one();
  }

  vector<float> ReadBatch() {
    unique_lock<mutex> l(mutex_);
    vector<float> r;
    if (eof_) return r;
    while (buffer_.empty()) {
      condReady_.wait(l);
    }
    r.swap(buffer_);
    condEmpty_.notify_one();
    return r;
  }

private:
  void IOThread() {
    unique_lock<mutex> l(mutex_);
    while (!exit_) {
      ifstream in(dataDir_ + "/v.bin", ios::binary);
      eof_ = false;
      reset_ = false;
      while (in.good()) {
        while (!buffer_.empty()) {
          if (reset_ || exit_) break;
          condEmpty_.wait(l);
        }
        if (reset_ || exit_) break;
        buffer_.resize(recordSize_ * batchSize_);
        size_t bytesToRead = recordSize_ * sizeof(float) * batchSize_;
        in.read((char*)&buffer_[0], bytesToRead);
        size_t bytesRead = in.gcount();
        CHECK_EQ(bytesRead % (sizeof(float)*recordSize_), 0);
        buffer_.resize(bytesRead / (sizeof(float) * recordSize_));
        condReady_.notify_one();
      }
      eof_ = true;
      while (!exit_ && !reset_) condEmpty_.wait(l);
    }
  }

  std::mutex mutex_;
  std::condition_variable condReady_;
  std::condition_variable condEmpty_;
  vector<float> buffer_;
  bool reset_;
  bool eof_;
  bool exit_;
  const int recordSize_;
  const int batchSize_;
  const std::string dataDir_;
  const std::thread ioThread_;
};


class Lenet {
public:
  Lenet()
    : ctx_cpu(Context(DeviceType::kCPU, 0)),
    ctx_dev(Context(DeviceType::kCPU, 0)) {}
  void Run() {
    /*define the symbolic net*/
    auto sym_x = Symbol::Variable("x");
    auto sym_label = Symbol::Variable("label");
    auto w1 = Symbol::Variable("w1");
    auto b1 = Symbol::Variable("b1");
    auto w2 = Symbol::Variable("w2");
    auto b2 = Symbol::Variable("b2");
    auto w3 = Symbol::Variable("w3");
    auto b3 = Symbol::Variable("b3");
    
    auto fc1 = FullyConnected("fc1", sym_x, w1, b1, 2048);
    auto act1 = Activation("act1", fc1, ActivationActType::relu);
    auto fc2 = FullyConnected("fc2", act1, w2, b2, 512);
    auto act2 = Activation("act2", fc2, ActivationActType::relu);
    auto fc3 = FullyConnected("fc3", act2, w3, b3, 1);
    auto mlp = LogisticRegressionOutput("softmax", fc3, sym_label);

    for (auto s : mlp.ListArguments()) {
      LG << s;
    }

    /*setup basic configs*/
    int val_fold = 1;
    int W = 28;
    int H = 28;
    int batch_size = 42;
    int max_epoch = 100000;
    float learning_rate = 1e-4;

    /*prepare the data*/
    vector<float> data_vec, label_vec;
    size_t data_count = GetData(&data_vec, &label_vec);
    const float *dptr = data_vec.data();
    const float *lptr = label_vec.data();
    NDArray data_array = NDArray(mshadow::Shape4(data_count, 1, W, H), ctx_cpu,
      false);  // store in main memory, and copy to
    // device memory while training
    NDArray label_array =
      NDArray(mshadow::Shape1(data_count), ctx_cpu,
      false);  // it's also ok if just store them all in device memory
    data_array.SyncCopyFromCPU(dptr, data_count * W * H);
    label_array.SyncCopyFromCPU(lptr, data_count);
    data_array.WaitToRead();
    label_array.WaitToRead();

    size_t train_num = data_count * (1 - val_fold / 10.0);
    train_data = data_array.Slice(0, train_num);
    train_label = label_array.Slice(0, train_num);
    val_data = data_array.Slice(train_num, data_count);
    val_label = label_array.Slice(train_num, data_count);

    LG << "here read fin";

    /*init some of the args*/
    // map<string, NDArray> args_map;
    args_map["data"] =
      NDArray(mshadow::Shape4(batch_size, 1, W, H), ctx_dev, false);
    args_map["data"] = data_array.Slice(0, batch_size).Copy(ctx_dev);
    args_map["data_label"] = label_array.Slice(0, batch_size).Copy(ctx_dev);
    NDArray::WaitAll();

    LG << "here slice fin";
    /*
    * we can also feed in some of the args other than the input all by
    * ourselves,
    * fc2-w , fc1-b for example:
    * */
    // args_map["fc2_w"] =
    // NDArray(mshadow::Shape2(500, 4 * 4 * 50), ctx_dev, false);
    // NDArray::SampleGaussian(0, 1, &args_map["fc2_w"]);
    // args_map["fc1_b"] = NDArray(mshadow::Shape1(10), ctx_dev, false);
    // args_map["fc1_b"] = 0;

    mlp.InferArgsMap(ctx_dev, &args_map, args_map);
    Optimizer opt("ccsgd");
    opt.SetParam("momentum", 0.9)
      .SetParam("wd", 1e-4)
      .SetParam("rescale_grad", 1.0)
      .SetParam("clip_gradient", 10);

    for (int ITER = 0; ITER < max_epoch; ++ITER) {
      size_t start_index = 0;
      while (start_index < train_num) {
        if (start_index + batch_size > train_num) {
          start_index = train_num - batch_size;
        }
        args_map["data"] =
          train_data.Slice(start_index, start_index + batch_size)
          .Copy(ctx_dev);
        args_map["data_label"] =
          train_label.Slice(start_index, start_index + batch_size)
          .Copy(ctx_dev);
        start_index += batch_size;
        NDArray::WaitAll();

        Executor *exe = mlp.SimpleBind(ctx_dev, args_map);

        exe->Forward(true);
        NDArray::WaitAll();

        exe->Backward();
        NDArray::WaitAll();

        exe->UpdateAll(&opt, learning_rate);
        NDArray::WaitAll();

        delete exe;
      }

      LG << "Iter " << ITER
        << ", accuracy: " << ValAccuracy(batch_size * 10, mlp);
    }
  }

private:
  Context ctx_cpu;
  Context ctx_dev;
  map<string, NDArray> args_map;
  NDArray train_data;
  NDArray train_label;
  NDArray val_data;
  NDArray val_label;

  size_t GetData(vector<float> *data, vector<float> *label) {
    const char *train_data_path = "./train.csv";
    ifstream inf(train_data_path);
    string line;
    inf >> line;  // ignore the header
    size_t _N = 0;
    while (inf >> line) {
      for (auto &c : line) c = (c == ',') ? ' ' : c;
      stringstream ss;
      ss << line;
      float _data;
      ss >> _data;
      label->push_back(_data);
      while (ss >> _data) data->push_back(_data / 256.0);
      _N++;
    }
    inf.close();
    return _N;
  }

  float ValAccuracy(int batch_size, Symbol lenet) {
    size_t val_num = val_data.GetShape()[0];

    size_t correct_count = 0;
    size_t all_count = 0;

    size_t start_index = 0;
    while (start_index < val_num) {
      if (start_index + batch_size > val_num) {
        start_index = val_num - batch_size;
      }
      args_map["data"] =
        val_data.Slice(start_index, start_index + batch_size).Copy(ctx_dev);
      args_map["data_label"] =
        val_label.Slice(start_index, start_index + batch_size).Copy(ctx_dev);
      start_index += batch_size;
      NDArray::WaitAll();

      Executor *exe = lenet.SimpleBind(ctx_dev, args_map);

      exe->Forward(false);
      NDArray::WaitAll();

      const auto &out = exe->outputs;
      NDArray out_cpu = out[0].Copy(ctx_cpu);
      NDArray label_cpu =
        val_label.Slice(start_index - batch_size, start_index).Copy(ctx_cpu);

      NDArray::WaitAll();

      mxnet::real_t *dptr_out = out_cpu.GetData();
      mxnet::real_t *dptr_label = label_cpu.GetData();
      for (int i = 0; i < batch_size; ++i) {
        float label = dptr_label[i];
        int cat_num = out_cpu.GetShape()[1];
        float p_label = 0, max_p = dptr_out[i * cat_num];
        for (int j = 0; j < cat_num; ++j) {
          float p = dptr_out[i * cat_num + j];
          if (max_p < p) {
            p_label = j;
            max_p = p;
          }
        }
        if (label == p_label) correct_count++;
      }
      all_count += batch_size;

      delete exe;
    }
    return correct_count * 1.0 / all_count;
  }
};

int main(int argc, char const *argv[]) {
  //Lenet lenet;
  //lenet.Run();

  DataReader r(".", 1, 1);

  return 0;
}
