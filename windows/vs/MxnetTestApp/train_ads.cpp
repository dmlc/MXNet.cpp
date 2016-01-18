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
#include "data.h"
using namespace std;
using namespace mxnet::cpp;

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
    int batchSize = 1;
    int numWorkers = 1;
    int maxEpoch = 100000;
    int sampleSize = 601;
    float learning_rate = 0.1;

    mlp.InferArgsMap(ctx_dev, &args_map, args_map);
    Optimizer opt("ccsgd");
    opt.SetParam("momentum", 0.9)
      .SetParam("wd", 1e-4)
      .SetParam("rescale_grad", 1.0 / (numWorkers * batchSize))
      .SetParam("clip_gradient", 10);

    for (int ITER = 0; ITER < maxEpoch; ++ITER) {
      DataReader dataReader("./v.bin", sampleSize, batchSize);
      while (!dataReader.Eof()) {
        // read data in
        auto r = dataReader.ReadBatch();
        size_t nSamples = r.size() / sampleSize;
        CHECK(!r.empty());
        vector<float> data_vec, label_vec;
        for (int i = 0; i < nSamples; i++) {
          float * rp = r.data() + sampleSize * i;
          label_vec.push_back(*rp);
          data_vec.insert(data_vec.end(), rp + 1, rp + sampleSize);
        }
        r.clear();
        r.shrink_to_fit();

        const float *dptr = data_vec.data();
        const float *lptr = label_vec.data();
        NDArray dataArray = NDArray(mshadow::Shape2(nSamples, sampleSize - 1), 
          ctx_cpu, false);
        NDArray labelArray =
          NDArray(mshadow::Shape1(nSamples), ctx_cpu, false); 
        dataArray.SyncCopyFromCPU(dptr, nSamples * (sampleSize - 1));
        labelArray.SyncCopyFromCPU(lptr, nSamples);

        args_map["data"] = dataArray;
        args_map["data_label"] = labelArray;
        Executor *exe = mlp.SimpleBind(ctx_dev, args_map);
        exe->Forward(true);
        exe->Backward();
        exe->UpdateAll(&opt, learning_rate);
        delete exe;
      }

      LG << "Iter " << ITER
        << ", accuracy: " << ValAccuracy(batchSize * 10, mlp);
    }
  }

private:
  Context ctx_cpu;
  Context ctx_dev;
  map<string, NDArray> args_map;

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
  return 0;
}
