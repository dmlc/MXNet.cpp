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
#include "util.h"
#include "data.h"
using namespace std;
using namespace mxnet::cpp;

class Mlp {
public:
  Mlp()
    : ctx_cpu(Context(DeviceType::kCPU, 0)),
    ctx_dev(Context(DeviceType::kCPU, 0)) {}
  void Run() {
    /*define the symbolic net*/
    auto sym_x = Symbol::Variable("data");
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

    NDArray w1m(Shape(2048, 600), ctx_cpu), 
      w2m(Shape(512, 2048), ctx_cpu), 
      w3m(Shape(1, 512), ctx_cpu);
    NDArray::SampleGaussian(0, 1, &w1m);
    NDArray::SampleGaussian(0, 1, &w2m);
    NDArray::SampleGaussian(0, 1, &w3m);
    NDArray b1m(Shape(2048), ctx_cpu),
      b2m(Shape(512), ctx_cpu),
      b3m(Shape(1), ctx_cpu);
    NDArray::SampleGaussian(0, 1, &b1m);
    NDArray::SampleGaussian(0, 1, &b2m);
    NDArray::SampleGaussian(0, 1, &b3m);

    for (auto s : mlp.ListArguments()) {
      LG << s;
    }  

    double samplesProcessed = 0;
    double sTime = get_time();

    /*setup basic configs*/
    Optimizer opt("ccsgd");
    opt.SetParam("momentum", 0.9)
      .SetParam("wd", 0.00001)
      .SetParam("rescale_grad", 1.0 / (numWorkers * batchSize));
      //.SetParam("clip_gradient", 10);
    const int nMiniBatches = 1;
    for (int ITER = 0; ITER < maxEpoch; ++ITER) {
      DataReader dataReader("f:/chhong/data/adsdnn/v.bin", sampleSize, batchSize);
      NDArray testData, testLabel;
      int mb = 0;
      while (!dataReader.Eof()) {
        //if (mb++ >= nMiniBatches) break;
        // read data in
        auto r = dataReader.ReadBatch();
        size_t nSamples = r.size() / sampleSize;
        samplesProcessed += nSamples;
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
        NDArray dataArray = NDArray(Shape(nSamples, sampleSize - 1),
          ctx_cpu, false);
        NDArray labelArray =
          NDArray(Shape(nSamples), ctx_cpu, false);
        dataArray.SyncCopyFromCPU(dptr, nSamples * (sampleSize - 1));
        labelArray.SyncCopyFromCPU(lptr, nSamples);
        args_map["data"] = dataArray;
        args_map["label"] = labelArray;
        args_map["w1"] = w1m;
        args_map["b1"] = b1m;
        args_map["w2"] = w2m;
        args_map["b2"] = b2m;
        args_map["w3"] = w3m;
        args_map["b3"] = b3m;
        Executor *exe = mlp.SimpleBind(ctx_dev, args_map);
        exe->Forward(true);
        NDArray::WaitAll();
        LG << "Iter " << ITER
          << ", accuracy: " << Auc(exe->outputs[0], labelArray)
          << "\t sample/s: " << samplesProcessed / (get_time() - sTime);
        exe->Backward();
        exe->UpdateAll(&opt, learning_rate);     
        NDArray::WaitAll();
        delete exe;
      }

      //LG << "Iter " << ITER
      //  << ", accuracy: " << ValAccuracy(mlp, testData, testLabel);
    }
  }

private:
  Context ctx_cpu;
  Context ctx_dev;
  map<string, NDArray> args_map;
  const static int batchSize = 300;
  const static int sampleSize = 601;
  const static int numWorkers = 1;
  const static int maxEpoch = 100000;
  float learning_rate = 0.01;

  float ValAccuracy(Symbol mlp, 
    const NDArray& samples, 
    const NDArray& labels) {
    size_t nSamples = samples.GetShape()[0];
    size_t nCorrect = 0;
    size_t startIndex = 0;
    args_map["data"] = samples;
    args_map["label"] = labels;

    Executor *exe = mlp.SimpleBind(ctx_dev, args_map);
    exe->Forward(false);
    const auto &out = exe->outputs;
    NDArray result = out[0].Copy(ctx_cpu);
    result.WaitToRead();
    const mx_float *pResult = result.GetData();
    const mx_float *pLabel = labels.GetData();
    for (int i = 0; i < nSamples; ++i) {
      float label = pLabel[i];
      int cat_num = result.GetShape()[1];
      float p_label = 0, max_p = pResult[i * cat_num];
      for (int j = 0; j < cat_num; ++j) {
        float p = pResult[i * cat_num + j];
        if (max_p < p) {
          p_label = j;
          max_p = p;
        }
      }
      if (label == p_label) nCorrect++;
    }
    delete exe;
    
    return nCorrect * 1.0 / nSamples;
  }
  
  float Auc(const NDArray& result, const NDArray& labels) {
    result.WaitToRead();
    const mx_float *pResult = result.GetData();
    const mx_float *pLabel = labels.GetData();
    int nSamples = labels.GetShape()[0];
    size_t nCorrect = 0;
    for (int i = 0; i < nSamples; ++i) {
      float label = pLabel[i];
      float p_label = pResult[i];
      if (label == (p_label >= 0.5)) nCorrect++;
    }
    return nCorrect * 1.0 / nSamples;
  }

};

int main(int argc, char const *argv[]) {
  Mlp mlp;
  mlp.Run();
  return 0;
}