/*!
 * Copyright (c) 2015 by Contributors
 */

#include <iostream>
#include <vector>
#include <mxnet/ndarray.h>
#include "MxNetCpp.h"
#include "MxNetOp.h"

using namespace std;
using namespace mxnet::cpp;

/*
 * In this example,
 * we make by hand some data in 10 classes with some pattern
 * and try to use MLP to recognize the pattern.
 */

void OutputAccuracy(mxnet::real_t* pred, mxnet::real_t* target) {
  int right = 0;
  for (int i = 0; i < 128; ++i) {
    float mx_p = pred[i * 10 + 0];
    float p_y = 0;
    for (int j = 0; j < 10; ++j) {
      if (pred[i * 10 + j] > mx_p) {
        mx_p = pred[i * 10 + j];
        p_y = j;
      }
    }
    if (p_y == target[i]) right++;
  }
  cout << "Accuracy: " << right / 128.0 << endl;
}

void MLP() {
  auto sym_x = Symbol::Variable("X");
  auto sym_w1 = Symbol::Variable("W1");
  auto sym_b1 = Symbol::Variable("B1");
  auto sym_w2 = Symbol::Variable("W2");
  auto sym_b2 = Symbol::Variable("B2");
  auto sym_label = Symbol::Variable("label");

  auto sym_fc_1 = FullyConnected("fc1", sym_x, sym_w1, sym_b1, 512);
  auto sym_act_1 = LeakyReLU("act_1", sym_fc_1, LeakyReLUActType::leaky);
  auto sym_fc_2 = FullyConnected("fc2", sym_act_1, sym_w2, sym_b2, 10);
  auto sym_act_2 = LeakyReLU("act_2", sym_fc_2, LeakyReLUActType::leaky);
  auto sym_out = SoftmaxOutput("softmax", sym_act_2, sym_label);

  Context ctx_dev(DeviceType::kCPU, 0);

  NDArray array_x(mshadow::Shape2(128, 28), ctx_dev, false);
  NDArray array_y(mshadow::Shape1(128), ctx_dev, false);

  mxnet::real_t* aptr_x = new mxnet::real_t[128 * 28];
  mxnet::real_t* aptr_y = new mxnet::real_t[128];

  // we make the data by hand, in 10 classes, with some pattern
  for (int i = 0; i < 128; i++) {
    for (int j = 0; j < 28; j++) {
      aptr_x[i * 28 + j] = i % 10 * 1.0f;
    }
    aptr_y[i] = i % 10;
  }
  array_x.SyncCopyFromCPU(aptr_x, 128 * 28);
  array_x.WaitToRead();
  array_y.SyncCopyFromCPU(aptr_y, 128);
  array_y.WaitToRead();

  // init the parameters
  NDArray array_w_1(mshadow::Shape2(512, 28), ctx_dev, false);
  NDArray array_b_1(mshadow::Shape1(512), ctx_dev, false);
  NDArray array_w_2(mshadow::Shape2(10, 512), ctx_dev, false);
  NDArray array_b_2(mshadow::Shape1(10), ctx_dev, false);

  // the parameters should be initialized in some kind of distribution,
  // so it learns fast
  // but here just give a const value by hand
  array_w_1 = 0.5f;
  array_b_1 = 0.0f;
  array_w_2 = 0.5f;
  array_b_2 = 0.0f;

  // the grads
  NDArray array_w_1_g(mshadow::Shape2(512, 28), ctx_dev, false);
  NDArray array_b_1_g(mshadow::Shape1(512), ctx_dev, false);
  NDArray array_w_2_g(mshadow::Shape2(10, 512), ctx_dev, false);
  NDArray array_b_2_g(mshadow::Shape1(10), ctx_dev, false);

  // Bind the symolic network with the ndarray
  // all the input args
  std::vector<NDArray> in_args;
  in_args.push_back(array_x);
  in_args.push_back(array_w_1);
  in_args.push_back(array_b_1);
  in_args.push_back(array_w_2);
  in_args.push_back(array_b_2);
  in_args.push_back(array_y);
  // all the grads
  std::vector<NDArray> arg_grad_store;
  arg_grad_store.push_back(NDArray());  // we don't need the grad of the input
  arg_grad_store.push_back(array_w_1_g);
  arg_grad_store.push_back(array_b_1_g);
  arg_grad_store.push_back(array_w_2_g);
  arg_grad_store.push_back(array_b_2_g);
  arg_grad_store.push_back(
      NDArray());  // neither do we need the grad of the loss
  // how to handle the grad
  std::vector<mxnet::OpReqType> grad_req_type;
  grad_req_type.push_back(mxnet::kNullOp);
  grad_req_type.push_back(mxnet::kWriteTo);
  grad_req_type.push_back(mxnet::kWriteTo);
  grad_req_type.push_back(mxnet::kWriteTo);
  grad_req_type.push_back(mxnet::kWriteTo);
  grad_req_type.push_back(mxnet::kNullOp);
  std::vector<NDArray> aux_states;

  cout << "make the Executor" << endl;
  Executor* exe = new Executor(sym_out, ctx_dev, in_args, arg_grad_store,
                               grad_req_type, aux_states);

  cout << "Training" << endl;
  int max_iters = 20000;
  mxnet::real_t learning_rate = 0.0001;
  for (int iter = 0; iter < max_iters; ++iter) {
    exe->Forward(true);

    if (iter % 100 == 0) {
      cout << "epoch " << iter << endl;
      std::vector<NDArray>& out = exe->outputs;
      float* cptr = new float[128 * 10];
      out[0].SyncCopyToCPU(cptr, 128 * 10);
      NDArray::WaitAll();
      OutputAccuracy(cptr, aptr_y);
      delete[] cptr;
    }

    // update the parameters
    exe->Backward();
    for (int i = 1; i < 5; ++i) {
      in_args[i] -= arg_grad_store[i] * learning_rate;
    }
    NDArray::WaitAll();
  }

  delete exe;
  delete[] aptr_x;
  delete[] aptr_y;
}

int main(int argc, char** argv) {
  MLP();
  return 0;
}

