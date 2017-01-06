/*!
 * Copyright (c) 2015 by Contributors
 */
#include <iostream>
#include <fstream>
#include <map>
#include <string>
#include <vector>
#include "mxnet-cpp/MxNetCpp.h"
using namespace std;
using namespace mxnet::cpp;


std::vector<Symbol> lstm(int num_hidden, Symbol indata,
                         std::vector<Symbol> prev_state, std::vector<Symbol> param,
                         int seqidx, int layeridx, mx_float dropout = 0.0) {
  if (dropout > 0) indata = Dropout("dp", indata, dropout);

  Symbol i2h =
      FullyConnected("t" + std::to_string(seqidx) + "_l" + std::to_string(layeridx) + "_i2h",
                     indata, param[0], param[1],
                     num_hidden *4);

  Symbol h2h =
      FullyConnected("t" + std::to_string(seqidx) + "_l" + std::to_string(layeridx) + "_h2h",
                     prev_state[1], param[2], param[3],
                     num_hidden * 4);

  Symbol gates = i2h + h2h;
  Symbol slice_gates =
      SliceChannel("t" + std::to_string(seqidx) + "_l" + std::to_string(layeridx) + "_slice",
                   gates, 4);

  Symbol in_gate =
      Activation("t" + std::to_string(seqidx) + "_l" + std::to_string(layeridx) + "_in_gates",
                 slice_gates[0], "sigmoid");
  Symbol in_transform =
      Activation("t" + std::to_string(seqidx) + "_l" + std::to_string(layeridx) + "_in_transform",
                 slice_gates[1], "tanh");
  Symbol forget_gate =
      Activation("t" + std::to_string(seqidx) + "_l" + std::to_string(layeridx) + "_forget_data",
                 slice_gates[2], "sigmoid");
  Symbol out_gate =
      Activation("t" + std::to_string(seqidx) + "_l" + std::to_string(layeridx) + "_out_gate",
                 slice_gates[3], "sigmoid");

  Symbol next_c = (forget_gate * prev_state[0]) + (in_gate * in_transform);
  Symbol next_h = out_gate * Activation("", next_c, "tanh");
  std::vector<Symbol> state;
  state.push_back(next_c);
  state.push_back(next_h);
  return state;
}

Symbol lstm_unroll(int num_lstm_layer, int seq_len, int input_size,
                   int num_hidden, int num_embed, int num_label,
                   mx_float dropout = 0.0) {
  Symbol embed_weight = Symbol::Variable("embed_weight");
  Symbol cls_weight = Symbol::Variable("cls_weight");
  Symbol cls_bias = Symbol::Variable("cls_bias");

  std::vector<std::vector<Symbol>> param_cells;
  std::vector<std::vector<Symbol>> last_states;

  for (int i = 0; i < num_lstm_layer; i++) {
    std::vector<Symbol> param;
    param.push_back(Symbol::Variable("l" + std::to_string(i) + "_i2h_weight"));
    param.push_back(Symbol::Variable("l" + std::to_string(i) + "_i2h_bias"));
    param.push_back(Symbol::Variable("l" + std::to_string(i) + "_h2h_weight"));
    param.push_back(Symbol::Variable("l" + std::to_string(i) + "_h2h_bias"));
    param_cells.push_back(param);

    std::vector<Symbol> state;
    state.push_back(Symbol::Variable("l" + std::to_string(i) + "_init_c"));
    state.push_back(Symbol::Variable("l" + std::to_string(i) + "_init_h"));
    last_states.push_back(state);
  }

  Symbol data = Symbol::Variable("data");
  Symbol label = Symbol::Variable("softmax_label");
  Symbol embed = Embedding("embed", data, embed_weight,
                           input_size, num_embed);
  Symbol wordvec = SliceChannel("wordvec", embed, seq_len);
  std::vector<Symbol> hidden_all;
  for (int seqidx = 0; seqidx < seq_len; seqidx++) {
    Symbol hidden = wordvec[seqidx];
    for (int i = 0; i < num_lstm_layer; i++) {
      mx_float dp_ratio;
      if (i == 0)
        dp_ratio = 0;
      else
        dp_ratio = dropout;
      std::vector<Symbol> next_state = lstm(num_hidden,
                                            hidden,
                                            last_states[i],
                                            param_cells[i],
                                            seqidx, i);
      hidden = next_state[1];
      last_states[i] = next_state;
      if (dp_ratio > 0)
        hidden = Dropout("dropout", hidden, dp_ratio);
      hidden_all.push_back(hidden);
    }
  }

  Symbol hidden_concat = Concat("hidden_concat", hidden_all, hidden_all.size(), 0);
  Symbol pred = FullyConnected("pred", hidden_concat,
                               cls_weight, cls_bias, num_label);
  label = Operator("transpose").SetInput("data", label).CreateSymbol("");
  label = Reshape(label, Shape(0));
  return SoftmaxOutput("sm", pred, label);
}

int main() {
  int batch_size = 32;
  int seq_len = 129;
  int num_hidden = 512;
  int num_embed = 256;
  int num_lstm_layer = 3;
  int num_round = 21;
  mx_float learning_rate = 0.01;
  mx_float wd = 0.00001;
  int clip_gradient = 1;
  int update_period = 1;
  int input_size = 65;
  int num_label = input_size;

  Symbol rnn_sym = lstm_unroll(num_lstm_layer, seq_len, input_size,
                               num_hidden, num_embed, num_label, 0.0);

  std::map<std::string, NDArray> args_map;
  args_map["data"] = NDArray(Shape(batch_size, 1, seq_len), Context::cpu());
  args_map["softmax_label"] = NDArray(Shape(batch_size, 1, seq_len), Context::cpu());

  rnn_sym.InferArgsMap(Context::cpu(), &args_map, args_map);
}
