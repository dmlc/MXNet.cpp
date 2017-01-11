/*!
 * Copyright (c) 2016 by Contributors
 * Hua Zhang mz24cn@hotmail.com
 * The code implements C++ version charRNN for mxnet\example\rnn\char-rnn.ipynb with MXNet.cpp API.
 * The generated params file is compatiable with python version.
 * train() and predict() has been verified with original data samples.
 */

#pragma warning(disable: 4996)
#include <iostream>
#include <fstream>
#include <unordered_map>
#include <vector>
#include <string>
#include <tuple>
#include <algorithm>
#include <functional>
#include <thread>
#include <chrono>
#include "mxnet-cpp/MxNetCpp.h"

using namespace std;
using namespace mxnet::cpp;

struct LSTMState {
	Symbol C;
	Symbol h;
};

struct LSTMParam {
	Symbol i2h_weight;
	Symbol i2h_bias;
	Symbol h2h_weight;
	Symbol h2h_bias;
};

//LSTM Cell symbol
LSTMState LSTM(int num_hidden, Symbol& indata, LSTMState& prev_state, LSTMParam& param, int seqidx, int layeridx, mx_float dropout=0)
{
	auto input = dropout > 0? Dropout(indata, dropout) : indata;
	auto prefix = string("t") + to_string(seqidx) + "_l" + to_string(layeridx);
	auto i2h = FullyConnected(prefix + "_i2h", input, param.i2h_weight, param.i2h_bias, num_hidden * 4);
	auto h2h = FullyConnected(prefix + "_h2h", prev_state.h, param.h2h_weight, param.h2h_bias, num_hidden * 4);
	auto gates = i2h + h2h;
	auto slice_gates = SliceChannel(prefix + "_slice", gates, 4);
	auto in_gate = Activation(slice_gates[0], ActivationActType::sigmoid);
	auto in_transform = Activation(slice_gates[1], ActivationActType::tanh);
	auto forget_gate = Activation(slice_gates[2], ActivationActType::sigmoid);
	auto out_gate = Activation(slice_gates[3], ActivationActType::sigmoid);

	LSTMState state;
	state.C = (forget_gate * prev_state.C) + (in_gate * in_transform);
	state.h = out_gate * Activation(state.C, ActivationActType::tanh);
	return state;
}

Symbol LSTMUnroll(int num_lstm_layer, int sequence_length, int input_dim,
        int num_hidden, int num_embed, int batch_size, mx_float dropout=0)
{
	auto data = Symbol::Variable("data");
	auto embed_weight = Symbol::Variable("embed_weight");
	auto embed = Embedding("embed", data, embed_weight, input_dim, num_embed);
	auto wordvec = SliceChannel(embed, sequence_length, 1, true);

	vector<LSTMState> last_states;
	vector<LSTMParam> param_cells;
	for (int l = 0; l < num_lstm_layer; l++) {
		string layer = "l" + to_string(l);
		LSTMParam param;
		param.i2h_weight = Symbol::Variable(layer + "_i2h_weight");
		param.i2h_bias = Symbol::Variable(layer + "_i2h_bias");
		param.h2h_weight = Symbol::Variable(layer + "_h2h_weight");
		param.h2h_bias = Symbol::Variable(layer + "_h2h_bias");
		param_cells.push_back(param);
		LSTMState state;
		state.C = Symbol::Variable(layer + "_init_c");
		state.h = Symbol::Variable(layer + "_init_h");
		last_states.push_back(state);
	}

	vector<Symbol> hidden_all;
	for (int i = 0; i < sequence_length; i++) {
		auto hidden = wordvec[i];
		for (int layer = 0; layer < num_lstm_layer; layer++) {
			double dp_ratio = layer == 0? 0 : dropout;
			auto next_state = LSTM(num_hidden, hidden, last_states[layer], param_cells[layer], i, layer, dp_ratio);
			hidden = next_state.h;
			last_states[layer] = next_state;
		}
        if (dropout > 0)
            hidden = Dropout(hidden, dropout);
        hidden_all.push_back(hidden);
	}

	auto hidden_concat = Concat(hidden_all, hidden_all.size(), 0);
	auto cls_weight = Symbol::Variable("cls_weight");
	auto cls_bias = Symbol::Variable("cls_bias");
	auto pred = FullyConnected("pred", hidden_concat, cls_weight, cls_bias, input_dim);

	auto label = Symbol::Variable("softmax_label");
	label = transpose(label);
	label = Reshape(label, Shape(0));
	auto sm = SoftmaxOutput("softmax", pred, label);
	if (sequence_length == 1) {
		vector<Symbol> outputs = { sm };
		for (auto& state : last_states) {
			outputs.push_back(state.C);
			outputs.push_back(state.h);
		}
		return Symbol::Group(outputs);
	}
	return sm;
}

class Shuffler {
	vector<int> sequence;
public:
	Shuffler(int size) : sequence(size) {
		int* p = sequence.data();
		for (int i = 0; i < size; i++)
			*p++ = i;
	}
	void shuffle(function<void(int, int)> lambda = nullptr) {
		random_shuffle(sequence.begin(), sequence.end());
		int n = 0;
		if (lambda != nullptr)
			for (int i : sequence)
				lambda(n++, i);
	}
	const int* data() {
		return sequence.data();
	}
};

class BucketSentenceIter : public DataIter {
	Shuffler* random;
	int batch, current, end, sequence_length;
	Context device;
	vector<vector<mx_float>> sequences;
	vector<wchar_t> index2chars;
	unordered_map<wchar_t, mx_float> charIndices;

public:
	BucketSentenceIter(string filename, int minibatch, Context context) : batch(minibatch), current(-1), device(context) {
		auto& content = readContent(filename);
		buildCharIndex(content);
		sequences = convertTextToSequences(content, '\n');

		int N = sequences.size() / batch * batch; // total used samples
		sequences.resize(N);
		sort(sequences.begin(), sequences.end(), [](const vector<mx_float>& a, const vector<mx_float>& b) { return a.size() < b.size(); });

		sequence_length = sequences.back().size();
		random = new Shuffler(N);
//		vector<vector<mx_float>>* target = &sequences;
//		random->shuffle([target](int n, int i) { (*target)[n].swap((*target)[i]); }); //We still can get random results if call Reset() firstly
		end = N / batch;
	}
	virtual ~BucketSentenceIter() {
		delete random;
	}

	unsigned int maxSequenceLength() {
		return sequence_length;
	}

	size_t characterSize() {
		return charIndices.size();
	}

	virtual bool Next(void) {
		return ++current < end;
	}
	virtual NDArray GetData(void) {
		const int* indices = random->data();
		mx_float *data = new mx_float[sequence_length * batch], *pdata = data;

		for (int i = current * batch, end = i + batch; i < end; i++) {
			memcpy(pdata, sequences[indices[i]].data(), sequences[indices[i]].size() * sizeof(mx_float));
			if (sequences[indices[i]].size() < sequence_length)
				memset(pdata + sequences[indices[i]].size(), 0, (sequence_length - sequences[indices[i]].size()) * sizeof(mx_float));
			pdata += sequence_length;
		}
		NDArray array(Shape(batch, sequence_length), device, false);
		array.SyncCopyFromCPU(data, batch * sequence_length);
		return array;
	}
	virtual NDArray GetLabel(void) {
		const int* indices = random->data();
		mx_float *label = new mx_float[sequence_length * batch], *plabel = label;

		for (int i = current * batch, end = i + batch; i < end; i++) {
			memcpy(plabel, sequences[indices[i]].data() + 1, (sequences[indices[i]].size() - 1) * sizeof(mx_float));
			memset(plabel + sequences[indices[i]].size() - 1, 0, (sequence_length - sequences[indices[i]].size() + 1) * sizeof(mx_float));
			plabel += sequence_length;
		}
		NDArray array(Shape(batch, sequence_length), device, false);
		array.SyncCopyFromCPU(label, batch * sequence_length);
		return array;
	}
	virtual int GetPadNum(void) {
		return sequence_length - sequences[random->data()[current * batch]].size();
	}
	virtual std::vector<int> GetIndex(void) {
		const int* indices = random->data();
		vector<int> list(indices + current * batch, indices + current * batch + batch);
		return list;
	}
	virtual void BeforeFirst(void) {
		current = -1;
		random->shuffle(nullptr);
	}

	wstring readContent(const string file)
	{
		wifstream ifs(file, ios::binary);
		if (ifs) {
			wostringstream os;
			os << ifs.rdbuf();
			return os.str();
		}
		return L"";
	}

	void buildCharIndex(wstring& content) // This version buildCharIndex() Compatiable with python version char_rnn dictionary
	{
		int n = 1;
		charIndices['\0'] = 0; //padding character
		index2chars.push_back(0); //padding character index
		for (auto c : content)
			if (charIndices.find(c) == charIndices.end()) {
				charIndices[c] = n++;
				index2chars.push_back(c);
			}
	}
//	void buildCharIndex(wstring& content)
//	{
//		for (auto c : content)
//			charIndices[c]++; // char-frequency map; then char-index map
//		vector<tuple<wchar_t, mx_float>> characters;
//		for (auto& iter : charIndices)
//			characters.push_back(make_tuple(iter.first, iter.second));
//		sort(characters.begin(), characters.end(), [](const tuple<wchar_t, mx_float>& a, const tuple<wchar_t, mx_float>& b) { return get<1>(a) > get<1>(b); });
//		mx_float index = 1; //0 is left for zero-padding
//		index2chars.clear();
//		index2chars.push_back(0); //zero-padding
//		for (auto& t : characters) {
//			charIndices[get<0>(t)] = index++;
//			index2chars.push_back(get<0>(t));
//		}
//	}

	inline wchar_t character(int i)
	{
		return index2chars[i];
	}

	inline mx_float index(wchar_t c)
	{
		return charIndices[c];
	}

	void saveCharIndices(const string file)
	{
		wofstream ofs(file, ios::binary);
		if (ofs) {
			ofs.write(index2chars.data() + 1, index2chars.size() - 1);
			ofs.close();
		}
	}

	static tuple<unordered_map<wchar_t, mx_float>, vector<wchar_t>> loadCharIndices(const string file)
	{
		wifstream ifs(file, ios::binary);
		unordered_map<wchar_t, mx_float> map;
		vector<wchar_t> chars;
		if (ifs) {
			wostringstream os;
			os << ifs.rdbuf();
			int n = 1;
			map[L'\0'] = 0;
			chars.push_back(L'\0');
			for (auto c : os.str()) {
				map[c] = (mx_float) n++;
				chars.push_back(c);
			}
		}
		return {map, chars};
	}

	vector<vector<mx_float>> convertTextToSequences(wstring& content, wchar_t spliter)
	{
		vector<vector<mx_float>> sequences;
		sequences.push_back(vector<mx_float>());
		for (auto c : content)
			if (c == spliter && !sequences.back().empty())
				sequences.push_back(vector<mx_float>());
			else
				sequences.back().push_back(charIndices[c]);
		return sequences;
	}
};

void OutputPerplexity(NDArray& labels, NDArray& output)
{
	vector<mx_float> charIndices, a;
	labels.SyncCopyToCPU(&charIndices, 0L);
	output.SyncCopyToCPU(&a, 0)/*4128*84*/;
	mx_float loss = 0;
	int batchSize = labels.GetShape()[0]/*32*/, sequenceLength = labels.GetShape()[1]/*129*/, nSamples = output.GetShape()[0]/*4128*/, vocabSize = output.GetShape()[1]/*84*/;
	for (int n = 0; n < nSamples; n++) {
		int row = n % batchSize, column = n / batchSize, labelOffset = column + row * sequenceLength; //Search based on column storage: labels.T
		mx_float safe_value = max(1e-10f, a[vocabSize * n + int(charIndices[labelOffset])]);
		loss += -log(safe_value); //Calculate Cross Entropy loss function for Softmax output
	}
	loss = exp(loss / nSamples);
	cout << "Train-Perplexity=" << loss << endl;
}

void SaveCheckpoint(const string filepath, Symbol net, Executor* exe)
{
	map<string, NDArray> params;
	for (auto iter : exe->arg_dict())
		if (iter.first.find("_init_") == string::npos && iter.first.rfind("data") != iter.first.length() - 4 && iter.first.rfind("label") != iter.first.length() - 5)
			params.insert({"arg:" + iter.first, iter.second});
	for (auto iter : exe->aux_dict())
			params.insert({"aux:" + iter.first, iter.second});
	NDArray::Save(filepath, params);
}

void LoadCheckpoint(const string filepath, Executor* exe)
{
	map<std::string, NDArray> params = NDArray::LoadToMap(filepath);
	for (auto iter : params) {
		string type = iter.first.substr(0, 4);
		string name = iter.first.substr(4);
		NDArray target;
		if (type == "arg:")
			target = exe->arg_dict()[name];
		else if (type == "aux:")
			target = exe->aux_dict()[name];
		else
			continue;
		iter.second.CopyTo(&target);
	}
}

int input_dim = 0;/*84*/
int sequence_length_max = 0;/*129*/
int num_embed = 256;
int num_lstm_layer = 3;
int num_hidden = 512;
mx_float dropout = 0.2;
void train(const string file, int batch_size, int max_epoch)
{
	Context device(DeviceType::kGPU, 0);
	BucketSentenceIter dataIter(file, batch_size, device);
	string prefix = file.substr(0, file.rfind("."));
	dataIter.saveCharIndices(prefix + ".dictionary");

	input_dim = (int) dataIter.characterSize();
	sequence_length_max = dataIter.maxSequenceLength();

	auto RNN = LSTMUnroll(num_lstm_layer, sequence_length_max, input_dim, num_hidden, num_embed, batch_size, dropout);
	map<string, NDArray> args_map;
	args_map["data"] = NDArray(Shape(batch_size, sequence_length_max), device, false);
	args_map["softmax_label"] = NDArray(Shape(batch_size, sequence_length_max), device, false);
	for (int i = 0; i < num_lstm_layer; i++) {
		string key = "l" + to_string(i) + "_init_";
		args_map[key + "c"] = NDArray(Shape(batch_size, num_hidden), device, false);
		args_map[key + "h"] = NDArray(Shape(batch_size, num_hidden), device, false);
	}
	vector<mx_float> zeros(batch_size * num_hidden, 0);
	Executor* exe = RNN.SimpleBind(device, args_map, {}, {{"data", kNullOp}});

	Xavier xavier = Xavier(Xavier::gaussian, Xavier::in, 2.34);
	for (auto &arg : exe->arg_dict())
		xavier(arg.first, &arg.second);

	mx_float learning_rate = 0.0002;
	mx_float weight_decay = 0.000002;
	Optimizer* opt = OptimizerRegistry::Find("ccsgd");
//	opt->SetParam("momentum", 0.9)->SetParam("rescale_grad", 1.0 / batch_size)->SetParam("clip_gradient", 10);
	char filepath[256];

	for (int epoch = 0; epoch < max_epoch; ++epoch) {
		dataIter.Reset();
		auto tic = chrono::system_clock::now();
		while (dataIter.Next()) {
			auto data_batch = dataIter.GetDataBatch();
			data_batch.data.CopyTo(&exe->arg_dict()["data"]);
			data_batch.label.CopyTo(&exe->arg_dict()["softmax_label"]);
			for (int l = 0; l < num_lstm_layer; l++) {
				string key = "l" + to_string(l) + "_init_";
				exe->arg_dict()[key + "c"].SyncCopyFromCPU(zeros);
				exe->arg_dict()[key + "h"].SyncCopyFromCPU(zeros);
			}
			NDArray::WaitAll();

			exe->Forward(true);
			exe->Backward();
			exe->UpdateAll(opt, learning_rate, weight_decay);
			NDArray::WaitAll();
		}
		auto toc = chrono::system_clock::now();
		cout << "Epoch[" << epoch << "] Time Cost:" << chrono::duration_cast<chrono::seconds>(toc - tic).count() << " seconds ";
		OutputPerplexity(exe->arg_dict()["softmax_label"], exe->outputs[0]);
		sprintf(filepath, "%s-%04d.params", prefix.c_str(), epoch + 1);
		SaveCheckpoint(filepath, RNN, exe);
	}
}

void predict(wstring& text, int sequence_length, const string param_file, const string dictionary_file)
{
	Context device(DeviceType::kGPU, 0);
	auto results = BucketSentenceIter::loadCharIndices(dictionary_file);
	auto dictionary = get<0>(results);
	auto charIndices = get<1>(results);
	input_dim = (int) charIndices.size();
	auto RNN = LSTMUnroll(num_lstm_layer, 1, input_dim, num_hidden, num_embed, 1, 0);

	map<string, NDArray> args_map;
	args_map["data"] = NDArray(Shape(1, 1), device, false);
	args_map["softmax_label"] = NDArray(Shape(1, 1), device, false);
	vector<mx_float> zeros(num_hidden, 0);
	for (int l = 0; l < num_lstm_layer; l++) {
		string key = "l" + to_string(l) + "_init_";
		args_map[key + "c"] = NDArray(Shape(1, num_hidden), device, false);
		args_map[key + "h"] = NDArray(Shape(1, num_hidden), device, false);
		args_map[key + "c"].SyncCopyFromCPU(zeros);
		args_map[key + "h"].SyncCopyFromCPU(zeros);
	}
	Executor* exe = RNN.SimpleBind(device, args_map);
	LoadCheckpoint(param_file, exe);

	mx_float index;
	wchar_t next;
	vector<mx_float> softmax;
	softmax.resize(input_dim);
	for (auto c : text) {
		exe->arg_dict()["data"].SyncCopyFromCPU(&dictionary[c], 1);
		exe->Forward(false);

		exe->outputs[0].SyncCopyToCPU(softmax.data(), input_dim);
		for (int l = 0; l < num_lstm_layer; l++) {
			string key = "l" + to_string(l) + "_init_";
			exe->outputs[l * 2 + 1].CopyTo(&args_map[key + "c"]);
			exe->outputs[l * 2 + 2].CopyTo(&args_map[key + "h"]);
		}

		size_t n = max_element(softmax.begin(), softmax.end()) - softmax.begin();
		index = (mx_float) n;
		next = charIndices[n];
	}
	text.push_back(next);

	for (int i = 0; i < sequence_length; i++) {
		exe->arg_dict()["data"].SyncCopyFromCPU(&index, 1);
		exe->Forward(false);

		exe->outputs[0].SyncCopyToCPU(softmax.data(), input_dim);
		for (int l = 0; l < num_lstm_layer; l++) {
			string key = "l" + to_string(l) + "_init_";
			exe->outputs[l * 2 + 1].CopyTo(&args_map[key + "c"]);
			exe->outputs[l * 2 + 2].CopyTo(&args_map[key + "h"]);
		}

		size_t n = max_element(softmax.begin(), softmax.end()) - softmax.begin();
		index = (mx_float) n;
		next = charIndices[n];
		text.push_back(next);
	}
}

int main(int argc, char** argv)
{
	if (argc < 5) {
		cout << "Usage for training: charRNN train {corpus file} {batch size} {max epoch}" << endl;
		cout << "Usage for prediction: charRNN predict {params file} {dictionary file} {beginning of text}" << endl;
		return 0;
	}

	string task = argv[1];
	if (task == "train")
		train(argv[2], atoi(argv[3]), atoi(argv[4])); // this function will generate dictionary file and params file.
	else if (task == "predict") {
		wstring text;// = L"If there is anyone out there who still doubts ";
		for (char c : string(argv[4])) // Considering of extending to Chinese samples in future, use wchar_t instead of char
			text.push_back((wchar_t) c);
		predict(text, 600, argv[2], argv[3]); // Python version predicts text default to random selecltions. Here I didn't write the random code, always choose the 'best' character. So the text length reduced to 600. Longer size often leads to repeated sentances, since training sequence length is only 129 for obama corpus.
		wcout << text << endl;
	}

	MXNotifyShutdown();
}
