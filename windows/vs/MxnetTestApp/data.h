#pragma once

#include <condition_variable>
#include <fstream>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include <dmlc/logging.h>

class DataReader {
public:
  DataReader(const std::string& filePath,
    int recordSize,
    int batchSize)
    : filePath_(filePath),
    recordSize_(recordSize),
    batchSize_(batchSize),
    reset_(false),
    eof_(false),
    exit_(false) {
    ioThread_ = std::thread([this](){this->IOThread(); });
  }
  ~DataReader() {
    {
      std::unique_lock<std::mutex> l(mutex_);
      exit_ = true;
      condEmpty_.notify_one();
    }
    ioThread_.join();
  }

  bool Eof() {
    std::lock_guard<std::mutex> l(mutex_);
    return eof_;
  }

  void Reset() {
    std::lock_guard<std::mutex> l(mutex_);
    reset_ = true;
    eof_ = false;
    if (!buffer_.empty()) buffer_.clear();
    condEmpty_.notify_one();
  }

  std::vector<float> ReadBatch() {
    std::unique_lock<std::mutex> l(mutex_);
    std::vector<float> r;
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
    std::unique_lock<std::mutex> l(mutex_);
    while (!exit_) {
      std::ifstream in(filePath_, std::ios::binary);
      CHECK(in.good()) << "error opening file " << filePath_ << std::endl;
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

  std::thread ioThread_;
  std::mutex mutex_;
  std::condition_variable condReady_;
  std::condition_variable condEmpty_;
  std::vector<float> buffer_;
  bool reset_;
  bool eof_;
  bool exit_;

  const int recordSize_;
  const int batchSize_;
  const std::string filePath_;
};

void TestDataReader() {
  DataReader r("d.txt", 3, 2);
  for (int i = 0; i < 3; i++) {
    while (!r.Eof()) {
      vector<float> v = r.ReadBatch();
      if (v.empty())
        break;
      cout << std::hex << *(uint32_t*)(&v[0]) << endl;
    }
    r.Reset();
  }
}