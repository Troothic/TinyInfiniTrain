#include "example/common/tiny_shakespeare_dataset.h"

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <memory>
#include <numeric>
#include <string>
#include <unordered_map>
#include <utility>
#include <variant>
#include <vector>

#include "glog/logging.h"

#include "infini_train/include/tensor.h"

namespace {
using DataType = infini_train::DataType;
using TinyShakespeareType = TinyShakespeareDataset::TinyShakespeareType;
using TinyShakespeareFile = TinyShakespeareDataset::TinyShakespeareFile;

const std::unordered_map<int, TinyShakespeareType> kTypeMap = {
    {20240520, TinyShakespeareType::kUINT16}, // GPT-2
    {20240801, TinyShakespeareType::kUINT32}, // LLaMA 3
};

const std::unordered_map<TinyShakespeareType, size_t> kTypeToSize = {
    {TinyShakespeareType::kUINT16, 2},
    {TinyShakespeareType::kUINT32, 4},
};

const std::unordered_map<TinyShakespeareType, DataType> kTypeToDataType = {
    {TinyShakespeareType::kUINT16, DataType::kUINT16},
    {TinyShakespeareType::kUINT32, DataType::kINT32},
};

std::vector<uint8_t> ReadSeveralBytesFromIfstream(size_t num_bytes, std::ifstream *ifs) {
    std::vector<uint8_t> result(num_bytes);
    ifs->read(reinterpret_cast<char *>(result.data()), num_bytes);
    return result;
}

template <typename T> T BytesToType(const std::vector<uint8_t> &bytes, size_t offset) {
    static_assert(std::is_trivially_copyable<T>::value, "T must be trivially copyable.");
    T value;
    std::memcpy(&value, &bytes[offset], sizeof(T));
    return value;
}

TinyShakespeareFile ReadTinyShakespeareFile(const std::string &path, size_t sequence_length) {
    /* =================================== Assignment ===================================
       Parse binary dataset file
       File format:
    ----------------------------------------------------------------------------------
    | HEADER (1024 bytes)                     | DATA (tokens)                        |
    | magic(4B) | version(4B) | num_toks(4B) | reserved(1012B) | token data          |
    ----------------------------------------------------------------------------------
       =================================== Assignment =================================== */

    std::ifstream file(path, std::ios::binary);
    CHECK(file.is_open()) << "Cannot open file: " << path;

    // Read header (1024 bytes)
    auto header = ReadSeveralBytesFromIfstream(1024, &file);
    
    // Parse header fields
    uint32_t magic = BytesToType<uint32_t>(header, 0);        // magic (4B)
    uint32_t version = BytesToType<uint32_t>(header, 4);      // version (4B)
    uint32_t num_tokens = BytesToType<uint32_t>(header, 8);   // num_toks (4B)

    // Check version validity
    CHECK(kTypeMap.count(version)) << "Unsupported version: " << version;
    TinyShakespeareType type = kTypeMap.at(version);
    size_t token_size = kTypeToSize.at(type);
    DataType data_type = kTypeToDataType.at(type);

    // Compute number of sequences (each sequence contains sequence_length tokens)
    size_t num_sequences = num_tokens / sequence_length;
    
    // Create dims
    std::vector<int64_t> dims = {static_cast<int64_t>(num_sequences), static_cast<int64_t>(sequence_length)};
    
    // Create Tensor to store token data
    infini_train::Tensor tensor(dims, data_type);
    
    // Read token data into Tensor
    size_t data_size = num_sequences * sequence_length * token_size;
    file.read(static_cast<char *>(tensor.DataPtr()), data_size);

    TinyShakespeareFile result;
    result.tensor = std::move(tensor);
    result.dims = dims;
    result.type = type;
    
    return result;
}
} // namespace

TinyShakespeareDataset::TinyShakespeareDataset(const std::string &filepath, size_t sequence_length)
    : sequence_length_(sequence_length),
      sequence_size_in_bytes_(sequence_length * sizeof(int64_t)),
      text_file_(ReadTinyShakespeareFile(filepath, sequence_length)),
      num_samples_(text_file_.dims[0] - 1) {
    // =================================== Assignment ===================================
    // Initialize dataset instance
    // Call ReadTinyShakespeareFile to load data file
    // =================================== Assignment ===================================
}

std::pair<std::shared_ptr<infini_train::Tensor>, std::shared_ptr<infini_train::Tensor>>
TinyShakespeareDataset::operator[](size_t idx) const {
    CHECK_LT(idx, text_file_.dims[0] - 1);
    std::vector<int64_t> dims = std::vector<int64_t>(text_file_.dims.begin() + 1, text_file_.dims.end());
    // x: (seq_len), y: (seq_len) -> stack -> (bs, seq_len) (bs, seq_len)
    return {std::make_shared<infini_train::Tensor>(text_file_.tensor, idx * sequence_size_in_bytes_, dims),
            std::make_shared<infini_train::Tensor>(text_file_.tensor, idx * sequence_size_in_bytes_ + sizeof(int64_t),
                                                   dims)};
}

size_t TinyShakespeareDataset::Size() const { return num_samples_; }
