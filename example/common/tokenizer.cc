#include "example/common/tokenizer.h"

#include <cctype>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <vector>

#include "glog/logging.h"

namespace infini_train {

constexpr uint32_t kGpt2Eot = 50256;
constexpr uint32_t kLLaMA3Eot = 128001;
constexpr uint64_t kRandomU32Multiplier = 0x2545F4914F6CDD1Dull;
constexpr float kF32Divisor = 16777216.0f; // 2^24
constexpr uint64_t kRngState = 1337;

using Version = Tokenizer::Version;

const std::unordered_map<uint32_t, uint32_t> kEotMap = {
    {20240328, kGpt2Eot},   // GPT-2
    {20240801, kLLaMA3Eot}, // LLaMA-3
};

const std::unordered_map<uint32_t, std::vector<uint32_t>> kPromptMap = {
    // e.g. "The meaning of life is"
    // ref: https://tiktokenizer.vercel.app/
    {20240328, std::vector<uint32_t>{464, 3616, 286, 1204, 318}}, // GPT-2
    {20240801, std::vector<uint32_t>{791, 7438, 315, 2324, 374}}, // LLaMA-3
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

unsigned int RandomU32(uint64_t &state) {
    state ^= state >> 12;
    state ^= state << 25;
    state ^= state >> 27;
    return (state * kRandomU32Multiplier) >> 32;
}

float RandomF32(uint64_t &state) { // random float32 in [0,1)
    return (RandomU32(state) >> 8) / kF32Divisor;
}

int SampleMult(float *probabilities, int n, float coin) {
    // sample index from probabilities (they must sum to 1!)
    // coin is a random number in [0, 1), usually from RandomF32()
    float cdf = 0.0f;
    for (int i = 0; i < n; i++) {
        cdf += probabilities[i];
        if (coin < cdf) {
            return i;
        }
    }
    return n - 1; // in case of rounding errors
}

Tokenizer::Tokenizer(const std::string &filepath) {
    /* ===================================== Assignment =====================================
    Load Tokenizer binary file

    File format:
    ----------------------------------------------------------------------------------
    | HEADER (1024 bytes)                     | VOCAB TABLE                           |
    | magic(4B) | version(4B) | vocab_size(4B) | reserved(1012B) | token vocab data   |
    ----------------------------------------------------------------------------------
    ===================================== Assignment ===================================== */

    std::ifstream file(filepath, std::ios::binary);
    CHECK(file.is_open()) << "Cannot open tokenizer file: " << filepath;

    // Read header (1024 bytes)
    auto header = ReadSeveralBytesFromIfstream(1024, &file);
    
    // Parse header fields
    magic_number_ = BytesToType<uint32_t>(header, 0);    // magic (4B)
    uint32_t version = BytesToType<uint32_t>(header, 4); // version (4B)
    vocab_size_ = BytesToType<uint32_t>(header, 8);      // vocab_size (4B)

    // Set end-of-text token
    CHECK(kEotMap.count(magic_number_)) << "Unsupported magic number: " << magic_number_;
    eot_token_ = kEotMap.at(magic_number_);

    // Read vocabulary table
    token_table_.resize(vocab_size_);
    for (uint32_t i = 0; i < vocab_size_; ++i) {
        // Read token length (1 byte)
        uint8_t token_len;
        file.read(reinterpret_cast<char *>(&token_len), 1);
        
        // Read token string
        std::string token(token_len, '\0');
        file.read(token.data(), token_len);
        token_table_[i] = std::move(token);
    }
}

std::string Tokenizer::Decode(uint32_t token_id) const {
    /* ===================================== Assignment =====================================
    Convert token_id to text
    Return the text fragment corresponding to token_id
    ===================================== Assignment ===================================== */
    
    // Check token_id validity
    CHECK_LT(token_id, token_table_.size()) << "token_id out of range: " << token_id;
    return token_table_[token_id];
}

void Tokenizer::GenerateText(infini_train::nn::Module &model, uint32_t batch_size, uint32_t sequence_length,
                             uint32_t text_length, Device device) const {
    std::vector<int64_t> dims;
    dims.assign({batch_size, sequence_length});
    // x_tensor (FLAGS_batch_size, FLAGS_sequence_length) eq:(4, 64)
    infini_train::Tensor x_tensor = infini_train::Tensor(dims, DataType::kINT64);
    int64_t *x_buff = static_cast<int64_t *>(x_tensor.DataPtr());
    for (int i = 0; i < batch_size * sequence_length; ++i) { x_buff[i] = eot_token_; }

    // Give some contexts: "The meaning of life is "
    auto prompt = kPromptMap.at(magic_number_);
    auto prompt_len = prompt.size();
    for (int i = 0; i < prompt_len; ++i) { x_buff[i] = prompt[i]; }
    std::cout << "The meaning of life is";

    auto x = std::make_shared<infini_train::Tensor>(x_tensor.To(device));
    uint64_t rng_state = kRngState;
    LOG(INFO) << "start generate text:";
    for (int t = prompt_len; t < text_length; t++) {
        /* ===================================== Assignment =====================================
        Single-step text generation:
        Call model.Forward to get logits, sample based on results, call Decode to get text
        ===================================== Assignment ===================================== */

        // Forward pass to get logits
        auto logits = model.Forward({x})[0];
        
        // Get logits at last position: logits[:, t-1, :] -> shape: (batch_size, vocab_size)
        auto logits_last = logits->Slice(1, t - 1, t, 1)->Squeeze(1);
        
        // Apply softmax to get probability distribution
        auto probs = nn::functional::Softmax(logits_last, -1);
        
        // Transfer probabilities to CPU for sampling
        auto probs_cpu = std::make_shared<Tensor>(probs->To(Device(DeviceType::kCPU, 0)));
        float *probs_ptr = static_cast<float *>(probs_cpu->DataPtr());
        
        // Sample to get next token (take first batch only)
        float coin = RandomF32(rng_state);
        int next_token = SampleMult(probs_ptr, vocab_size_, coin);
        
        // Update input sequence
        int64_t *x_ptr = static_cast<int64_t *>(x_tensor.DataPtr());
        x_ptr[t] = next_token;
        x = std::make_shared<Tensor>(x_tensor.To(device));
        
        // Decode and output text
        std::cout << Decode(next_token) << std::flush;
    }
    std::cout << std::endl;
}
} // namespace infini_train
