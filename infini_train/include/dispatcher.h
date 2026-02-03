#pragma once

#include <iostream>
#include <map>
#include <type_traits>
#include <utility>

#include "glog/logging.h"

#include "infini_train/include/device.h"

namespace infini_train {
class KernelFunction {
public:
    template <typename FuncT> explicit KernelFunction(FuncT &&func) : func_ptr_(reinterpret_cast<void *>(func)) {}

    template <typename RetT, class... ArgsT> RetT Call(ArgsT... args) const {
        // =================================== 作业 ===================================
        // 实现通用kernel调用接口
        // 功能描述：将存储的函数指针转换为指定类型并调用
        // =================================== 作业 ===================================

        using FuncT = RetT (*)(ArgsT...);
        // 将 void* 函数指针转换为目标函数类型
        FuncT func = reinterpret_cast<FuncT>(func_ptr_);
        // 调用函数并返回结果
        return func(args...);
    }

private:
    void *func_ptr_ = nullptr;
};

class Dispatcher {
public:
    using KeyT = std::pair<DeviceType, std::string>;

    static Dispatcher &Instance() {
        static Dispatcher instance;
        return instance;
    }

    const KernelFunction GetKernel(const KeyT &key) const {
        // key: {device_type, name}
        auto it = key_to_kernel_map_.find(key);
        CHECK(it != key_to_kernel_map_.end())
            << "Kernel not found: " << key.second << " on device: " << static_cast<int>(key.first);
        return it->second;
    }

    template <typename FuncT> void Register(const KeyT &key, FuncT &&kernel) {
        // =================================== Assignment ===================================
        // Implement kernel registration mechanism
        // =================================== Assignment ===================================

        // Check for duplicate registration
        CHECK(key_to_kernel_map_.find(key) == key_to_kernel_map_.end()) << "Kernel already registered: " << key.second;
        // Wrap kernel function as KernelFunction and store in map
        key_to_kernel_map_.emplace(key, KernelFunction(std::forward<FuncT>(kernel)));
    }

private:
    std::map<KeyT, KernelFunction> key_to_kernel_map_;
};
} // namespace infini_train

#define REGISTER_KERNEL(device, kernel_name, kernel_func)                                                              \
    static int __register_kernel_##kernel_name##__##__COUNTER__ = []() {                                               \
        infini_train::Dispatcher::Instance().Register({device, #kernel_name}, kernel_func);                            \
        return 0;                                                                                                      \
    }();
