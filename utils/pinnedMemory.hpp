#pragma once

#include "cuda_runtime.h"

#include <cstddef>
#include <limits>
#include <new>
#include <type_traits>
#include <vector>

template <typename T>
class PinnedHostAllocator
{
public:
    using value_type = T;

    PinnedHostAllocator() noexcept = default;

    template <typename U>
    PinnedHostAllocator(const PinnedHostAllocator<U> &) noexcept {}

    [[nodiscard]] T *allocate(std::size_t n)
    {
        if (n > std::numeric_limits<std::size_t>::max() / sizeof(T))
            throw std::bad_alloc{};

        void *ptr{};
        const cudaError_t status{cudaMallocHost(&ptr, n * sizeof(T))};
        if (status != cudaSuccess)
            throw std::bad_alloc{};

        return static_cast<T *>(ptr);
    }

    void deallocate(T *ptr, std::size_t) noexcept
    {
        cudaFreeHost(ptr);
    }

    template <typename U>
    bool operator==(const PinnedHostAllocator<U> &) const noexcept
    {
        return true;
    }

    template <typename U>
    bool operator!=(const PinnedHostAllocator<U> &) const noexcept
    {
        return false;
    }
};

using PinnedFloatVector = std::vector<float, PinnedHostAllocator<float>>;
