// Copyright 2018 The Dawn & Tint Authors
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice, this
//    list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution.
//
// 3. Neither the name of the copyright holder nor the names of its
//    contributors may be used to endorse or promote products derived from
//    this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
// DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
// FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
// DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
// SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
// CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
// OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#include "dawn/native/vulkan/ShaderModuleVk.h"

#include <cstdint>
#include <map>
#include <string>
#include <unordered_set>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "dawn/common/HashUtils.h"
#include "dawn/common/MatchVariant.h"
#include "dawn/native/CacheRequest.h"
#include "dawn/native/PhysicalDevice.h"
#include "dawn/native/Serializable.h"
#include "dawn/native/TintUtils.h"
#include "dawn/native/vulkan/BindGroupLayoutVk.h"
#include "dawn/native/vulkan/DeviceVk.h"
#include "dawn/native/vulkan/FencedDeleter.h"
#include "dawn/native/vulkan/PhysicalDeviceVk.h"
#include "dawn/native/vulkan/PipelineLayoutVk.h"
#include "dawn/native/vulkan/UtilsVulkan.h"
#include "dawn/native/vulkan/VulkanError.h"
#include "dawn/platform/DawnPlatform.h"
#include "dawn/platform/metrics/HistogramMacros.h"
#include "dawn/platform/tracing/TraceEvent.h"
#include "partition_alloc/pointers/raw_ptr.h"
#include "tint/tint.h"
#include "dawn/native/Extension.h"

#ifdef DAWN_ENABLE_SPIRV_VALIDATION
#include "dawn/native/SpirvValidation.h"
#endif

namespace dawn::native::vulkan {

#define COMPILED_SPIRV_MEMBERS(X)   \
    X(std::vector<uint32_t>, spirv) \
    X(std::string, remappedEntryPoint)

// Represents the result and metadata for a SPIR-V compilation.
DAWN_SERIALIZABLE(struct, CompiledSpirv, COMPILED_SPIRV_MEMBERS){};
#undef COMPILED_SPIRV_MEMBERS

bool TransformedShaderModuleCacheKey::operator==(
    const TransformedShaderModuleCacheKey& other) const {
    if (layoutPtr != other.layoutPtr || entryPoint != other.entryPoint ||
        constants.size() != other.constants.size()) {
        return false;
    }
    if (!std::equal(constants.begin(), constants.end(), other.constants.begin())) {
        return false;
    }
    if (maxSubgroupSizeForFullSubgroups != other.maxSubgroupSizeForFullSubgroups) {
        return false;
    }
    if (emitPointSize != other.emitPointSize) {
        return false;
    }
    return true;
}

size_t TransformedShaderModuleCacheKeyHashFunc::operator()(
    const TransformedShaderModuleCacheKey& key) const {
    size_t hash = 0;
    HashCombine(&hash, key.layoutPtr, key.entryPoint, key.emitPointSize);
    for (const auto& entry : key.constants) {
        HashCombine(&hash, entry.first, entry.second);
    }
    return hash;
}

class ShaderModule::ConcurrentTransformedShaderModuleCache {
  public:
    explicit ConcurrentTransformedShaderModuleCache(Device* device) : mDevice(device) {}

    ~ConcurrentTransformedShaderModuleCache() {
        std::lock_guard<std::mutex> lock(mMutex);

        for (const auto& [_, moduleAndSpirv] : mTransformedShaderModuleCache) {
            mDevice->GetFencedDeleter()->DeleteWhenUnused(moduleAndSpirv.vkModule);
        }
    }

    std::optional<ModuleAndSpirv> Find(const TransformedShaderModuleCacheKey& key) {
        std::lock_guard<std::mutex> lock(mMutex);

        auto iter = mTransformedShaderModuleCache.find(key);
        if (iter != mTransformedShaderModuleCache.end()) {
            return iter->second.AsRefs();
        }
        return {};
    }
    ModuleAndSpirv AddOrGet(const TransformedShaderModuleCacheKey& key,
                            VkShaderModule module,
                            CompiledSpirv compilation,
                            bool hasInputAttachment) {
        DAWN_ASSERT(module != VK_NULL_HANDLE);
        std::lock_guard<std::mutex> lock(mMutex);

        auto iter = mTransformedShaderModuleCache.find(key);
        if (iter == mTransformedShaderModuleCache.end()) {
            bool added = false;
            std::tie(iter, added) = mTransformedShaderModuleCache.emplace(
                key, Entry{module, std::move(compilation.spirv),
                           std::move(compilation.remappedEntryPoint), hasInputAttachment});
            DAWN_ASSERT(added);
        } else {
            // No need to use FencedDeleter since this shader module was just created and does
            // not need to wait for queue operations to complete.
            // Also, use of fenced deleter here is not thread safe.
            mDevice->fn.DestroyShaderModule(mDevice->GetVkDevice(), module, nullptr);
        }
        return iter->second.AsRefs();
    }

  private:
    struct Entry {
        VkShaderModule vkModule;
        std::vector<uint32_t> spirv;
        std::string remappedEntryPoint;
        bool hasInputAttachment;

        ModuleAndSpirv AsRefs() const {
            return {
                vkModule,           spirv.data(), spirv.size(), remappedEntryPoint.c_str(),
                hasInputAttachment,
            };
        }
    };

    raw_ptr<Device> mDevice;
    std::mutex mMutex;
    absl::flat_hash_map<TransformedShaderModuleCacheKey,
                        Entry,
                        TransformedShaderModuleCacheKeyHashFunc>
        mTransformedShaderModuleCache;
};

// static
ResultOrError<Ref<ShaderModule>> ShaderModule::Create(
    Device* device,
    const UnpackedPtr<ShaderModuleDescriptor>& descriptor,
    const std::vector<tint::wgsl::Extension>& internalExtensions,
    ShaderModuleParseResult* parseResult,
    OwnedCompilationMessages* compilationMessages) {
    Ref<ShaderModule> module = AcquireRef(new ShaderModule(device, descriptor, internalExtensions));
    DAWN_TRY(module->Initialize(parseResult, compilationMessages));
    return module;
}

ShaderModule::ShaderModule(Device* device,
                           const UnpackedPtr<ShaderModuleDescriptor>& descriptor,
                           std::vector<tint::wgsl::Extension> internalExtensions)
    : ShaderModuleBase(device, descriptor, std::move(internalExtensions)),
      mTransformedShaderModuleCache(
          std::make_unique<ConcurrentTransformedShaderModuleCache>(device)) {}

MaybeError ShaderModule::Initialize(ShaderModuleParseResult* parseResult,
                                    OwnedCompilationMessages* compilationMessages) {
    ScopedTintICEHandler scopedICEHandler(GetDevice());
    return InitializeBase(parseResult, compilationMessages);
}

void ShaderModule::DestroyImpl() {
    ShaderModuleBase::DestroyImpl();
    // Remove reference to internal cache to trigger cleanup.
    mTransformedShaderModuleCache = nullptr;
}

ShaderModule::~ShaderModule() = default;

#if TINT_BUILD_SPV_WRITER

#define SPIRV_COMPILATION_REQUEST_MEMBERS(X)                                                     \
    X(SingleShaderStage, stage)                                                                  \
    X(const tint::Program*, inputProgram)                                                        \
    X(std::optional<tint::ast::transform::SubstituteOverride::Config>, substituteOverrideConfig) \
    X(LimitsForCompilationRequest, limits)                                                       \
    X(std::string_view, entryPointName)                                                          \
    X(bool, disableSymbolRenaming)                                                               \
    X(tint::spirv::writer::Options, tintOptions)                                                 \
    X(CacheKey::UnsafeUnkeyedValue<dawn::platform::Platform*>, platform)                         \
    X(std::optional<uint32_t>, maxSubgroupSizeForFullSubgroups)

DAWN_MAKE_CACHE_REQUEST(SpirvCompilationRequest, SPIRV_COMPILATION_REQUEST_MEMBERS);
#undef SPIRV_COMPILATION_REQUEST_MEMBERS

#endif  // TINT_BUILD_SPV_WRITER

ResultOrError<ShaderModule::ModuleAndSpirv> ShaderModule::GetHandleAndSpirv(
    SingleShaderStage stage,
    const ProgrammableStage& programmableStage,
    const PipelineLayout* layout,
    bool clampFragDepth,
    bool emitPointSize,
    std::optional<uint32_t> maxSubgroupSizeForFullSubgroups) {
    TRACE_EVENT0(GetDevice()->GetPlatform(), General, "ShaderModuleVk::GetHandleAndSpirv");

    ScopedTintICEHandler scopedICEHandler(GetDevice());

    // Check to see if we have the handle and spirv cached already
    // TODO(chromium:345359083): Improve the computation of the cache key. For example, it isn't
    // ideal to use `reinterpret_cast<uintptr_t>(layout)` as the layout may be freed and
    // reallocated during the runtime.
    auto cacheKey = TransformedShaderModuleCacheKey{
        reinterpret_cast<uintptr_t>(layout), programmableStage.entryPoint.c_str(),
        programmableStage.constants, maxSubgroupSizeForFullSubgroups, emitPointSize};
    auto handleAndSpirv = mTransformedShaderModuleCache->Find(cacheKey);
    if (handleAndSpirv.has_value()) {
        return std::move(*handleAndSpirv);
    }

    return DAWN_INTERNAL_ERROR("TINT_BUILD_SPV_WRITER is not defined.");
}

}  // namespace dawn::native::vulkan
