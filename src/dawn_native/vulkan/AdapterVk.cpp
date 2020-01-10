// Copyright 2019 The Dawn Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "dawn_native/vulkan/AdapterVk.h"

#include "dawn_native/vulkan/BackendVk.h"
#include "dawn_native/vulkan/DeviceVk.h"

namespace dawn_native { namespace vulkan {

    Adapter::Adapter(Backend* backend, VkPhysicalDevice physicalDevice)
        : AdapterBase(backend->GetInstance(), wgpu::BackendType::Vulkan),
          mPhysicalDevice(physicalDevice),
          mBackend(backend) {
    }

    const VulkanDeviceInfo& Adapter::GetDeviceInfo() const {
        return mDeviceInfo;
    }

    VkPhysicalDevice Adapter::GetPhysicalDevice() const {
        return mPhysicalDevice;
    }

    Backend* Adapter::GetBackend() const {
        return mBackend;
    }

    MaybeError Adapter::Initialize() {
        DAWN_TRY_ASSIGN(mDeviceInfo, GatherDeviceInfo(*this));
        if (!mDeviceInfo.maintenance1 &&
            mDeviceInfo.properties.apiVersion < VK_MAKE_VERSION(1, 1, 0)) {
            return DAWN_DEVICE_LOST_ERROR(
                "Dawn requires Vulkan 1.1 or Vulkan 1.0 with KHR_Maintenance1 in order to support "
                "viewport flipY");
        }

        InitializeSupportedExtensions();

        mPCIInfo.deviceId = mDeviceInfo.properties.deviceID;
        mPCIInfo.vendorId = mDeviceInfo.properties.vendorID;
        mPCIInfo.name = mDeviceInfo.properties.deviceName;

        switch (mDeviceInfo.properties.deviceType) {
            case VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU:
                mAdapterType = wgpu::AdapterType::IntegratedGPU;
                break;
            case VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU:
                mAdapterType = wgpu::AdapterType::DiscreteGPU;
                break;
            case VK_PHYSICAL_DEVICE_TYPE_CPU:
                mAdapterType = wgpu::AdapterType::CPU;
                break;
            default:
                mAdapterType = wgpu::AdapterType::Unknown;
                break;
        }

        return {};
    }

    void Adapter::InitializeSupportedExtensions() {
        if (mDeviceInfo.features.textureCompressionBC == VK_TRUE) {
            mSupportedExtensions.EnableExtension(Extension::TextureCompressionBC);
        }
    }

    ResultOrError<DeviceBase*> Adapter::CreateDeviceImpl(const DeviceDescriptor* descriptor) {
        std::unique_ptr<Device> device = std::make_unique<Device>(this, descriptor);
        DAWN_TRY(device->Initialize());
        return device.release();
    }

}}  // namespace dawn_native::vulkan
