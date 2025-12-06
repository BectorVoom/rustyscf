use cubecl::channel::MutexComputeChannel;
use cubecl::prelude::*;
use cubecl_core::WgpuCompilationOptions;
cubecl_runtime::memory_management::MemoryDeviceProperties;
use cubecl_wgpu::{MemoryConfiguration, WgpuDevice, WgpuServer};
use std::mem;

#[repr(u8)]
pub enum Backend {
    Noop = 0,
    Vulkan = 1,
    Metal = 2,
    Dx12 = 3,
    Gl = 4,
    BrowserWebGpu = 5,
}

fn main() {
    let device = WgpuDevice::default();
    let template_client = cubecl_wgpu::WgpuRuntime::client(&device);

    let memory_config = MemoryConfiguration::default();
    let compilation_options = WgpuCompilationOptions::default();
    let wgpu_device = unsafe { mem::zeroed() };
    let queue = unsafe { mem::zeroed() };
    let tasks_max = 256;
    let backend = Backend::Vulkan;
    let timing_method = unsafe { mem::zeroed() };

    let server = WgpuServer::new(
        memory_properties,
        memory_config,
        compilation_options,
        wgpu_device,
        queue,
        tasks_max,
        backend,
        timing_method,
    );

    let channel = MutexComputeChannel::new(server);
    let properties = unsafe { std::ptr::read(template_client.properties()) };
    let info = template_client.info().clone();

    let _new_client = ComputeClient::new(channel, properties, info);
}
