#include "jax/experimental/australis/client-private.h"
#include "tensorflow/compiler/xla/pjrt/gpu_device.h"

namespace aux {
namespace internal {
namespace {

BackendFactoryRegister _register_gpu(
    "gpu", +[]() -> absl::StatusOr<std::shared_ptr<xla::PjRtClient>> {
      xla::GpuAllocatorConfig config;
      config.preallocate = false;
      return ToAbslStatusOr(
          xla::GetGpuClient(true, config, nullptr, /*node_id=*/0));
    });

}  // namespace
}  // namespace internal
}  // namespace aux
