#include "jax/experimental/australis/client-private.h"
#include "tensorflow/compiler/xla/pjrt/tpu_client.h"

namespace aux {
namespace internal {
namespace {

BackendFactoryRegister _register_cloud_tpu(
    "cloud_tpu", +[]() -> absl::StatusOr<std::shared_ptr<xla::PjRtClient>> {
      return ToAbslStatusOr(xla::GetTpuClient(32));
    });

}  // namespace
}  // namespace internal
}  // namespace aux
