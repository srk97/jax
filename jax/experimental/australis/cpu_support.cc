#include "jax/experimental/australis/client-private.h"
#include "tensorflow/compiler/xla/pjrt/tfrt_cpu_pjrt_client.h"

namespace aux {
namespace internal {

BackendFactoryRegister _register_cpu(
    "cpu", +[]() -> absl::StatusOr<std::shared_ptr<xla::PjRtClient>> {
      return ToAbslStatusOr(xla::GetTfrtCpuClient(false));
    });

}  // namespace internal
}  // namespace aux
