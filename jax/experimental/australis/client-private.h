#ifndef JAX_EXPERIMENTAL_AUSTRALIS_CLIENT_PRIVATE_H_
#define JAX_EXPERIMENTAL_AUSTRALIS_CLIENT_PRIVATE_H_

#include <string_view>

#include "absl/status/statusor.h"
#include "tensorflow/compiler/xla/pjrt/pjrt_client.h"

namespace aux {
namespace internal {

absl::Status ToAbslStatus(const xla::Status& s);

template <typename T>
absl::StatusOr<T> ToAbslStatusOr(xla::StatusOr<T> s) {
  if (s.ok()) {
    return absl::StatusOr<T>(std::move(s.ValueOrDie()));
  }
  return ToAbslStatus(s.status());
}

using FactoryFn = absl::StatusOr<std::shared_ptr<xla::PjRtClient>> (*)();

class BackendFactoryRegister {
 public:
  BackendFactoryRegister(const char* name, FactoryFn factory);
};

}  // namespace internal
}  // namespace aux

#endif  // JAX_EXPERIMENTAL_AUSTRALIS_CLIENT_PRIVATE_H_
