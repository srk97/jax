#ifndef JAX_EXPERIMENTAL_AUSTRALIS_INTERNALS_H_
#define JAX_EXPERIMENTAL_AUSTRALIS_INTERNALS_H_

#include "jax/experimental/australis/australis.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"

namespace aux {
namespace internal {

xla::PrimitiveType PrimitiveTypeToXla(PrimitiveType type);
PrimitiveType PrimitiveTypeFromXla(xla::PrimitiveType type);

}  // namespace internal
}  // namespace aux

#endif  // JAX_EXPERIMENTAL_AUSTRALIS_INTERNALS_H_
