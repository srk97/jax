#include "jax/experimental/australis/internals.h"

#include "tensorflow/compiler/xla/xla_data.pb.h"

namespace aux {
namespace internal {

xla::PrimitiveType PrimitiveTypeToXla(PrimitiveType type) {
#define CASE(type)          \
  case PrimitiveType::type: \
    return xla::PrimitiveType::type
  switch (type) {
    CASE(PRIMITIVE_TYPE_INVALID);
    CASE(PRED);
    CASE(S8);
    CASE(S16);
    CASE(S32);
    CASE(S64);
    CASE(U8);
    CASE(U16);
    CASE(U32);
    CASE(U64);
    CASE(F16);
    CASE(F32);
    CASE(BF16);
    CASE(F64);
    CASE(TUPLE);
    CASE(TOKEN);
  }
#undef CASE
}

PrimitiveType PrimitiveTypeFromXla(xla::PrimitiveType type) {
#define CASE(type)               \
  case xla::PrimitiveType::type: \
    return PrimitiveType::type
  switch (type) {
    CASE(PRIMITIVE_TYPE_INVALID);
    CASE(PRED);
    CASE(S8);
    CASE(S16);
    CASE(S32);
    CASE(S64);
    CASE(U8);
    CASE(U16);
    CASE(U32);
    CASE(U64);
    CASE(F16);
    CASE(F32);
    CASE(BF16);
    CASE(F64);
    CASE(TUPLE);
    CASE(TOKEN);
    default:
      return PrimitiveType::PRIMITIVE_TYPE_INVALID;
  }
#undef CASE
}

}  // namespace internal
}  // namespace aux
