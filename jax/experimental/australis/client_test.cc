#include "jax/experimental/australis/client.h"

#include "gtest/gtest.h"
#include "jax/experimental/australis/australis.h"

namespace aux {
namespace {

Array CreateFloatRange(size_t n, absl::Span<int64_t const> dims) {
  auto data = std::make_shared<std::vector<float>>();
  for (size_t i = 0; i < n; ++i) {
    data->push_back(float(i));
  }
  return Array::CreateRN<float>(*data, dims, data);
}

TEST(ArrayTest, CreateFromArray) {
  auto device = Client::GetDefault()->LocalDevices()[0];
  Array data =
      (*DeviceArray::Create({CreateFloatRange(12, {2, 2, 3})}, {device})
            ->ToArrays())[0];
  ASSERT_EQ(data.ToString(), CreateFloatRange(12, {2, 2, 3}).ToString());
}

}  // namespace
}  // namespace aux
