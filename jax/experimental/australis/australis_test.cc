#include "jax/experimental/australis/australis.h"

#include "gtest/gtest.h"

namespace aux {
namespace {

Array CreateFloatRange(size_t n, absl::Span<int64_t const> dims) {
  auto data = std::make_shared<std::vector<float>>();
  for (size_t i = 0; i < n; ++i) {
    data->push_back(float(i));
  }
  return Array::CreateRN<float>(*data, dims, data);
}

TEST(ArrayTest, ToString) {
  Array data = CreateFloatRange(12, {2, 2, 3});

  ASSERT_EQ(data.ToString(), R"(F32[2,2,3] {
  {
    {0, 1, 2},
    {3, 4, 5}
  },
  {
    {6, 7, 8},
    {9, 10, 11}
  }
})");
}

TEST(ArrayTest, Slice) {
  Array data = CreateFloatRange(12, {2, 2, 3})[1][0];
  ASSERT_EQ(data.ToString(), "F32[3] {6, 7, 8}");
}

}  // namespace
}  // namespace aux
