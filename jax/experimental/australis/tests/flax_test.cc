#include "testing/base/public/gmock.h"
#include "gtest/gtest.h"
#include "absl/status/status.h"
#include "jax/experimental/australis/australis.h"
#include "jax/experimental/australis/petri.h"
#include "jax/experimental/australis/tests/flax_jit.h"

namespace aux {
namespace {

TEST(Flax, OptimizerStep) {
  ASSERT_OK_AND_ASSIGN(auto client, Client::GetDefault());
  auto dev = client.LocalDevices()[0];
  ASSERT_OK_AND_ASSIGN(auto init_fn, australis::test::FlaxInit::Load(client));
  ASSERT_OK_AND_ASSIGN(auto optimizer_step_fn,
                       australis::test::FlaxOptimizerStep::Load(client));

  ASSERT_OK_AND_ASSIGN(auto optimizer_weights, init_fn());

  const float inputs[] = {0, 1, 2, 3, 4, 5, 6, 7};
  ASSERT_OK_AND_ASSIGN(auto x, PTree::BufferRN<float>(inputs, {2, 4}, dev));
  ASSERT_OK_AND_ASSIGN(optimizer_weights,
                       optimizer_step_fn(optimizer_weights, x));

  // TODO(parkers): Properly encode the flax weights structure (or an
  // approximation of it) into the new executable proto type.
  EXPECT_EQ(
      "((Buffer(), (((((Buffer(), Buffer()), (Buffer(), Buffer())), "
      "((Buffer(), Buffer()), (Buffer(), Buffer())), ((Buffer(), Buffer()), "
      "(Buffer(), Buffer())))))), ((((Buffer(), Buffer()), (Buffer(), "
      "Buffer()), (Buffer(), Buffer())))))",
      optimizer_weights.ToString());
}

}  // namespace
}  // namespace aux
