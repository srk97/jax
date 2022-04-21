#include <iostream>
#include <optional>

#include "base/init_google.h"
#include "testing/base/public/gmock.h"
#include "gtest/gtest.h"
#include "absl/status/status.h"
#include "jax/experimental/australis/australis.h"
#include "jax/experimental/australis/petri.h"
#include "jax/experimental/australis/tests/donate_arg_jit.h"
#include "jax/experimental/australis/tests/higher_arity_jit.h"
#include "jax/experimental/australis/tests/multi_return_jit.h"
#include "jax/experimental/australis/tests/tuple_jit.h"

namespace {

using PrimitiveType = aux::PrimitiveType;
using PTree = aux::PTree;
using aux::Client;

const float inputs[] = {1, 2, 3, 4, 5, 6, 7, 8, 9};

TEST(Jit, Tuple) {
  ASSERT_OK_AND_ASSIGN(auto client, Client::GetDefault());
  ASSERT_OK_AND_ASSIGN(auto computation,
                       australis::test::TupleJit::Load(client));
  auto dev = client.LocalDevices()[0];
  ASSERT_OK_AND_ASSIGN(
      auto lhs,
      PTree::BufferRN<float>(inputs, {2, 3}, dev));
  ASSERT_OK_AND_ASSIGN(
      auto rhs,
      PTree::BufferRN<float>(inputs, {2, 3}, dev));
  auto tree = PTree::Tuple(std::move(lhs), std::move(rhs));

  ASSERT_OK_AND_ASSIGN(auto results, computation(tree));

  ASSERT_OK_AND_ASSIGN(auto lit, results.ToArray());

  EXPECT_EQ(3, lit.data<float>()[0]);
  EXPECT_EQ(6, lit.data<float>()[1]);
  EXPECT_EQ(9, lit.data<float>()[2]);
  EXPECT_EQ(12, lit.data<float>()[3]);
  EXPECT_EQ(15, lit.data<float>()[4]);
  EXPECT_EQ(18, lit.data<float>()[5]);
}

TEST(Jit, HigherArity) {
  ASSERT_OK_AND_ASSIGN(auto client, Client::GetDefault());
  ASSERT_OK_AND_ASSIGN(auto computation,
                       australis::test::HigherArityJit::Load(client));
  auto dev = client.LocalDevices()[0];
  ASSERT_OK_AND_ASSIGN(
      auto lhs,
      PTree::BufferRN<float>(inputs, {2, 3}, dev));
  ASSERT_OK_AND_ASSIGN(
      auto rhs,
      PTree::BufferRN<float>(inputs, {2, 3}, dev));
  auto tuple = PTree::Tuple(std::move(lhs), std::move(rhs));
  ASSERT_OK_AND_ASSIGN(
      auto y,
      PTree::BufferRN<float>({3}, {}, dev));
  ASSERT_OK_AND_ASSIGN(auto results, computation(tuple, y));

  ASSERT_OK_AND_ASSIGN(auto lit, results.ToArray());

  EXPECT_EQ(4, lit.data<float>()[0]);
  EXPECT_EQ(8, lit.data<float>()[1]);
  EXPECT_EQ(12, lit.data<float>()[2]);
  EXPECT_EQ(16, lit.data<float>()[3]);
  EXPECT_EQ(20, lit.data<float>()[4]);
  EXPECT_EQ(24, lit.data<float>()[5]);
}

TEST(Jit, MultiReturn) {
  ASSERT_OK_AND_ASSIGN(auto client, Client::GetDefault());
  ASSERT_OK_AND_ASSIGN(auto computation,
                       australis::test::MultiReturnJit::Load(client));
  auto dev = client.LocalDevices()[0];

  ASSERT_OK_AND_ASSIGN(
      auto lhs,
      PTree::BufferRN<float>(inputs, {2, 3}, dev));
  ASSERT_OK_AND_ASSIGN(
      auto rhs,
      PTree::BufferRN<float>(inputs, {2, 3}, dev));
  auto tuple = PTree::Tuple(std::move(lhs), std::move(rhs));
  ASSERT_OK_AND_ASSIGN(
      auto y,
      PTree::BufferRN<float>({3}, {}, dev));
  ASSERT_OK_AND_ASSIGN(auto results,
                       PTree::DestructureTuple(computation(tuple, y)));
  EXPECT_EQ(results.size(), 3);

  EXPECT_EQ(3, results[0].Elements()->size());
  // TODO(saeta): check results[0].
  ASSERT_OK_AND_ASSIGN(auto y_plus_one, results[1].ToArray());
  EXPECT_EQ(1, y_plus_one.data<float>().size());
  EXPECT_EQ(4, y_plus_one.data<float>()[0]);

  EXPECT_EQ(2, results[2].Elements()->size());
  // TODO(saeta): check x.
}

TEST(Jit, DonateArg) {
  ASSERT_OK_AND_ASSIGN(auto client, Client::GetDefault());
  ASSERT_OK_AND_ASSIGN(auto computation,
                       australis::test::DonateArgJit::Load(client));
  auto dev = client.LocalDevices()[0];

  ASSERT_OK_AND_ASSIGN(
      auto lhs,
      PTree::BufferRN<float>(inputs, {2, 3}, dev));
  ASSERT_OK_AND_ASSIGN(
      auto rhs,
      PTree::BufferRN<float>(inputs, {2, 3}, dev));
  auto tuple = PTree::Tuple(std::move(lhs), std::move(rhs));
  ASSERT_OK_AND_ASSIGN(
      auto y,
      PTree::BufferRN<float>({3}, {}, dev));

  ASSERT_OK_AND_ASSIGN(auto results, computation(tuple, std::move(y)));
  ASSERT_OK_AND_ASSIGN(auto lit, results.ToArray());

  EXPECT_EQ(1, lit.data<float>().size());
  EXPECT_EQ(84, lit.data<float>()[0]);
}

}  // namespace
