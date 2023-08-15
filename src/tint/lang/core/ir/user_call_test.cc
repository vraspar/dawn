// Copyright 2023 The Tint Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "src/tint/lang/core/ir/user_call.h"

#include "gmock/gmock.h"
#include "gtest/gtest-spi.h"
#include "src/tint/lang/core/ir/ir_helper_test.h"

namespace tint::core::ir {
namespace {

using namespace tint::core::number_suffixes;  // NOLINT
using IR_UserCallTest = IRTestHelper;

TEST_F(IR_UserCallTest, Usage) {
    auto* func = b.Function("myfunc", mod.Types().void_());
    auto* arg1 = b.Constant(1_u);
    auto* arg2 = b.Constant(2_u);
    auto* e = b.Call(mod.Types().void_(), func, Vector{arg1, arg2});
    EXPECT_THAT(func->Usages(), testing::UnorderedElementsAre(Usage{e, 0u}));
    EXPECT_THAT(arg1->Usages(), testing::UnorderedElementsAre(Usage{e, 1u}));
    EXPECT_THAT(arg2->Usages(), testing::UnorderedElementsAre(Usage{e, 2u}));
}

TEST_F(IR_UserCallTest, Results) {
    auto* func = b.Function("myfunc", mod.Types().void_());
    auto* arg1 = b.Constant(1_u);
    auto* arg2 = b.Constant(2_u);
    auto* e = b.Call(mod.Types().void_(), func, Vector{arg1, arg2});

    EXPECT_TRUE(e->HasResults());
    EXPECT_FALSE(e->HasMultiResults());
    EXPECT_TRUE(e->Result()->Is<InstructionResult>());
    EXPECT_EQ(e->Result()->Source(), e);
}

TEST_F(IR_UserCallTest, Fail_NullType) {
    EXPECT_FATAL_FAILURE(
        {
            Module mod;
            Builder b{mod};
            b.Call(nullptr, b.Function("myfunc", mod.Types().void_()));
        },
        "");
}

}  // namespace
}  // namespace tint::core::ir
