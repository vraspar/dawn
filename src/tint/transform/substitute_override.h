// Copyright 2022 The Tint Authors.
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

#ifndef SRC_TINT_TRANSFORM_SUBSTITUTE_OVERRIDE_H_
#define SRC_TINT_TRANSFORM_SUBSTITUTE_OVERRIDE_H_

#include <string>
#include <unordered_map>

#include "src/tint/transform/transform.h"

namespace tint::transform {

/// A transform that replaces overrides with the constant values provided.
///
/// # Example
/// ```
/// override width: f32;
/// @id(1) override height: i32 = 4;
/// override depth = 1i;
/// ```
///
/// When transformed with `width` -> 1, `1` -> 22, `depth` -> 42
///
/// ```
/// const width: f32 = 1f;
/// const height: i32 = 22i;
/// const depth = 42i;
/// ```
///
/// @see crbug.com/tint/1582
class SubstituteOverride final : public Castable<SubstituteOverride, Transform> {
  public:
    /// Configuration options for the transform
    struct Config final : public Castable<Config, Data> {
        /// Constructor
        Config();

        /// Copy constructor
        Config(const Config&);

        /// Destructor
        ~Config() override;

        /// Assignment operator
        /// @returns this Config
        Config& operator=(const Config&);

        /// The map of override name (either the identifier or id) to value.
        /// The value is always a double coming into the transform and will be
        /// converted to the correct type through and initializer.
        std::unordered_map<std::string, double> map;
    };

    /// Constructor
    SubstituteOverride();

    /// Destructor
    ~SubstituteOverride() override;

    /// @param program the program to inspect
    /// @param data optional extra transform-specific input data
    /// @returns true if this transform should be run for the given program
    bool ShouldRun(const Program* program, const DataMap& data = {}) const override;

  protected:
    /// Runs the transform using the CloneContext built for transforming a
    /// program. Run() is responsible for calling Clone() on the CloneContext.
    /// @param ctx the CloneContext primed with the input program and
    /// ProgramBuilder
    /// @param inputs optional extra transform-specific input data
    /// @param outputs optional extra transform-specific output data
    void Run(CloneContext& ctx, const DataMap& inputs, DataMap& outputs) const override;
};

}  // namespace tint::transform

#endif  // SRC_TINT_TRANSFORM_SUBSTITUTE_OVERRIDE_H_
