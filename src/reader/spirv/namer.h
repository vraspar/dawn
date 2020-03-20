// Copyright 2020 The Tint Authors.
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

#ifndef SRC_READER_SPIRV_NAMER_H_
#define SRC_READER_SPIRV_NAMER_H_

#include <cstdint>
#include <string>
#include <unordered_map>

#include "src/reader/spirv/fail_stream.h"

namespace tint {
namespace reader {
namespace spirv {

/// A Namer maps SPIR-V IDs to strings.
class Namer {
 public:
  /// Creates a new namer
  /// @param fail_stream the error reporting stream
  explicit Namer(const FailStream& fail_stream);
  /// Destructor
  ~Namer();

  /// Registers a failure.
  /// @returns a fail stream to accumulate diagnostics.
  FailStream& Fail() { return fail_stream_.Fail(); }

  /// @param id the SPIR-V ID
  /// @returns true if we the given ID already has a registered name.
  bool HasName(uint32_t id) {
    return id_to_name_.find(id) != id_to_name_.end();
  }

  /// @param id the SPIR-V ID
  /// @returns the name for the ID. It must have been registered.
  const std::string& GetName(uint32_t id) {
    return id_to_name_.find(id)->second;
  }

  /// Records a mapping from the given ID to a name. Emits a failure
  /// if the ID already has a registered name.
  /// @param id the SPIR-V ID
  /// @param name the name to map to the ID
  /// @returns true if the ID did not have a previously registered name.
  bool SaveName(uint32_t id, const std::string& name);

 private:
  FailStream fail_stream_;

  // Maps an ID to its registered name.
  std::unordered_map<uint32_t, std::string> id_to_name_;
};

}  // namespace spirv
}  // namespace reader
}  // namespace tint

#endif  // SRC_READER_SPIRV_NAMER_H_
