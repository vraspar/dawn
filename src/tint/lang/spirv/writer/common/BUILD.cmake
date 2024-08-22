# Copyright 2023 The Dawn & Tint Authors
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

################################################################################
# File generated by 'tools/src/cmd/gen' using the template:
#   tools/src/cmd/gen/build/BUILD.cmake.tmpl
#
# To regenerate run: './tools/run gen'
#
#                       Do not modify this file directly
################################################################################

if(TINT_BUILD_SPV_WRITER)
################################################################################
# Target:    tint_lang_spirv_writer_common
# Kind:      lib
# Condition: TINT_BUILD_SPV_WRITER
################################################################################
tint_add_target(tint_lang_spirv_writer_common lib
  lang/spirv/writer/common/binary_writer.cc
  lang/spirv/writer/common/binary_writer.h
  lang/spirv/writer/common/function.cc
  lang/spirv/writer/common/function.h
  lang/spirv/writer/common/instruction.cc
  lang/spirv/writer/common/instruction.h
  lang/spirv/writer/common/module.cc
  lang/spirv/writer/common/module.h
  lang/spirv/writer/common/operand.cc
  lang/spirv/writer/common/operand.h
  lang/spirv/writer/common/option_helper.cc
  lang/spirv/writer/common/option_helpers.h
  lang/spirv/writer/common/options.h
)

tint_target_add_dependencies(tint_lang_spirv_writer_common lib
  tint_api_common
  tint_lang_core_common
  tint_utils_containers
  tint_utils_diagnostic
  tint_utils_ice
  tint_utils_macros
  tint_utils_math
  tint_utils_memory
  tint_utils_reflection
  tint_utils_result
  tint_utils_rtti
  tint_utils_text
  tint_utils_traits
)

if(TINT_BUILD_SPV_READER OR TINT_BUILD_SPV_WRITER)
  tint_target_add_external_dependencies(tint_lang_spirv_writer_common lib
    "spirv-headers"
  )
endif(TINT_BUILD_SPV_READER OR TINT_BUILD_SPV_WRITER)

endif(TINT_BUILD_SPV_WRITER)
if(TINT_BUILD_SPV_WRITER)
################################################################################
# Target:    tint_lang_spirv_writer_common_test
# Kind:      test
# Condition: TINT_BUILD_SPV_WRITER
################################################################################
tint_add_target(tint_lang_spirv_writer_common_test test
  lang/spirv/writer/common/binary_writer_test.cc
  lang/spirv/writer/common/helper_test.h
  lang/spirv/writer/common/instruction_test.cc
  lang/spirv/writer/common/module_test.cc
  lang/spirv/writer/common/operand_test.cc
  lang/spirv/writer/common/options_test.cc
  lang/spirv/writer/common/spv_dump_test.cc
  lang/spirv/writer/common/spv_dump_test.h
)

tint_target_add_dependencies(tint_lang_spirv_writer_common_test test
  tint_api_common
  tint_lang_core
  tint_lang_core_constant
  tint_lang_core_intrinsic
  tint_lang_core_ir
  tint_lang_core_type
  tint_utils_containers
  tint_utils_diagnostic
  tint_utils_ice
  tint_utils_id
  tint_utils_macros
  tint_utils_math
  tint_utils_memory
  tint_utils_reflection
  tint_utils_result
  tint_utils_rtti
  tint_utils_symbol
  tint_utils_text
  tint_utils_traits
)

tint_target_add_external_dependencies(tint_lang_spirv_writer_common_test test
  "gtest"
)

if(TINT_BUILD_SPV_READER OR TINT_BUILD_SPV_WRITER)
  tint_target_add_external_dependencies(tint_lang_spirv_writer_common_test test
    "spirv-headers"
    "spirv-tools"
  )
endif(TINT_BUILD_SPV_READER OR TINT_BUILD_SPV_WRITER)

if(TINT_BUILD_SPV_WRITER)
  tint_target_add_dependencies(tint_lang_spirv_writer_common_test test
    tint_lang_spirv_writer
    tint_lang_spirv_writer_common
  )
endif(TINT_BUILD_SPV_WRITER)

endif(TINT_BUILD_SPV_WRITER)