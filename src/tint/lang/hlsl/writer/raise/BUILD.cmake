# Copyright 2024 The Dawn & Tint Authors
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

################################################################################
# Target:    tint_lang_hlsl_writer_raise
# Kind:      lib
################################################################################
tint_add_target(tint_lang_hlsl_writer_raise lib
  lang/hlsl/writer/raise/binary_polyfill.cc
  lang/hlsl/writer/raise/binary_polyfill.h
  lang/hlsl/writer/raise/builtin_polyfill.cc
  lang/hlsl/writer/raise/builtin_polyfill.h
  lang/hlsl/writer/raise/decompose_storage_access.cc
  lang/hlsl/writer/raise/decompose_storage_access.h
  lang/hlsl/writer/raise/decompose_uniform_access.cc
  lang/hlsl/writer/raise/decompose_uniform_access.h
  lang/hlsl/writer/raise/fxc_polyfill.cc
  lang/hlsl/writer/raise/fxc_polyfill.h
  lang/hlsl/writer/raise/promote_initializers.cc
  lang/hlsl/writer/raise/promote_initializers.h
  lang/hlsl/writer/raise/raise.cc
  lang/hlsl/writer/raise/raise.h
  lang/hlsl/writer/raise/shader_io.cc
  lang/hlsl/writer/raise/shader_io.h
)

tint_target_add_dependencies(tint_lang_hlsl_writer_raise lib
  tint_api_common
  tint_lang_core
  tint_lang_core_common
  tint_lang_core_constant
  tint_lang_core_intrinsic
  tint_lang_core_ir
  tint_lang_core_ir_transform
  tint_lang_core_type
  tint_lang_hlsl
  tint_lang_hlsl_intrinsic
  tint_lang_hlsl_ir
  tint_lang_hlsl_type
  tint_lang_hlsl_writer_common
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

################################################################################
# Target:    tint_lang_hlsl_writer_raise_test
# Kind:      test
################################################################################
tint_add_target(tint_lang_hlsl_writer_raise_test test
  lang/hlsl/writer/raise/binary_polyfill_test.cc
  lang/hlsl/writer/raise/builtin_polyfill_test.cc
  lang/hlsl/writer/raise/decompose_storage_access_test.cc
  lang/hlsl/writer/raise/decompose_uniform_access_test.cc
  lang/hlsl/writer/raise/fxc_polyfill_test.cc
  lang/hlsl/writer/raise/promote_initializers_test.cc
  lang/hlsl/writer/raise/shader_io_test.cc
)

tint_target_add_dependencies(tint_lang_hlsl_writer_raise_test test
  tint_api_common
  tint_lang_core
  tint_lang_core_constant
  tint_lang_core_intrinsic
  tint_lang_core_ir
  tint_lang_core_ir_transform_test
  tint_lang_core_type
  tint_lang_hlsl_writer_raise
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

tint_target_add_external_dependencies(tint_lang_hlsl_writer_raise_test test
  "gtest"
)

################################################################################
# Target:    tint_lang_hlsl_writer_raise_fuzz
# Kind:      fuzz
################################################################################
tint_add_target(tint_lang_hlsl_writer_raise_fuzz fuzz
  lang/hlsl/writer/raise/promote_initializers_fuzz.cc
)

tint_target_add_dependencies(tint_lang_hlsl_writer_raise_fuzz fuzz
  tint_api_common
  tint_cmd_fuzz_ir_fuzz
  tint_lang_core
  tint_lang_core_constant
  tint_lang_core_ir
  tint_lang_core_type
  tint_lang_hlsl_writer_raise
  tint_utils_bytes
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
