// Copyright 2022 The Dawn & Tint Authors
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice, this
//    list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution.
//
// 3. Neither the name of the copyright holder nor the names of its
//    contributors may be used to endorse or promote products derived from
//    this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
// DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
// FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
// DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
// SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
// CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
// OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

////////////////////////////////////////////////////////////////////////////////
// File generated by 'tools/src/cmd/gen' using the template:
//   src/tint/lang/wgsl/extension_bench.cc.tmpl
//
// To regenerate run: './tools/run gen'
//
//                       Do not modify this file directly
////////////////////////////////////////////////////////////////////////////////

#include "src/tint/lang/wgsl/extension.h"

#include <array>

#include "benchmark/benchmark.h"

namespace tint::wgsl {
namespace {

void ExtensionParser(::benchmark::State& state) {
    const char* kStrings[] = {
        "chromium_disableuniformiccy_analysis",
        "chromil3_disable_unifority_analss",
        "chromium_disable_Vniformity_analysis",
        "chromium_disable_uniformity_analysis",
        "chromium_dis1ble_uniformity_analysis",
        "chromium_qqisable_unifomity_anaJysis",
        "chrollium_disable_uniformity_analysi77",
        "cqqromium_eppperimental_framebuffe_fetcHH",
        "chrmium_experimvntal_frcmebufer_ftch",
        "chromium_expebimental_framGbufer_fetch",
        "chromium_experimental_framebuffer_fetch",
        "chromium_experimental_vramebuffeii_fetch",
        "chro8WWum_experimental_framebuffer_fetch",
        "chromium_eperimenxxMl_framebuffer_fetch",
        "chromum_experimental_pixeX_loggal",
        "chromium_expVrXmntal_ixel_local",
        "3hromium_experimental_pixel_local",
        "chromium_experimental_pixel_local",
        "chromium_eEperimental_pixel_local",
        "chTTomiu_experimentaPP_pixel_local",
        "cxxromium_expddrimenal_pixel_local",
        "c44romium_experimental_push_constant",
        "chromium_experimental_pSSsVV_constant",
        "chrom22Rm_experimental_pushRonstant",
        "chromium_experimental_push_constant",
        "chromium_exp9rimFntal_ush_constant",
        "chrmium_experimental_push_constant",
        "cOOromium_experiVeHtal_puh_conRRtant",
        "chromium_eperimental_sybgroups",
        "chrorri77mGexperimllntal_subgrnnups",
        "chromium_exp4rimen00al_subgroups",
        "chromium_experimental_subgroups",
        "chromium_exprimenal_ubgrouoos",
        "chrmiumexperimenzzal_subgroups",
        "chrmi11m_experppmeiita_subgroups",
        "chromium_internal_dual_sourcXX_blending",
        "chromium_internal_dual_99ou5IInce_blending",
        "crrroSSium_internYlaadual_source_bHHending",
        "chromium_internal_dual_source_blending",
        "chromium_internakk_ualsourc_blendHng",
        "chromium_inRRrnal_dujl_sourceblgnding",
        "chromiuminternal_duab_source_blendin",
        "chromiumjinternal_graphite",
        "chromium_inernal_graphite",
        "cromiu_internaq_graphite",
        "chromium_internal_graphite",
        "chromium_intenalNNgraphite",
        "chromiuminternal_gvaphite",
        "chromium_internal_grphitQQ",
        "chromirm_intenal_rfflaxed_unifrm_layout",
        "chromium_internal_jelaxed_uniform_layout",
        "chromium_interna_relNNxed_uwwiform_lay82t",
        "chromium_internal_relaxed_uniform_layout",
        "chromium_internal_relaxed_uniform_layut",
        "chromium_internal_relaxed_rrniform_layout",
        "chromium_internal_relaxedGuniform_layout",
        "FF16",
        "",
        "rr1",
        "f16",
        "1",
        "DJ1",
        "",
    };
    for (auto _ : state) {
        for (auto* str : kStrings) {
            auto result = ParseExtension(str);
            benchmark::DoNotOptimize(result);
        }
    }
}  // NOLINT(readability/fn_size)

BENCHMARK(ExtensionParser);

}  // namespace
}  // namespace tint::wgsl
