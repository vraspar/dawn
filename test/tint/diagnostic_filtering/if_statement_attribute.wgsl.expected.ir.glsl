SKIP: FAILED

<dawn>/test/tint/diagnostic_filtering/if_statement_attribute.wgsl:8:14 warning: 'dpdx' must only be called from uniform control flow
  } else if (dpdx(1.0) > 0)  {
             ^^^^^^^^^

<dawn>/test/tint/diagnostic_filtering/if_statement_attribute.wgsl:7:3 note: control flow depends on possibly non-uniform value
  if (x > 0) {
  ^^

<dawn>/test/tint/diagnostic_filtering/if_statement_attribute.wgsl:7:7 note: user-defined input 'x' of 'main' may be non-uniform
  if (x > 0) {
      ^

<dawn>/src/tint/lang/glsl/writer/printer/printer.cc:585 internal compiler error: Switch() matched no cases. Type: tint::core::type::Sampler
********************************************************************
*  The tint shader compiler has encountered an unexpected error.   *
*                                                                  *
*  Please help us fix this issue by submitting a bug report at     *
*  crbug.com/tint with the source program that triggered the bug.  *
********************************************************************

tint executable returned error: signal: trace/BPT trap
