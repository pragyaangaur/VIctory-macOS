# Mesh shader inputs on D3DMetal

`VICTORY_MAC_PRIMITIVE_VARYINGS=1` selects an experimental mesh/pixel
pipeline on Darwin. The mesh stage gathers the original three vertex
values into flat per-primitive outputs. The pixel stage reads those
outputs and runs its original interpolation math. Channel discovery follows
the ordinary pixel lowering, including packed f16 signature components.

The linked stages reuse the ordinary root bindings and have their own
pipeline cache keys. Stage or mesh-budget refusal and a PSO creation error
fall back to the ordinary pipeline. A mesh with a loop-limit exit that can
bypass the group barrier is refused. Vertex/pixel pipelines are outside
this path. A successful linked PSO logs `primitive_varyings=true`.

## Verification

Run `cargo test -p victory-shader --lib linked_varying_signatures_match`
on the native host. Run the Windows test `linked_varyings_hardware_readback`
with hardware D3D12 and the Microsoft DXIL validator. The D3DMetal probe
verified 512 asymmetric corner-order pixels, 512 cull and invalid-index
pixels, and 256 pixels across two triangles sharing packed f16 vertices.

A local world capture covered 87 distinct mesh/pixel pairs with
`attributeAtVertex`. The translator linked 54 pairs, covering 921 of its
1,167 affected mesh commands. All 108 emitted stages passed Microsoft's
DXIL validator and Apple's Metal IR Converter. The remaining 33 pairs
exceeded mesh output or shared-memory limits (28), exceeded signature
limits (4), or had a loop-limit exit (1). The four affected vertex/pixel
pairs are outside the prototype. Ordinary translation emitted byte-identical
DXIL for all 155 source shaders shared across the 87 pairs compared with
`35db0038`.

The same 54 pairs were also submitted to headless D3D12 pipeline creation.
Every call returned success. D3DMetal logged a fragment compilation error
and marked every ordinary pipeline as a no-op. It logged no fragment
compilation errors for the linked pipelines and marked eight as no-ops
during render pipeline creation. The other 46 account for 884 captured
mesh commands. The probe inferred root and render target state and issued
no draws. Title draw correctness and frame cost remain unmeasured.

Two natively f16 channels consume six f32 primitive components in this
prototype. A lossless packed representation could fit the originals in
three u32 components if both ends preserve the existing f16 expansion,
including signed zero and NaN behavior. Its coverage is unmeasured.
