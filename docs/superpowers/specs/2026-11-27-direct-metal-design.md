# Direct Metal with prerequisites first

## Intent and approved direction

Deliver a direct Metal GPU backend for VIctory on Apple Silicon. First address the missing shared functionality that would otherwise follow the project into the new backend. Keep the existing executable loader, console runtime, Rosetta CPU execution, and shader translator. Reach Metal from the Windows process through a Wine Unix bridge.

The user approved these four steps on 2026-11-27:

1. Keep the existing loader, console runtime, and shader translator.
2. Prove correct Metal access to game memory, including aliases and remapping.
3. Execute translated compute through DXIL, Metal Shader Converter, and Metal.
4. Extend to rendering while retaining D3DMetal for comparison.

The current deliverable is the plan. The approved direction does not require another architecture-selection discussion. Implementation proceeds with the prerequisites first; later details are settled from the hardware evidence described below.

## Current evidence

Baseline checkout: `1fcc386`. Counts describe the audit on this checkout, not permanent regression-test expectations.

| Area | Observed baseline |
| --- | --- |
| Owned game package | All 104 manifest-listed files present; 22 RPF headers valid |
| Executable | Recognized Series 1.0.2.1 build |
| Static binding | 587 executable imports resolved; four initially loaded title DLLs have no unresolved imports |
| Host capabilities | Mesh shaders and shader model 6.6 reported supported |
| Installed D3DMetal | 4.0 beta 1; repository documentation asks for 4.0 beta 3 or newer |
| Native tests | 727 passed, 18 ignored; full Wine suite was not run in the audit |
| Runtime shader translation | 25,941 distinct containers: 25,675 emitted, 266 refused |
| Refusal classes | 195 unsupported-stage containers; 71 other refusals |
| BVH builder | 49 of 52 compute containers emitted; three cross-half first-lane reads refused |

Emission was measured with the runtime cache path and its actual options in a separate, cold cache. It did not validate DXIL, create every host pipeline, or establish correct GPU results. The separate diagnostic command has different option construction and different counting units; its counts must not be substituted for this baseline.

## Approach and tradeoffs

The chosen approach is shared correctness work followed by an incremental direct Metal backend. It makes shader fixes useful to both backends and preserves a comparison implementation while the new backend grows.

Continuing only with D3DMetal offers a shorter route to individual fixes but retains its translation and memory constraints. A complete native CPU/runtime rewrite would expand the scope beyond the approved GPU work. Those alternatives are not part of this effort.

D3DMetal is a comparison path, not a semantic oracle: known differences must be checked against synthetic CPU expectations and existing console-calibrated tests. Direct Metal is not assumed to make the GPU execute Xbox wave64 instructions correctly or to understand the game's acceleration-structure representation automatically.

## Constraints

- Retain the current loader, console runtime, and Rosetta CPU execution.
- Keep D3D12/D3DMetal available as the reference backend throughout migration.
- Keep all game files, derived shaders, dumps, captures, and Apple redistributables outside the repository.
- Commit only original source, synthetic tests, build identities, and aggregate measurements allowed by `docs/asset-policy.md`.
- Use the existing Rust workspace and local Apple/Wine toolchain; add dependencies only when a required capability is unavailable locally.
- The existing macOS baseline is Apple Silicon and macOS 15 or newer. Report any higher requirement discovered for the selected Metal Shader Converter explicitly.
- Use D3DMetal 4.0 beta 3 or newer for the comparison baseline; never replace a working runtime with an unverified or partial installation.
- Keep backend selection explicit during development; an explicitly requested Metal backend must report unavailable capabilities instead of silently switching to D3DMetal.
- Missing graphics operations must fail visibly or remain counted as unsupported; they must not become successful no-ops.
- Preserve saves and original game files. Use isolated external cache/data directories for controlled game checks and record the backend and shader options with each measurement.

## Work order and gates

### A. Trustworthy baseline and missing shared functionality

First establish the acceptance workload: the prologue through first controllable movement/camera response, with a repeatable input sequence, settings, and isolated data. Initially record the furthest reachable checkpoint and repeat after fixes; a shader omitted by a refused pipeline or a held builder is not evidence of an unused shader. Trace attempted work before it is dropped, including indirect work, and keep hashes and detailed traces outside the repository.

Unify shader option selection between diagnostics and runtime, including cross-half synchronization, image atomics, narrow waves, and per-container overrides. Comparisons must also use the same identity: the length-bounded full container and the native program selected by the runtime. The diagnostic's extra program analysis remains separately labeled. Make setup detect the installed toolkit version and support a safe refresh from a user-supplied toolkit. Run the native and Wine suites and preserve the original baseline when upgrading.

Then reproduce the three BVH failures with original synthetic programs. Correct the first-lane semantics for active lanes in both halves, upper-half-only activity, divergence, early exits, loops, and partial groups. Keep the existing all-or-nothing builder gate: it prevents partial builds from damaging title memory.

Classify the other refusals by semantic family and by attempted use in the agreed game acceptance workload. Implement required stages/instructions and verify numerical behavior. A packaged shader is not proof that the selected workload executes it. Label coverage as observed/required, excluded by identified profile or stage selection, or not yet observed; absence from an incomplete run proves nothing beyond that recorded prefix. A deferred family stays in the coverage inventory with its evidence and uncertainty, never counted as implemented. Resolve known shared blockers before the Metal slice; do not require proving every future game path unused before starting the memory probe.

**Gate A:** diagnostic and runtime choices agree for identical inputs; reference runtime is version-checked; all 52 BVH containers emit and their standalone host pipelines can be made; synthetic semantic tests execute correctly; the title's own builder pipelines become ready and at least one actual builder dispatch passes its gate in the controlled run. Every refusal family has a coverage status, and known shared blockers in the observed acceptance workload are resolved. Unobserved package coverage and unexecuted paths remain explicit follow-up work for game acceptance.

### B. Minimal backend boundary and shared memory

Reuse `HostGpu`, `HostMemory`, `alias.rs`, `inplace_probe.rs`, and the XMem mapping tests. Inventory concrete D3D12 types at their callers. Move copies, barriers, and lifetime handling behind the minimum boundary needed by the memory/compute slice, preserving existing behavior. Avoid moving all roughly 200 command sites before there is a working slice.

Build the Wine-to-Metal bridge and test a Metal buffer over eligible owned memory. The in-process bridge must match the running Wine process's architecture and calling convention; an ARM64-only library cannot simply be loaded into its x86-64 host. Shader conversion may use a separate native command-line tool without moving title-memory operations out of process. Eligibility includes actual host-page alignment, one valid VM region, checked sizes, stable ownership, and device limits. Unsupported mappings must have an explicit outcome and a valid copying fallback where the slice supports one. Do not claim zero-copy for ineligible memory.

Aliases of the same physical pages must resolve to one canonical GPU buffer with offsets. The existing roadmap reports that separate GPU aliases are not necessarily coherent within a submission. Remapping and release must wait for in-flight GPU use; no stale pointers or resource destruction before completion.

**Gate B:** deterministic CPU-to-GPU-to-CPU results, aliases, remapping, unmapping, allocation failure, and repeated lifetime tests pass through the real Wine bridge on this Mac. Report bytes copied, wall-clock submission-to-completion time, and live allocation totals against D3DMetal. GPU execution timings are optional evidence only when a validated timing source exists; D3DMetal timestamps that remain zero are unavailable measurements, not zero GPU cost. A standalone native Metal demo does not satisfy this gate.

### C. One complete Metal compute path

Reuse VIctory's emitted DXIL and compile it with the locally installed Metal Shader Converter. Implement only the bindings, descriptors, pipeline creation, residency, command submission, and synchronization required by an existing synthetic compute workload. Keep converter-generated data outside the repository.

Test ordinary buffers first, then cross-wave and atomic operations required by the BVH shaders. Associate compiled artifacts with backend, translator options/build, converter version, and applicable device/OS identity so incompatible caches cannot be reused.

**Gate C:** the Windows process invokes Metal directly; an existing translated compute workload writes the correct bytes into the shared title allocation; repeat runs and cache reloads agree; required BVH pipelines compile and controlled semantic probes execute. D3DMetal must not execute the selected Metal compute work.

### D. Rendering through Metal

Extend the proven slice in testable units: resource bindings, texture formats and detiling, graphics/mesh pipelines, indirect work, depth/blend state, fences and queries, then presentation. Preserve the console behavior already encoded in `draw.rs`, `descriptor/`, `memory.rs`, `texture.rs`, and the shader translator. Query actual device limits and keep unsupported cases visible.

Integrate explicit backend selection into the existing launcher/configuration and capability checks. Validate that all GPU work for a Metal game run reaches Metal, while the CPU side continues through the retained runtime.

**Gate D:** synthetic render cases pass with CPU/reference expectations where available; resource lifetime and failure recovery remain correct; Metal presents a game frame with its backend and dropped-work counters visible. First frame is a milestone, not final acceptance.

### E. Game correctness and performance acceptance

Reuse the workload established in A with the owned Series 1.0.2.1 build and a separate save/data location. Compare equal resolution, settings, camera path, shader options, and cold/warm cache conditions. Record hardware, OS, runtime/compiler versions, translation refusals, dropped commands, GPU faults, wall-clock frame and submission/completion times, copied bytes, and peak memory. Report GPU execution time only when measurable with a validated source.

Start with the prologue through first controllable movement and camera response. Verify representative geometry, lighting, reflections, UI, and audio/input continuity. Test save/reload against the isolated data location when the game reaches that functionality. Require a 20-minute warm-cache run without a hang, GPU reset, or unbounded memory growth; identify any known visual deviation rather than hiding it in a performance number.

**Goal complete:** Direct Metal is selectable and handles the full GPU workload for the defined scene; required correctness checks pass; measured performance and memory costs are reported against D3DMetal. If gameplay remains impractically slow, report that limitation and keep performance work open. No FPS promise is made before the baseline is measured.

## Decisions left to implementation evidence

Bridge ABI and the smallest command abstraction are chosen after the memory probe, using the Wine build already present. Converter version and hardware-specific capability floors are verified locally before integration. Exact instruction-lowering changes follow synthetic reproductions of the real refusal shapes. These are engineering decisions within the approved direction, not reasons to repeat architecture approval.

## Explicit exclusions

This effort does not replace Wine/Rosetta for CPU execution, implement online services, redistribute game/toolkit material, add a launcher UI, or undertake unrelated repository refactoring. It does not require spending time optimizing D3DMetal paths that the new backend will remove unless they are needed for a reliable comparison or reveal shared correctness defects.
