# Direct Metal Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement the prerequisite tasks in order. Use superpowers:subagent-driven-development only if delegated execution is subsequently selected. Steps use checkbox syntax for tracking.

**Goal:** Deliver the approved direct Metal GPU path after fixing its shared correctness and tooling prerequisites.

**Architecture:** Retain the loader, HLE runtime, Rosetta, and Xbox-to-DXIL translator. Introduce direct Metal through a Wine Unix bridge, starting with shared memory and compute. Keep D3DMetal available for comparison throughout.

**Tech Stack:** Existing Rust workspace, CrossOver Wine, Rosetta, Objective-C/C Metal bridge where required, Apple Metal Shader Converter, synthetic native and Wine GPU tests.

**Spec:** [Direct Metal with prerequisites first](../specs/2026-11-27-direct-metal-design.md).

## Global Constraints

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

## Review Focus

1. Divergent lanes, upper-half-only activity, partial groups, and early exits must preserve first-active-lane behavior without deadlocking: Task 3.
2. Diagnostics, runtime overrides, and cache/library identities must choose compatible shader semantics: Task 1.
3. Missing, older, malformed, or partially copied toolkit installations must leave the previous runtime intact: Task 2.
4. Aliased or remapped memory still referenced by a GPU submission must remain valid and coherent: Milestone B's required tests.
5. Compiler failures, device limits, unsupported stages, and GPU timeouts must remain observable and must not be reported as correct rendering: Tasks 3–4 and Milestones C–E.

## Delivery structure

Tasks 1–4 are the first work package. Later milestones define the approved destination, dependencies, owned files, and acceptance gates. Write their executable task plans after the preceding hardware gate establishes the required interfaces; do not invent a bridge ABI or promise timings before that evidence exists. This is sequencing, not a reduction of the Direct Metal goal.

Execute inline by default, preserving a small reviewable change per semantic fix. A commit is an integration boundary after its checks pass; no push, merge, or publication is part of this plan. Avoid unrelated formatting and do not repeat full suites after every small edit without a reason.

Before the first source edit, record the current revision/configuration and run both baseline suites from the verification section. Label results from the inherited beta-1 runtime accordingly; Task 2 establishes the qualifying comparison baseline. Preserve unrelated local changes and use the repository's isolation workflow when implementation starts.

## File map

| Area | Existing files to reuse or modify | Responsibility |
| --- | --- | --- |
| Shader options/diagnostics | `crates/victory-gpu/src/shader_cache.rs`, `apps/victory/src/translate.rs`, `apps/victory/tests/translate.rs` | One option policy and accurate diagnostic outcomes |
| Cache consumers | `crates/victory-gpu/src/pipeline.rs`, `crates/victory-gpu/src/pipeline/library.rs` | Preserve linked-shader behavior and cache compatibility |
| Runtime setup | `setup.sh`, `tools/macos/victory.sh`, `docs/macos.md` | Version checks and safe toolkit refresh |
| Wave64 correctness | `crates/victory-shader/src/emit/lower/halves.rs`, `lower/meet.rs`, `lower.rs`, `emit/mod.rs`, `emit/halves_tests.rs` | Correct lane communication and control flow |
| Other shader gaps | `crates/victory-shader/src/native.rs`, `resource.rs`, `emit/mod.rs`, `emit/lower.rs`, existing stage/instruction tests | Stage selection, operands, descriptors, and instructions |
| Builder integrity | `crates/victory-gpu/src/pipeline/builder.rs`, `pipeline/host.rs`, `draw.rs` | Preserve complete-builder gating and visible failures |
| Memory/backend | `crates/victory-gpu/src/backend.rs`, `host.rs`, `alias.rs`, `inplace_probe.rs`, `shadow.rs`, `memory.rs`; `crates/victory-hle/src/modules/xmem.rs` | Minimal backend boundary, allocation identity, aliases, lifetime |
| Rendering | `crates/victory-gpu/src/descriptor/`, `pipeline/`, `texture.rs`, `draw.rs`, `list.rs`, `queue.rs`, `dxgi.rs` | Binding, graphics commands, synchronization, presentation |

Potential bridge files are scoped to `tools/macos/metal/` for the Wine/native build and `crates/victory-gpu/src/metal/` for Rust integration. Create only the files needed by Milestone B's approved probe; choose the exact module split from its ABI and existing build conventions.

## Task 1: Make diagnostics use runtime shader choices

**Files:** modify `crates/victory-gpu/src/shader_cache.rs` and `apps/victory/src/translate.rs`; extend their unit tests and `apps/victory/tests/translate.rs`. Read all `runtime_options()` consumers in `pipeline.rs` and `pipeline/library.rs` before changing shared construction. For the baseline, reuse diagnostics in `draw.rs`, `pipeline.rs`, and `pipeline/builder.rs`; modify those only if the required attempted-work evidence is currently missing.

**Interfaces:** retain the existing `runtime_options() -> victory_shader::emit::Options` contract for pipeline consumers. Expose one shared entry point `pub fn translation_options(container: &[u8], host_wave32: bool) -> victory_shader::emit::Options` for runtime translation and CLI diagnostics. The supplied wave capability must avoid requiring a GPU merely to perform an offline scan. Move common option assembly into shared code; do not copy its field list into the CLI. Apply existing per-container overrides in this entry point.

- [ ] Establish the repeatable acceptance sequence and isolated data before fixes: prologue through first controllable movement/camera response. Capture the furthest reachable checkpoint, requested settings, and attempted work, including refused/dropped pipelines, indirect work, and BVH gate decisions. Reuse existing diagnostics; add only missing pre-drop counters needed to distinguish attempted work from successfully executed work. Store detailed traces outside the repository.
- [ ] Add `diagnostic_options_match_runtime_translation`: for identical container, host capability, and environment, compare the complete `Options`, including rendezvous, image-atomic settings, narrow waves, literal halves, and selective UAV-size behavior. Run environment-dependent cases in child processes to avoid `OnceLock` state leaking across cases.
- [ ] Add a synthetic divergent-wave regression that exposes the current diagnostic/runtime difference and a test that `--wave32` analysis works without an available GPU.
- [ ] Run the new tests before the change and record the failure showing the mismatch.
- [ ] Route `host_shader` and diagnostic emission through the shared entry point. Preserve linked mesh/pixel overrides and the current cache/library key inputs; add key-invalidation coverage for every newly shared setting that changes emitted bytes.
- [ ] Add `same_payload_containers_keep_distinct_overrides`: construct two original synthetic containers with the same DXIL/ACCC payload and different full-container hashes, target one with a selective override, and assert that each retains its own option/result. Compare using the full length-bounded container hash and the runtime-selected native program (`emit::prepare`), rather than the diagnostic's payload-only deduplication. Cover a container with multiple native programs and one containing no supported runtime stage; keep extra diagnostic programs separately counted.
- [ ] Keep program counts and distinct-container counts explicitly separate. A successful IR build must not be presented as successful host shader execution. Preserve named refusal reporting through `translate --list`.
- [ ] Run `cargo test -p victory -p victory-gpu --locked` for native coverage, then `tools/macos/test-wine.sh -p victory --test translate -- --nocapture` with D3DMetal overrides as described below. Run the affected shader-cache/library tests under Wine.
- [ ] Repeat the owned-file diagnostic and runtime-cache scans with identical options and a cold external cache. Reconcile outcomes for the same containers/stages, not only total counts. Record aggregate results outside the repository.
- [ ] Review the change and commit the option/diagnostic fix as one unit after checks pass.

**Done:** identical requested translation semantics no longer depend on which CLI path was used; cache identities and offline analysis behavior remain correct.

## Task 2: Establish a versioned, recoverable reference runtime

**Files:** modify `setup.sh`, `tools/macos/victory.sh`, and `docs/macos.md`. Create a small reusable version helper at `tools/macos/gptk-version.sh` and its assert-based shell check at `tools/macos/tests/gptk-version.sh` if sharing the parser between setup and launcher requires it.

**Interfaces:** a shell function `gptk_version_supported VERSION` returns success for recognized versions at or above `4.0b3`, failure for older or malformed versions. Read the installed framework's version with `plutil`. Keep `GPTK_DMG` as the existing explicit input for refresh; do not add another installer configuration mechanism.

- [ ] Add cases rejecting `3.0`, `4.0b1`, `4.0b2`, empty and malformed values; accepting `4.0b3`, `4.0b10`, `4.0`, and `4.1`. Verify an unknown spelling produces an actionable error rather than an optimistic success.
- [ ] Demonstrate the current issue with a synthetic existing framework marked `4.0b1`: setup must not call that version satisfactory simply because the directory exists.
- [ ] Make setup report the installed version and refresh from an explicitly selected valid toolkit. Stage and validate the replacement redistributable set before replacing installed components; preserve the previous set on extraction, validation, or copy failure. Keep cache and save directories untouched.
- [ ] Test a failed staged copy, an older input toolkit, and a repeat run on an accepted installation using temporary synthetic directory trees. Assert the installed sentinel files remain unchanged on failure and no Wine rebuild occurs on a normal accepted repeat.
- [ ] Run `zsh -n setup.sh`, `sh -n tools/macos/victory.sh`, and `sh tools/macos/tests/gptk-version.sh`; run the staging failure checks. Do not redistribute or commit Apple files.
- [ ] With a user-supplied qualifying toolkit available, refresh locally, inspect the installed version, and run `tools/macos/victory.sh gpu-check`. If the download requires Apple sign-in, record that concrete external dependency and continue source work that does not require it.
- [ ] Establish both native and Wine suite baselines using the commands below. Record every skip and any existing failure separately from regressions.
- [ ] Review and commit setup, tests, and documentation together.

**Done:** the comparison runtime meets the documented version floor, setup detects stale installations, and failed refreshes preserve the previous runtime. Source work can proceed while a licensed download is unavailable; hardware comparison claims cannot.

## Task 3: Fix the three BVH translation blockers

**Files:** start in `crates/victory-shader/src/emit/lower/halves.rs`, `lower/meet.rs`, `lower.rs`, and `emit/halves_tests.rs`. Change `emit/mod.rs` or `crates/victory-gpu/src/pipeline/builder.rs` only when the traced defect requires it. Keep all owned shader inputs and detailed disassembly outside the repository.

**Interfaces:** preserve `emit::translate_with(container: &[u8], options: Options) -> Result<HostShader, ShaderError>`, `Lower::first_lane_across`, and the builder's all-created-pipelines-ready contract. Reuse `readfirstlane_in_branch` and the existing synthetic GPU comparison helpers rather than introducing another harness.

- [ ] Reproduce the three refusals through Task 1's shared runtime options and identify their control-flow shapes. Search callers of the lowering operation before editing it.
- [ ] Add original synthetic cases named `bvh_first_lane_shapes_lower` and `bvh_first_lane_shapes_match_console_on_gpu`. Assertions: the first active lane across the whole console wave wins; upper-half-only activity reads the upper half; inactive lanes do not supply values; different waves/groups do not exchange values.
- [ ] Exercise group sizes 32, 64, 96, and 128; active lanes at 0, 31, 32, and 63; divergence, unequal loop progress, early exits, and empty masks. Use an explicit CPU oracle for each case and run GPU cases in a bounded child process so a deadlock fails the test.
- [ ] Verify the synthetic regressions fail for the same reason before the fix. Implement the smallest sound change to uniformity analysis, exchange placement, or rendezvous handling supported by that evidence. Do not weaken a refusal without establishing the required lane semantics.
- [ ] Run `cargo test -p victory-shader halves_tests --locked`, then `tools/macos/test-wine.sh -p victory-shader halves_tests -- --nocapture` with D3DMetal overrides. A skipped hardware test is not a semantic pass.
- [ ] Run the owned BVH container scan and GPU pipeline creation check: require 52/52 emitted and 52/52 pipelines accepted. Validate emitted DXIL when a compatible validator is installed; record absence rather than counting it as success.
- [ ] In the controlled acceptance sequence using isolated external data, require a positive observation of the title's own builder pipelines becoming ready and an actual dispatch passing the builder gate. Standalone CLI pipelines use different creation state and cannot substitute for this evidence. Keep the whole-builder gate and its negative test: a refused pipeline must still hold it back.
- [ ] Review and commit each independently testable semantic fix with its regression.

**Done:** all 52 builder shaders translate and create standalone host pipelines; synthetic behavior is correct on the GPU; the title's own builder reaches ready state and dispatches in the controlled run; builder integrity checks remain enforced. This gate alone does not prove correct lighting throughout the game.

## Task 4: Close the remaining required shader gaps

**Files:** use the file map and the refusal's existing module. Reuse `emit/flow_tests.rs`, `emit/halves_tests.rs`, `emit/valu_tests.rs`, `emit/semantics_tests.rs`, and resource/stage tests as appropriate. Avoid a new generic shader subsystem.

**Interfaces:** preserve the translator's `Result<HostShader, ShaderError>` boundary and named failure categories. Extend stage representation and GPU pipeline consumers together when adding a stage; a translator-only success is incomplete.

- [ ] Rebuild the refusal inventory after Tasks 1–3. The original 266 includes the three BVH failures, so do not add them twice. Separate unsupported stages (195 initially) from cross-wave/mask/control-flow, floating-point conversion, gather-level, and descriptor failures (71 initially).
- [ ] Record which stage families and semantic shapes are present and which are attempted by the acceptance sequence established in Task 1. Include suppressed, refused, indirect, and builder-gated work. Assign each family a status: observed/required, excluded by identified profile or stage selection, or not yet observed. Incomplete traces cannot prove a shader unused. Keep unobserved families and remaining uncertainty visible without making all future game paths a prerequisite to the Metal memory probe.
- [ ] Before implementing each family, write a small task with its exact failing synthetic case, affected caller/pipeline paths, and expected CPU or console-calibrated output. Check that test fails first. Avoid prescribing a floating-point or synchronization algorithm before identifying the required console semantics.
- [ ] Implement required families in this order: shared BVH/lighting wave behavior, masks and loops, descriptors, conversions/gathers, then required additional stages. Change dependent formats, exports, cache keys, and pipeline creation together.
- [ ] Validate integers exactly. For floating-point cases, specify NaN, infinity, signed-zero, rounding, and tolerance behavior from the existing hardware-calibrated conventions; do not select tolerances merely to make results pass.
- [ ] Run focused native and Wine GPU tests per family. Re-run the runtime scan from a cold external cache at integration boundaries, with count and reason changes recorded.
- [ ] Finish Gate A: all 52 standalone BVH pipelines accepted and positive title-builder dispatch evidence recorded; known shared blockers in the observed workload resolved; every remaining refusal classified; both complete suites checked. Carry unobserved/unexecuted paths forward explicitly to game acceptance. Preserve visible failure behavior for still-unsupported inputs.
- [ ] Review and commit each family independently. Update the milestone record with measured coverage and remaining limitations.

**Done:** missing shared functionality needed for the planned Metal workload is fixed and tested. Reaching zero refusals without correct execution does not satisfy this task.

## Milestone B: Backend boundary and shared Metal memory

**Depends on:** Gate A. **Approved steps:** 1 and 2.

- [ ] Read `backend.rs`, `HostList`, `alias.rs`, `inplace_probe.rs`, and XMem mapping lifetimes together; trace all affected callers before defining the bridge ABI.
- [ ] Write the executable memory-slice plan with exact Rust/native interfaces and ownership rules. Match the actual Wine host architecture/ABI and test packing/calling conventions across the bridge. Reuse the existing alias/probe findings; move only copies, barriers, and resource lifetime operations needed by this slice.
- [ ] Implement an end-to-end Wine bridge probe over eligible game-style allocations. Required tests: page alignment, overflow and device limits, CPU/GPU round trip, two CPU aliases of one physical region, canonical GPU identity, remap/unmap during queued work, rejected mappings, allocation failure, and repeated create/submit/release cycles.
- [ ] Compare equal workloads against D3DMetal, including existing alias support. Require wall-clock submission-to-completion and CPU submission/wait timings, bytes copied, and live allocations. Report GPU execution time only from a validated source; known zero D3DMetal timestamps mean unavailable. Do not infer savings from API names.
- [ ] Pass Gate B before expanding the abstraction or committing to an ABI for the rest of the renderer.

## Milestone C: Direct Metal compute

**Depends on:** Gate B. **Approved step:** 3.

- [ ] Write the executable compute-slice plan using the proven bridge ABI and locally verified converter capabilities.
- [ ] Integrate DXIL-to-Metal conversion, minimal bindings/root layout, residency, submission, and completion for the same synthetic programs used in the reference tests.
- [ ] Test compile/link failure, invalid bindings, cross-wave behavior, required atomic operations, cold/warm caches, compiler/options changes, and buffer visibility after completion.
- [ ] Require correct output written back to the original shared allocation through the Windows-to-Metal path. Count and reject accidental D3DMetal execution for selected Metal work.
- [ ] Pass Gate C; record copied bytes and time relative to the same D3DMetal workloads. Only then expand beyond compute.

## Milestone D: Direct Metal rendering

**Depends on:** Gate C. **Approved step:** 4.

- [ ] Split implementation into independently verifiable binding, texture, graphics/mesh, indirect-command, synchronization, and presentation tasks using the existing file map.
- [ ] Reuse existing texture-layout conversion and shader translation. Test mip/array formats, resource aliasing, read/write transitions, depth/stencil/blend, actual mesh-output limits, indirect bounds, and query/fence semantics.
- [ ] Add explicit backend selection through existing CLI/configuration surfaces and capability reporting. An unavailable Metal request must fail clearly, and its caches must not collide with D3DMetal artifacts.
- [ ] Exercise resource exhaustion, compiler errors, command-buffer failure, resize/close, and teardown with in-flight work. Preserve diagnostic causes and avoid silent dropped work.
- [ ] Pass synthetic render comparisons and present a real game frame using Metal for the complete GPU path. Continue to Gate E before calling the goal achieved.

## Milestone E: Game acceptance

**Depends on:** Gate D. **Approved step:** 4, completion.

- [ ] Reuse the prologue scene/input sequence established in Task 1 and isolated save/data directories. Complete any previously unexecuted portion and revisit its unobserved shader families. Use equal resolution/settings and separate cold/warm measurements for each backend.
- [ ] Verify first controllable movement and camera response, geometry, lighting/reflections, UI, and audio/input continuity. Test save/reload when reachable; preserve original saves.
- [ ] Complete a 20-minute warm-cache run with no hang or GPU reset and no unbounded memory growth. Record dropped work, refusal counts, peak memory, copied bytes, and wall-clock frame-time median/tails. GPU timings remain separately labeled with their measurement source or unavailable status.
- [ ] Report visual discrepancies and practical playability with the results. If a required defect remains, keep the goal open and prioritize the measured blocker.
- [ ] Run the integrated checks, review the complete change, and update backend/setup documentation with only verified support and limitations.

## Verification commands and evidence

Use focused tests while changing one behavior. At the baseline and integrated milestone gates, use the repository's documented commands:

```sh
cargo test --workspace --locked
WINEDLLOVERRIDES="d3d12,d3d12core,dxgi=b" tools/macos/test-wine.sh --workspace --no-fail-fast --locked
cargo clippy --workspace --all-targets --target x86_64-pc-windows-gnullvm --locked -- -D warnings
cargo clippy --workspace --all-targets --all-features --target x86_64-pc-windows-gnullvm --locked -- -D warnings
cargo xtask policy
```

For focused Wine commands in tasks above, use the same `WINEDLLOVERRIDES` setting. Add the helper DXC environment from `tools/macos/victory.sh` if the build requires it. Format changed code only; do not reformat unrelated files.

Owned-file checks use `VICTORY_GAME_DIR` or explicit paths from the user's configuration. Set `VICTORY_CACHE_DIR` and `VICTORY_DATA_DIR` to canonical locations outside the repository before scans or game execution, including when using Wine drive mappings. Use the existing `translate --list`, `translate --pipelines`, and `translate --cache` commands; retain counts, options, and version metadata outside the repository. A cache hit on a previous refusal is not a newly tested translation.

Distinguish these evidence levels in every milestone: IR construction, host shader emission, validator acceptance, pipeline creation, GPU execution with known expected output, and game behavior. Each level proves only its own result.

## Plan status

The four architectural steps are approved. This document records the prerequisite-first order and the implementation gates. No product code or runtime upgrade has been performed as part of writing the plan. No completion date or FPS target is asserted before the prerequisite and memory measurements exist.
