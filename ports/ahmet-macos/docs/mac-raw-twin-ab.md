# Raw descriptor twin A/B and Metal attribution

Use a diagnostics build of `agent/one`. `VICTORY_RAW_TWIN_COPY` is off by default; naming it in `VICTORY_AB` switches it per phase. (Codex's packed slot allocator was not ported.)

## One-run descriptor comparison

Use the same save and route with `VICTORY_AB=15,VICTORY_RAW_TWIN_COPY,VICTORY_AB_NULL`. Leave `VICTORY_METAL_BRACKETS`, `VICTORY_TIMELINE`, and Instruments off. Keep an uninterrupted 60-second cycle of completed phases after the same scene settles. The diagnostic logger drops the first second of each phase.

| Phase bits | Exact raw twins | Control |
| ---: | --- | --- |
| 0 | Create separately | Off |
| 1 | Copy from CPU staging | Off |
| 2 | Create separately | On |
| 3 | Copy from CPU staging | On |

Compare phases 1/0 and 3/2 for the effect, plus 2/0 and 3/1 for A/A noise. Use the existing `xcl-mac/tools/phases.py --draws 3000 LOG` to summarize phase lines, then check complete cycles, comparable draw counts, and neutral A/A results. The reader drops the first 100 seconds of world by default. A/B timing only counts with visibly equivalent output and a change beyond the measured A/A drift.

The raw-twin line gives the number copied. Within a reporting window wholly inside an off phase, expect zero copies. Within an on phase, the count is the actual number of identical raw twins that qualified for copying. Counter windows can span phase changes, so use only fully contained windows for this check.

## Separate attribution window

For attribution, leave `VICTORY_AB` and `VICTORY_TIMELINE` unset and use the normal raw-twin default. Set `VICTORY_METAL_BRACKETS=<seconds after startup>,3,<absolute Windows TSV path>` and overlap a short Metal System Trace with that settled scene. The recorder arms on a present after that offset and logs when collection begins. Allow the asynchronous TSV write to finish.

Check that the TSV has `qpc_hz`, `windows_pid`, `lost=0`, the present boundaries, and command and submit rows. Use the exact same-process segment join and the native macOS PID described in `xcl-mac/lanes/gpu-attribution/README.md`. A missing or ambiguous join stays unattributed.
