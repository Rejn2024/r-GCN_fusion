# Accelerated shared-candidate HGT from a saved graph

Run `notebooks/track_shared_candidate_hgt_accelerated.ipynb` from the repository
root or `notebooks/`. The original notebook remains available for construction
and comparisons. The new notebook never constructs a graph or scores candidates.

## Cache prerequisites

Run the original notebook's **Save or reload the shared-candidate graph** cell
with `GRAPH_CACHE_ACTION="save"` after constructing the graph. It records the
input fingerprints and graph settings used by the new loader. Keep these files
together in `artifacts/track_shared_candidate_hgt/`:

```text
graph_construction_outputs.pt
relational_graph_outputs.pt
nodes.float32.memmap
incidences.float32.memmap
```

Keep `intelligence_routing.jsonl` and `graph_reduction_audit.json` for auditing.
`SHARED_GRAPH_CACHE_DIR` selects a different bundle. Metadata is loaded normally;
FP32 node/incidence features use NumPy copy-on-write memmaps, and topology uses
`torch.load(..., mmap=True)`. All graph fields and downstream aliases are restored.
The loader checks provenance, feature sizes/schema, and index ranges in bounded
chunks. Missing, stale or incompatible artifacts raise an error instead of
triggering an expensive rebuild. Load only trusted local PyTorch metadata files.

The original dataset and KG JSON are still required for labels, KG-valid output
vocabularies and priors. Their hashes are computed incrementally. Graph loading
does not bypass target or complete-track batch preparation. The source bundle is
not rewritten. All checkpoints, logs, dashboards and predictions go to
`artifacts/track_shared_candidate_hgt_accelerated/`.

## Compiler setup

The early bootstrap cell adapts the **last cell of `C++ compiler debug.ipynb`**:
on Windows it runs `vcvars64.bat` and imports its environment into the current
kernel. The default is Visual Studio 18 Community, matching that notebook.
Set `VCVARS64` to another installation's batch file. On POSIX, use `CXX` or `c++`
on PATH. The cell actually compiles, links and runs a tiny C++ program.

This establishes C++ access, not complete GPU Inductor support. When compilation
is enabled, model configuration also runs a small compiled forward/backward
probe. Windows GPU compilation still needs a supported PyTorch/backend/toolchain
combination; a working MSVC installation alone does not establish that. Failures
are surfaced. `COMPILE_MODEL=0` selects eager training after compiler setup.

## Training changes

* Pack the three independent Q/K/V projections per node type into one linear
  operation. Retain original parameter names/shapes and initialization. Typed
  outputs use disjoint writes into a fresh buffer instead of repeated full-buffer
  out-of-place copies. Output projections and normalization remain type specific.
* Deduplicate exactly equal static attribute rows within each prepared batch.
  Keep every edge, relation and incidence value. Representatives remain compact
  indices into the shared memmap; no full-split dense feature cache is retained.
  Group candidate and intelligence rows, omit columns that are exactly zero in
  each group, and pack edge-key/value projections. Learned values are recomputed
  every layer/step. This optimization assumes raw edge attributes are fixed;
  it is not a feature-gradient optimization for trainable raw inputs.
* Adapt the existing track-star two-slot prefetch implementation to shared
  graphs. Include incidence attributes, source caches, type indices and projection
  groups. Reuse pinned host buffers, wait before overwriting DMA inputs, register
  tensor lifetimes on the consumer stream, and drain pending copies on early exit.
  At most the current device batch and one lookahead input batch are retained.
* Use fused CUDA AdamW and a packed L1 reduction, optionally compiled separately.
  Preserve the L1 coefficient, division by train-batch count and GradScaler order.
  Both initial and continuation loops use this path. L1 is not replaced by decay.
* Compile the packed projection regions with dynamic shapes by default on CUDA;
  retain the raw module for optimizer/checkpoint state. Full-model compilation is
  optional: a native CPU check of even a two-layer model generated dozens of C++
  kernels and was stopped after several minutes of compilation. Restricting the
  default scope avoids unrolling all attention relations/layers into one compile.
  Reduce train diagnostics to every ten epochs
  and record `train_metrics_epoch` so cached metrics are identifiable. Fresh test
  metrics still select checkpoints every completed epoch. Validation remains
  reserved for final evaluation. Restore safe construction-object cleanup.

Default grouping stays at eight tracks, with four layers, width 120, four heads,
FP16 CUDA AMP, original losses and priors. Parameter state dictionaries remain
compatible with the reference implementation at matching dimensions. Full
notebook checkpoint configuration intentionally identifies this experiment and
its acceleration options; old experiment checkpoints are not silently resumed.
The original implementation's global incoming-edge softmax, source-transform
caches and forward/reverse attribute reuse are retained.

## Controls and memory

| Variable | Default | Meaning |
|---|---|---|
| `SHARED_GRAPH_CACHE_DIR` | Original artifact directory | Saved graph bundle |
| `VCVARS64` | VS 18 Community batch file | Windows compiler environment |
| `COMPILE_MODEL` | `1` on CUDA, `0` on CPU | Enable projection/model and L1 compilation |
| `COMPILE_SCOPE` | `projections` | Use `model` to opt into full-model compilation |
| `PREFETCH_BATCHES` | `1` on CUDA | Bounded asynchronous staging |
| `COMPACT_EDGE_FEATURES` | `1` | Exact attribute row/column compaction |
| `METRICS_INTERVAL` | `10` | Full train diagnostic interval |
| `TRACKS_PER_BATCH` | `8` | Complete tracks in each induced graph |
| `BASELINE_PEAK_GPU_BYTES` | `0` | Original-run allocated CUDA peak for a relative check |

The preflight includes lookahead inputs and extra packed-parameter/L1 workspace.
It checks available VRAM but remains an analytical estimate. Epochs report peak
allocated and reserved memory. When `BASELINE_PEAK_GPU_BYTES` is supplied, an
allocated peak above **3×** raises an error after the epoch's work completes; this
is detection, not an allocator-level hard cap. Without a measured baseline the
notebook cannot establish a measured relative bound. Compilation can allocate
additional workspace or cached variants, so measure the combined configuration.

Try 16 tracks before 24 if measured memory permits. Larger groups change shared
KG context and optimizer updates per epoch, so compare independent-validation
quality and calibration. They are optional experiments, not equivalent batching.
Full-GPU graph caching, full track-level intelligence rerouting, further layer
reduction and a custom sparse-attention kernel are not enabled by this change.

## Validation and timing

Run the regression checks with repository dependencies, pytest, nbformat,
scikit-learn and TensorBoard installed:

```sh
python -m pytest -q tests/test_track_shared_acceleration.py tests/test_track_shared_candidate.py tests/test_track_star_loss.py tests/test_parallel_graph_construction.py
```

Checks cover cache equality and disk protection, stale/incomplete cache rejection,
reference-versus-packed outputs/input/parameter gradients in FP32 and CPU BF16,
L1/Adam updates, Dynamo capture and checkpoint keys, CPU staging and CUDA-prefetch
parity/early exit when CUDA is available. The integration test executes every
code cell on a saved 12-track fixture, through two initial and two continuation
epochs, checkpoint reload, independent validation, dashboards and prediction
export. Plots are computed but heavyweight rendering and MLflow I/O are stubbed.

CPU correctness tests do not measure RTX 5080 throughput or Windows compilation.
On PyTorch 2.11.0+cpu, a separate native Inductor check also completed packed
projection and compiled L1 forward/backward successfully (about 37 seconds
including first-use compilation on the small fixture). This verifies the default
compilation path on CPU, not target-GPU speed or backend support.
The planning estimate remains **1.5–2.2×** faster complete epochs for combined
packing, compaction, overlap and successful compilation; individual benefits
overlap and must not be multiplied. No GPU speedup is claimed as measured here.

On the target RTX 5080 / PyTorch 2.11.0+cu130 runtime, restore matched model,
optimizer and RNG state for comparisons, keep eight-track groups and precision,
and warm up representative training/evaluation shapes. Measure several complete
epochs spanning a diagnostic refresh, reporting compilation separately. The
epoch timer includes training, diagnostics and checkpoint saves; record a whole-run
timer for logging too. Profile a few batches with CPU/CUDA activities to distinguish
dispatch, copies, dense projections and sparse reductions. Inspect graph breaks
and recompilation (`TORCH_LOGS=graph_breaks,recompiles`); `dynamic=True` does not
promise one graph for all relation/cache patterns. Compare measured allocated and
reserved peaks, and independent-validation accuracy, calibration and open-set
uncertainty before adopting a changed batch size.
