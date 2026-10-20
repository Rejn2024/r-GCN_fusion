# Track-star HGT training performance

`notebooks/track_star_hgt.ipynb` defaults to eight complete tracks per batch and
eager forward execution, allowing training without a native build toolchain.
Set `COMPILE_MODEL=1` before model configuration to enable
`torch.compile(model, dynamic=True)` on a configured runtime. The raw module owns
the optimizer and checkpoint state in both modes, so saved keys do not gain an
`_orig_mod.` prefix and existing uncompiled checkpoints can still be loaded.
Eager execution retains CUDA, mixed precision, and batch prefetch when available.

## Batch preparation and transfers

`rgcn_fusion.track_star_training` replaces each original batch with compact
lookup vectors, observation counts and one sorted CPU topology. After preparation
the notebook releases the original unsorted edges/types and batch containers.
Re-running the training cell reuses prepared batches; re-run target/batch
construction to change the grouping. Construction-only dataset and graph objects
are released before training, while the memmap and evaluation metadata survive.

On CUDA, iteration uses two reusable pinned host slots and a dedicated copy
stream. A device batch holds features, topology, indices, labels, priors and
counts, shared by its forward, loss and metrics. Only the current batch and one
lookahead batch are staged. Pinned slots wait for their previous copy before
being overwritten; consumer streams wait for copy events and record tensor
lifetimes. Context exit completes pending copies on early stop or exceptions.
The notebook deletes each consumed batch before advancing the iterator.

Features remain memmapped and are gathered only for staged batches. No learned
representations, full split feature cache or full split device topology is
retained. The memory preflight includes lookahead input storage; compilation and
allocator peaks still need measurement on the target GPU.

## Losses, metrics and cleanup

`rgcn_fusion.track_star_loss` computes the component-marginal RAO loss matrix once
for both the objective and optional diagnostics. Prepared observation counts are
reused by mode EDL and KL. Training skips unused diagnostic reductions. The
objective retains the same complete/component weighting, nonuniform RAO priors,
track-balanced mode loss and FP32 KL calculations.

Training loss and streaming metrics accumulate detached device totals with
double-precision accumulation. Each split transfers its compact metric vector
once, preserving track-weighted losses/RAO accuracy and observation-weighted mode
accuracy. Fresh test loss still selects a checkpoint every completed epoch;
training diagnostics retain their configured interval and validation remains
reserved for final evaluation. Both initial and continuation loops use this path.

Full garbage collection and CUDA cache clearing occur at phase boundaries, rather
than after every epoch, allowing allocator reuse. `epoch_time_seconds` retains
its existing scope, including training, fresh diagnostics and checkpoint saving;
logging and phase-boundary cleanup require a separate whole-run timer.

## Controls and validation

| Environment variable | Default | Purpose |
|---|---|---|
| `TRACKS_PER_BATCH` | `8` | Complete tracks per induced batch |
| `COMPILE_MODEL` | `0` | Set `1` to opt into dynamic compilation with a configured toolchain |
| `PREFETCH_BATCHES` | `1` on CUDA | Set `0` for synchronous transfers |
| `METRICS_INTERVAL` | `10` | Train diagnostic refresh interval |

When enabled, dynamic compilation accommodates changing graph sizes, but
PyG/backend versions can still introduce graph breaks or recompilation. First-use
compilation cost is included in the first applicable epoch. Record PyTorch, PyG,
CUDA and optional kernel-backend versions when comparing runs. Eager runs may be
slower than successfully compiled runs; measure on the intended hardware.

### Windows: `InductorError: RuntimeError: Compiler: cl is not found.`

The default `torch.compile` backend, Inductor, generates native code. This error
means it could not find Microsoft's C++ compiler in the Jupyter kernel's
environment. Compilation is lazy: creating the wrapper can succeed before the
first forward triggers code generation and fails. This is a compiler setup
failure, rather than evidence of a model, input, or loss defect. CPU code-generation
frames do not establish which device the entire model uses.

To resume without compilation, run this before the model-configuration cell:

```python
import os
os.environ["COMPILE_MODEL"] = "0"
```

Then rerun model configuration and training. Restarting the kernel and running
the notebook from the beginning is the simplest way to clear an existing compiled
wrapper. Changing the environment variable after wrapper creation does not
replace it. With the updated default, an unset variable also selects eager mode;
an existing explicit `COMPILE_MODEL=1` still requests compilation.

For optional compiled runs on Windows:

1. Install Visual Studio 2022 Build Tools with **Desktop development with C++**,
   including the MSVC x64 tools and Windows SDK.
2. Open an **x64 Native Tools Command Prompt for VS 2022**, activate the intended
   Conda environment (for example, `cmo_sensors`), and launch Jupyter there.
   Restart the kernel so it inherits the compiler, include, and library paths.
3. Check compiler discovery from that kernel:

   ```python
   import os
   import shutil
   print(shutil.which(os.environ.get("CXX", "cl")))
   ```

   `None` means the selected compiler is unavailable. Correct a stale `CXX`
   override if necessary. Finding the executable alone does not establish that
   the complete toolchain works.
4. Confirm that the installed PyTorch version supports Inductor on the intended
   platform/device, including any additional CUDA/backend requirements. Test a
   small compiled forward **and backward** before enabling `COMPILE_MODEL=1` and
   rerunning model configuration. MSVC alone does not establish GPU backend support.

Eight-track grouping changes optimizer steps per epoch and the shared-KG message
context compared with four tracks. Compare independent validation accuracy,
calibration and open-set uncertainty under matched training budgets.

Run the relevant regression tests with PyTorch, PyG, pytest, NumPy, matplotlib,
scikit-learn, Plotly and TensorBoard installed:

```sh
python -m pytest tests/test_track_star_hgt.py tests/test_track_star_hgt_notebook.py tests/test_track_star_loss.py tests/test_track_star_training.py tests/test_graph_optimisation.py -q
```

Tests cover independent loss/gradient parity, uneven-batch metric weighting,
initial and continuation training with unset, disabled, and enabled compilation,
dynamic capture, raw checkpoint reload,
complete evaluation/export consumers, memmap preservation and topology ownership.
CUDA tests cover transfer parity, bounded pinned-slot reuse and early termination;
they skip when CUDA is unavailable. Compilation integration tests use Dynamo's
`eager` backend, so they do not validate Windows Inductor toolchains or GPU code
generation. Synthetic CPU checks do not establish a GPU epoch speedup.
For performance validation, warm up and compare complete ordinary
epochs plus a train-diagnostic refresh, recording host RSS, CUDA peak allocated
and reserved memory, checkpoint I/O and total wall time.
