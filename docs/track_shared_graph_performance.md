# Shared-candidate graph construction performance

The `graph_data = build_shared_evidence_graph(...)` cell in
`notebooks/track_shared_candidate_hgt.ipynb` rebuilds candidate evidence, merges
track-local candidates and intelligence routes, then writes node/incidence
memmaps and topology artifacts. The scorer evaluates up to 320 retrieved
hypotheses before retaining 18. Reducing either limit could save time but changes
the experiment; these changes keep both limits and all scoring semantics.

## Implemented changes

* **Process parallelism across tracks.** Each worker scores and materialises an
  index-local fragment using the existing `parallel_graph_construction` worker.
  Candidate templates are installed once per worker. The parent consumes
  fragments in source order and assigns all global node, relation and feature
  indices. This preserves tie ordering, track isolation and routing audits.
  Unlike threads, processes can run the scorer's Python loops on multiple CPUs.
* **Bounded disk transport.** At most twice the worker count of tasks is submitted
  at once. Workers pickle fragments directly into temporary files and return
  small descriptors, avoiding an additional fragment-sized IPC byte buffer.
  Loaded files are deleted immediately; pool, spool and audit handles are closed
  on success and failure. This also prevents an unbounded queue of completed
  fragments when one early track is slow.
* **Faster feature materialisation.** Private temporary feature rows use binary
  pickle instead of JSON encoding/decoding. The fixed signed `log1p` transform
  runs as NumPy operations on bounded chunks rather than one scalar ufunc call
  per value. Float64 intermediates preserve the original transform before its
  final float32 cast. Value/sign scratch arrays together use approximately
  16 MiB at most, except for an exceptionally wide single row. Final memmap names,
  sorted columns, values and manifest schemas are unchanged.
* **Reuse report proximity while routing.** A report and its claims share one
  proximity result per observation. With `C` locally routed claims, that routing
  calculation falls from `1+C` calls to one. This cache is track-local and does
  not change observation-specific applicability or recency in candidate scoring.

## Measurements and estimates

Measurements used Python 3.12.14, NumPy 2.3.5 and CPU PyTorch 2.14.1, with a
**two-CPU container quota**. The independent synthetic workload had 24 tracks,
264 observations, the repository's saved KG and 732 candidate rows. Each result
is the median of three full builder calls, including process startup and disk
writes. Graph digests matched the unmodified builder in every run.

| Full graph build | Median time | Speedup over original |
| --- | ---: | ---: |
| Original serial builder | 33.10 s | 1.00× |
| Optimized, one worker | 28.27 s | 1.17× |
| Optimized, two workers | 20.86 s | 1.59× |
| Optimized, four workers | 23.49 s | 1.41× |

The baseline ranged from 30.09–35.34 s, so small differences need confirmation
on the target machine. Four workers oversubscribe this container's quota;
two are faster here. A separate short pilot (12 tracks, 60 observations) took
6.87 s originally, 6.76 s with one worker, 6.75 s with two, and 11.67 s with
four. Startup costs can outweigh parallelism on small inputs.

The isolated feature-spool benchmark (20,000 rows × 48 features, three repeats)
fell from **2.80 s to 0.44 s, about 6.4× faster** for append plus materialisation,
with exactly equal values. That is a stage speedup, not a whole-cell multiplier.
Routing proximity goes from two calls to one for the local report/claim pairs
in the regression fixture; the general reduction is `1+C` to one for `C`
locally routed claims. This stage is small compared with candidate scoring.

An additional baseline profile spent 72.67 of 74.39 profiled seconds inside
track fragment construction, primarily candidate/intelligence scoring and DS
fusion. Profiling changes the relative cost of Python loops, so use this as
evidence of the bottleneck rather than an exact wall-time forecast. For a long
scoring-dominated run, assuming a parallel fraction `p = 0.90–0.98`, Amdahl's
estimate `1 / ((1-p) + p/N)` gives **1.82–1.96× for two CPUs** and
**3.08–3.77× for four CPUs**, before startup, serialization, disk contention and
track-size imbalance. Budget roughly **2.5–3.5× on four genuinely available CPUs**
for sufficiently long, balanced workloads; this is an estimate, not a measured
four-CPU result. The observed two-CPU end-to-end improvement was 1.59×.

## Controls and tradeoffs

The notebook selects up to four workers for at least 1,000 observations and one
for smaller workloads. This is a conservative startup heuristic, not a measured
break-even guarantee. Set `GRAPH_BUILD_WORKERS=2` or `4` before running the
configuration cell to choose explicitly, or `1` for serial execution. Direct
Python callers retain the serial default and can pass `num_workers=4`.
Workers are capped at the number of tracks. Choose no more workers than your
available CPU quota and memory can support; CPU counts can exceed container
quotas.

`GRAPH_BUILD_START_METHOD` defaults to `spawn`, including on Linux, because the
notebook has already imported PyTorch and may have initialized CUDA. Importable
worker entry points work in notebook kernels and on Windows. A plain Python
script must call the builder under `if __name__ == "__main__":` when spawning.
Other platform-supported start methods are explicit opt-ins; forking a live
CUDA/threaded kernel is unsuitable.

Each worker owns its candidate context, active input track and fragment. There
can also be up to `2 * workers` queued/completed fragment files. Put the artifact
directory on fast local storage and allow temporary disk space. Global graph
assembly, routing, final feature transforms and artifact writes remain serial;
one very large track also limits parallel speedup. Parallelising individual
observations would duplicate report preparation and require more IPC, so tracks
are the preferred partition. Threads are unlikely to help the Python scoring
loops; GPUs do not directly accelerate their dictionary/string processing.

## Reproduce on your dataset

The full `generated/demo_esm_observation_series_open_set_ufo.json` is not present
in the checkout used for these measurements. The notebook's original inputs
were not regenerated or overwritten. The benchmark can load your existing file,
or explicitly generate an independent in-memory synthetic workload. It writes
graphs only to temporary directories and checks that all variants produce the
same hash of metadata, topology, feature values and routing audit. Dataset/KG
loading and synthetic generation are outside the timed graph-building call;
process startup and artifact writes are included.

```bash
mkdir -p work
git show 6810d60b61624fdc6855023a86beae81d04bc41f:rgcn_fusion/track_shared_graph.py > work/shared_graph_baseline.py
python scripts/benchmark_shared_graph.py \
  --series-json generated/demo_esm_observation_series_open_set_ufo.json \
  --workers 1 2 4 --repeats 3 --baseline-source work/shared_graph_baseline.py
```

To reproduce the separate synthetic workload, replace `--series-json ...` with
`--synthetic-tracks 24 --duration 5`. Use `--spool-rows 20000` instead of a dataset
argument for the 48-feature feature-spool benchmark. Add
`--profile work/shared_graph.prof` to profile one extra baseline/serial run;
profiling overhead is excluded from the reported timings.

The regression tests compare serial and spawned builds under both intelligence
routing modes, including their persisted artifacts. They also check bounded
submission, worker-failure cleanup, invalid controls, proximity reuse and exact
feature transforms across chunk boundaries. Existing graph/model/loss tests
continue to exercise the training contract.
