# Track HGT training optimisations

`notebooks/track_hgt.ipynb` prepares the immutable topology of each induced
track batch once. `prepare_relation_graph` groups edges by relation and replaces
the batch's original edge storage. All relation tensors are views into that one
replacement tensor. Training, diagnostics, continuation, and final evaluation use
the same prepared graphs; no pass rebuilds relation masks or incoming degrees.
Only the active graph and its lookup tensors transfer to the accelerator.

The custom r-GCN computes `sum(x[source] @ W[relation]) / degree[destination]`.
For relations with at least two edges per distinct source, it now computes the
dense transform once per distinct source and gathers those results per edge.
This preserves duplicate edges, relation assignments, total incoming-degree
normalization across all relations, and the model's parameter/checkpoint format.
HGT still receives every original edge, including empty relation partitions.
Floating-point rounding can differ with matrix shape and device kernels.

Source lookups use compact indices where possible. The persistent degree and
source-lookup tensors together are bounded by the original batch's edge,
relation-id and node-index storage bytes. Once that budget is exhausted, further
relations use the original edge transform path. Features remain memmapped.
Edge gathers remain chunked; if the distinct-source projection would exceed one
edge chunk, that relation also falls back to the original path. The memory
preflight includes the topology-cache upper bound. These bounds control the
added storage, but actual CUDA allocator peaks still require measurement on the
training data.

## Expected performance and validation

The earlier whole-epoch estimates were 1.4–2.5x for reducing repeated r-GCN
transforms when source reuse is high and r-GCN dominates runtime, and 1.05–1.25x
for topology caching if preparation previously took 5–20% of runtime. These are
conditional estimates, not measured training results, and should not be
multiplied. Sparse relations with little source reuse retain the original dense
transform cost. Static degrees also remove a scatter operation per relation per
r-GCN layer and checkpoint recomputation.

Run the targeted regression checks with PyTorch, PyG and pytest installed:

```sh
python -m pytest tests/test_track_hgt_optimisation.py tests/test_track_hgt_cleanup.py tests/test_graph_optimisation.py tests/test_track_output_gathering.py tests/test_parallel_graph_construction.py -q
```

Tests compare outputs and input/weight gradients against the original formula,
exercise actual notebook training with and without checkpointing and mixed
precision, verify storage sharing/cache bounds, and check preparation is reused.
CUDA mixed-precision coverage runs only when CUDA is available.

For representative performance measurements, compare the parent commit and this
change with identical source artifacts, seeds, batch sizes and precision. Warm
up the model, then measure complete epochs including fresh test metrics and
checkpoint saves. Record host RSS and reset/read CUDA peak allocated and reserved
memory around each run. Accept memory growth only below 2x baseline; compare
final independent-validation accuracy and uncertainty as well as throughput.
Full-data timing and CUDA peak-memory measurements were unavailable in the
implementation environment because it had no GPU or saved training artifacts.
