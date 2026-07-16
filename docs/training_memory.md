# Training memory reduction guide

The observation-series notebooks and `rgcn_fusion.train` mini-batch the supervised
loss over observation indices, but each batch still executes message passing over
the full graph. If CUDA reports an out-of-memory error inside `model(X,
edge_index, edge_types)` or a task-head dropout layer, treat the graph encoder and
all full-node task logits as the primary memory drivers rather than only the
`DataLoader` batch size.

## Highest-impact changes

1. **Enable gradient checkpointing.** Set `model.gradient_checkpointing: true` in
   config, or pass `gradient_checkpointing=True` to the notebook classifier. This
   trades extra forward recomputation for lower activation storage across the
   residual r-GCN stack.

   Config file example:

   ```yaml
   model:
     gradient_checkpointing: true
   ```

   Notebook classifier example:

   ```python
   GRADIENT_CHECKPOINTING = True

   model = SeriesRGCNClassifier(
       in_dim=X.shape[1],
       hidden_dim=HIDDEN_DIM,
       num_relations=len(RELATIONS),
       class_sizes=class_sizes,
       gradient_checkpointing=GRADIENT_CHECKPOINTING,
   )
   ```
2. **Chunk relation message construction.** Use `model.edge_chunk_size` (for
   example `2000`, `5000`, or `10000`) so each relation processes source/target
   edges in slices instead of materializing one large relation message tensor.
   Reduce the value until the forward pass fits.
3. **Reduce full-graph embedding size.** Lower `model.hidden_features`,
   `model.num_layers`, and `model.task_head_hidden_features`. Full-node tensors
   scale roughly with `num_nodes * hidden_features * num_layers`, and every
   auxiliary task head can also produce full-node logits.
4. **Trim auxiliary classification heads.** Disable tasks that are not needed for
   the run, or make their hidden heads linear by setting
   `model.task_head_hidden_features: 0`/`null`. A dropout OOM in a head usually
   means the shared embedding plus all per-node logits have already filled the
   GPU.
5. **Prefer smaller graph projections.** Turn off optional graph expansion such
   as candidate nodes, segment edges, or extra shortcut relations when they are
   not needed for the experiment. Fewer nodes and edges directly reduce encoder
   activation and message memory.

## Training-loop and runtime changes

- Keep automatic mixed precision (AMP) enabled on CUDA (`training.use_amp: true`).
  AMP runs eligible operations in lower-precision dtypes, such as `bfloat16` or
  `float16`, to reduce activation memory while preserving full-precision values
  where PyTorch requires them.
- Decrease `training.batch_size` to reduce the indexed loss tensors, but expect
  limited savings when the full graph is still encoded for every mini-batch.
- Accumulate gradients over several smaller supervised batches only after the
  per-batch full-graph forward fits in memory.
- Run validation/test under `torch.no_grad()` or `torch.inference_mode()` and
  avoid storing full logits/history tensors beyond the metrics needed for the
  epoch.
- If fragmentation appears in the CUDA error, start the process with
  `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`. This can reduce allocator
  fragmentation but will not fix an intrinsically oversized model/graph.
- Restart the Python kernel after an OOM before re-running smaller settings;
  failed CUDA allocations can leave the process in a high-water memory state.

## Example low-memory starting point

```yaml
model:
  hidden_features: 64
  num_layers: 3
  num_bases: 2
  task_head_hidden_features: 0
  edge_chunk_size: 2000
  gradient_checkpointing: true

training:
  batch_size: 16
  use_amp: true
```

For the notebooks, make the equivalent edits to `HIDDEN_DIM`,
`NUM_RGCN_LAYERS`, `NUM_BASES`, `TASK_HEAD_HIDDEN_DIM`, `EDGE_CHUNK_SIZE`,
`GRADIENT_CHECKPOINTING`, `BATCH_SIZE`, and `USE_AMP` before constructing the
model and optimizer.
