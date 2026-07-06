# r-GCN Fusion for Dempster-Shafer Evidence

This repository contains a Python training pipeline for learning evidential masses from a Neo4j knowledge graph with a relational graph convolutional network (r-GCN). The trained model predicts Dempster-Shafer mass functions for graph entities and can report belief/plausibility intervals for each hypothesis.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m rgcn_fusion.train --config configs/example.yaml
```

The example configuration documents the expected Neo4j connection settings, label format, model hyperparameters, and output paths.
