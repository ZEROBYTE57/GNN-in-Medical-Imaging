# Retinal-GNN

Robustness benchmark for graph-based retinal vessel segmentation (dissertation phase 2).

## Status
- [x] Evaluation metrics (`retinal_gnn/metrics.py`), 7 unit tests passing
- [ ] Data loaders and fixed splits
- [ ] U-Net baseline
- [ ] VGA-Net-style graph module (reimplemented from the paper; the official repo is not runnable)
- [ ] Corruption suite and annotation-noise suite
- [ ] Experiments, tables, paper

## Setup (Windows, RTX 4060 8GB)
```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
python -m pytest tests -q
```
Datasets go in `data/` (not committed). See `docs/DATASETS.md`.
