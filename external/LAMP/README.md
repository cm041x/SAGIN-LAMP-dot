# ![LAMP](assets/lamp.svg) Long-range Adaptive Memory Propagation (LAMP)

Official implementation of

> **Adaptive Memory Retention in Dynamic Graphs.**
> *Fabrizio De Castelli, Alessio Gravina, Moshe Eliasof, Carola-Bibiane Schönlieb, Davide Bacciu,*
> ICML 2026 Spotlight.

[Project page](https://fabridecastelli.github.io/projects/LAMP/) ·
[Paper (PMLR)](https://proceedings.mlr.press/v306/de-castelli26a.html)

LAMP is a differential-equation-based GNN for discrete-time dynamic graphs. It
treats incoming graph snapshots as **impulses** driving a neural impulsive ODE:
an antisymmetric (non-dissipative) parameterization propagates information
conservatively across space and time, while a learned, channel-wise **adaptive
dissipation** term $\Gamma$ regulates how much information is retained or erased.

The model is in [`src/models/lamp.py`](src/models/lamp.py), its message passing in
[`src/layers/convolutions/lamp_conv.py`](src/layers/convolutions/lamp_conv.py),
the baselines in `src/models/baseline/`.

## Quick start

```bash
git clone https://github.com/FabriDeCastelli/LAMP.git
cd LAMP
uv sync --extra cu121      # or cu118, or cpu
source .venv/bin/activate
python -m paper.reproduce paper/node_prediction.yaml
```

The last command reproduces the node-prediction table of the paper.

## Documentation

- [Installation](docs/installation.md): environments, PyTorch builds, checking the setup
- [Data](docs/data.md): datasets, where they are stored, splits
- [Reproducing the paper](docs/reproducing.md): one command per table and figure, exactness
- [Running experiments](docs/experiments.md): model selection, assessment, configuration files, GPUs
- [Collecting results](docs/results.md): output folders and files, gathering tables
- [Traffic forecasting](docs/traffic.md): the METR-LA and PeMS-Bay pipeline

## Repository layout

```
paper/                   one file per paper table + scripts for tables, figures, runtimes
docs/                    documentation
configurations/
  datasets/              dataset settings
  hyperparameters/       hyperparameter grids for model selection
src/
  training.py            training loop, early stopping, evaluation, parallel runs
  models/                LAMP (lamp.py) and the baselines (baseline/)
  layers/                LAMP convolution, dissipation, discretization, readouts
  datasets/              ColoredLeafCounting, node and link prediction tasks
main.py                  model selection
run_reinit.py            model assessment
traffic_forecasting/     traffic experiments (tsl)
```

## Citation

If you use LAMP or the `ColoredLeafCounting` benchmark in your research,
please cite:

```bibtex
@InProceedings{pmlr-v306-de-castelli26a,
  title =      {{Adaptive Memory Retention in Dynamic Graphs}},
  author =       {De Castelli, Fabrizio and Gravina, Alessio and Eliasof, Moshe and Sch\"{o}nlieb, Carola-Bibiane and Bacciu, Davide},
  booktitle =      {Proceedings of the 43rd International Conference on Machine Learning},
  pages =      {23292--23315},
  year =      {2026},
  editor =      {Zhang, Tong and Dudik, Miroslav and Jaggi, Martin and Agarwal, Alekh and Li, Sharon and Schuurmans, Dale and Zhu, Jerry and Berkenkamp, Felix and Dong, Hanze and Bietti, Alberto},
  volume =      {306},
  series =      {Proceedings of Machine Learning Research},
  month =      {06--11 Jul},
  publisher =    {PMLR},
  pdf =      {https://raw.githubusercontent.com/mlresearch/v306/main/assets/de-castelli26a/de-castelli26a.pdf},
  url =      {https://proceedings.mlr.press/v306/de-castelli26a.html}
}
```

## License

This project is released under the [MIT License](LICENSE).
