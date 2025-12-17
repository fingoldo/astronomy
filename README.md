# Astronomy

Tools for astronomical data analysis and visualization.

## Dataset

This module operates on the [ZTF M-dwarf Flares 2025](https://huggingface.co/datasets/snad-space/ztf-m-dwarf-flares-2025) dataset as described in the paper [arXiv:2510.24655](https://arxiv.org/pdf/2510.24655).

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```python
from astronomy import norm_series, compare_classes

# Plot normalized magnitude series for a single record
norm_series(df, i=0)

# Compare random samples from each class
compare_classes(df)
```

## Requirements

- numpy
- polars
- plotly

## License

MIT
