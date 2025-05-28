# Private Generative Models: Balancing Privacy and Utility in AI


This repository accompanies my PyCon talk on **Private Generative AI**, showcasing how increasing privacy affects data utility using simple synthetic data generation techniques.

### TL,DR

- Uses the **Iris dataset**.
- Applies **Gaussian noise** at three privacy levels to generate synthetic data.
- Computes:
  - **Utility** via Wasserstein Distance
  - **Privacy** via Signal-to-Noise Ratio (SNR)
- Plots 2D KDE of real vs synthetic data to show the privacy–utility tradeoff visually.

### Setup (using [uv](https://github.com/astral-sh/uv))

```bash
uv venv
source .venv/bin/activate  # or .venv\\Scripts\\activate on Windows
uv pip install  # installs from pyproject.toml
```