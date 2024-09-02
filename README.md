# Yase - Depth Estimation Library

Yase (Yet Another Semantic Extractor) is a Python library for depth estimation using the Monodepth2 and FastDepth models. This project is designed for use in applications that require semantic field extraction from video streams, while maintaining real-time performance.

## Features

- Depth estimation from images using the Monodepth2 model.
- Automatic downloading of pre-trained model weights.
- GPU support for increased performance.
- Structured to allow easy extension and integration into larger projects.

## Installation

### Prerequisites

- Python 3.8 or higher
- `Poetry` for package management and installation
- A CUDA-enabled GPU for running on GPU (optional)

### Steps

1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/yase.git
    cd yase
    ```

2. Install the package using `Poetry`:
    ```bash
    poetry install
    ```

3. If you don't have `Poetry` installed, you can install it via `pip`:
    ```bash
    pip install poetry
    ```

    Then run the installation again:
    ```bash
    pip install dist/yase-0.1.0-py3-none-any.whl
    ```

## Usage

### Example: Depth Estimation from an Image

Here's how to use the `Yase` library to estimate depth from an image:

```python
from yase.yase import Yase

# Initialize the Yase object
yase = Yase()
