<div align="center"><h2>CP6D: Quantifying Camera Relocalization Uncertainty via Conformal Prediction</h1></div>

<p align="center">
    <!-- community badges -->
    <a href=""><img src="https://img.shields.io/badge/Project-Page-ffa"/></a>
    <!-- doc badges -->
    <a href="">
        <img src='https://img.shields.io/badge/arXiv-Page-aff'>
    </a>
    <a href=""><img src="https://img.shields.io/badge/pypi package-0.3.4-brightgreen" alt="PyPI version"></a>
    <a href=''>
        <img src='https://img.shields.io/badge/Poster-PDF-pink' />
    </a>
</p>

### Attention: All Rotation Quaternions are in w,x,y,z!!!

## 1. Installation: Setup the environment

### Prerequisites

🥳 CPU is enough for the Conformal Prediction part. To run this code base, CPU is sufficient.

If you intend to train a model from scratch or fine-tune a pre-trained model, refer to the camera relocalization code documentation. This process typically requires CUDA, as specified in their system requirements.

### Create environment

We tested out codes on python=3.9 and python=3.10. We recommend using conda to manage dependencies. Make sure to install [Conda](https://docs.conda.io/en/latest/miniconda.html) before proceeding.

```bash
conda create --name CP6D -y python=3.10
conda activate CP6D
python -m pip install --upgrade pip
```

The required libraries are Torch, Numpy, Scipy, Matplotlib and some additional libs. For CUDA available users, we recommend a installation with CUDA
```bash
pip install torch==2.1.2+cu118 torchvision==0.16.2+cu118 --extra-index-url https://download.pytorch.org/whl/cu118
```
or
```bash
pip install torch torchvision
```
The rest of additional libs could be easily installed with
```bash
pip install numpy==1.24 scipy pandas matplotlib IPython tqdm scikit-image opencv-python
```
For quick installation, please refer to environment.yaml. 
```bash
conda create -f environment.yaml
```