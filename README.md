<h1 align="center">Human Universal Grasping</h1>

<p align="center">
  <a href="https://grasping.io"><img src="https://img.shields.io/badge/Project-Website-2ea44f.svg" alt="Project Website"></a>
  <a href="https://arxiv.org/pdf/2606.17054"><img src="https://img.shields.io/badge/Paper-PDF-1f6feb.svg" alt="Paper PDF"></a>
  <a href="https://arxiv.org/abs/2606.17054"><img src="https://img.shields.io/badge/arXiv-2606.17054-b31b1b.svg" alt="arXiv"></a>
  <a href="https://huggingface.co/kevinywu/hug"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20HF-Weights-yellow.svg" alt="Weights"></a>
  <a href="https://github.com/KevinyWu/aria2mano"><img src="https://img.shields.io/badge/GH-Dataset-181717.svg?logo=github" alt="Dataset"></a>
  <a href="https://github.com/KevinyWu/aria2mesh"><img src="https://img.shields.io/badge/GH-Benchmark-181717.svg?logo=github" alt="Benchmark"></a>
</p>

Trained solely on real-world human grasping data, HUG generates diverse human hand grasps for any user-selected object in a single RGB-D image captured from a stereo camera.  **HUG works with any stereo camera, anywhere, out of the box.**

<p align="center">
  <img src="assets/img/hug_demo.gif" alt="HUG demo" width="100%"/>
  <br/>
  <em>HUG grasps on unseen objects in unseen environments.</em>
</p>

## 🗓️ Release

- [x] Paper and website
- [x] Inference + visualization code
- [x] [1M-HUGs](https://huggingface.co/datasets/kevinywu/1m-hugs/tree/main) dataset released
- [x] [HUG-Bench](https://huggingface.co/datasets/kevinywu/hug-bench/tree/main) assets released
- [x] [aria2mano](https://github.com/KevinyWu/aria2mano) Aria data processing code released!
- [x] [aria2mesh](https://github.com/KevinyWu/aria2mesh) metric-scale mesh generation code released!
- [ ] 1M-HUGs dataset creation and visualization code
- [ ] HUG-Bench simulation code
- [ ] HUG training code
- [ ] Support RealSense and ZeD live streaming

## 📦 Installation

Tested on Ubuntu 22.04/24.04, CUDA 12.8, PyTorch 2.9.1, Python 3.10.

```bash
# 1) Environment
conda env create -f environment.yaml && conda activate hug
pip install torch==2.9.1 torchvision==0.24.1 torchaudio==2.9.1 --index-url https://download.pytorch.org/whl/cu128
pip install torch-cluster -f https://data.pyg.org/whl/torch-2.9.1+cu128.html
pip install --no-build-isolation git+https://github.com/mattloper/chumpy.git@580566e
pip install -e .

# 2) Download required assets listed below
```

- **MANO**: [Register](https://mano.is.tue.mpg.de/) → download and unzip the MANO models → copy contents of `mano_v*_*/` to `assets/mano_models/`
- **DINOv2**: Auto-downloads on first use
- **HUG weights**: `hf download kevinywu/hug hug_full.safetensors --local-dir checkpoints/`

## 🚀 Usage

<p align="center">
  <img src="assets/img/teaser.webp" alt="HUG teaser" width="100%"/>
</p>

HUG predicts human grasps in MANO form for selected objects in the camera frame. Currently, only inference is supported. We provide sample inputs of one image from each scene in HUG-Bench.

```bash
CKPT=checkpoints/hug_full.safetensors
DATA=data/hug_bench/

# App: click an object to predict a grasp
# --save-pred saves each clicked prediction to $DATA/grasp_pred/
python -m hug.app --checkpoint-path "$CKPT" --dataset-path "$DATA" --save-pred
```

If predictions are saved with `--save-pred`, you can visualize them with:

```bash
python -m hug.visualize_predictions --dataset-path "$DATA"
```

### Custom inputs

You can also run inference on your own captures. Put three files in one folder, we provide an example in `data/custom/` for a ZED 2i output.

- **RGB**: 8-bit image ("`rgb.png`"/"`rgb.jpg`"), any H×W, grayscale is also supported.
- **Depth**: 16-bit single-channel PNG ("`depth.png`" in `uint16`), **millimeter** units, same H×W as RGB and registered to it. Use [S2M2](https://junhong-3dv.github.io/s2m2-project/) to estimate depth for best results.
- **Intrinsics**: text file ("`intrinsics.txt`") at the RGB resolution: either four numbers `fx fy cx cy` or a 3×3 K matrix. `.npy`/`.json` also accepted.

```bash
# Prepare inputs writes <stem>.pkl into the folder
python -m hug.prepare_inputs --dataset-path data/custom
python -m hug.app --checkpoint-path "$CKPT" --dataset-path data/custom --save-pred
```

## 📝 Citation

If you find our work useful, please consider citing our paper:

```bibtex
@article{wu2026hug,
  title={Human Universal Grasping},
  author={Kevin Yuanbo Wu and Tianxing Zhou and Isaac Tu and Billy Yan and Irmak Guzey and David Fouhey and Dandan Shan and Lerrel Pinto},
  journal={arXiv preprint arXiv:2606.17054},
  year={2026}
}
```
