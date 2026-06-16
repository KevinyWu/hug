<h1 align="center">Human Universal Grasping</h1>

<p align="center"><i>Learning dexterous multifingered grasping entirely from human data.</i></p>

<p align="center">
  <a href="https://arxiv.org/abs/2606.17054"><img src="https://img.shields.io/badge/arXiv-2606.17054-b31b1b.svg" alt="arXiv"></a>
  <a href="https://arxiv.org/pdf/2606.17054"><img src="https://img.shields.io/badge/Paper-PDF-1f6feb.svg" alt="Paper PDF"></a>
  <a href="https://grasping.io"><img src="https://img.shields.io/badge/Project-Website-2ea44f.svg" alt="Project Website"></a>
  <a href="https://huggingface.co/kevinywu/hug"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Weights-yellow.svg" alt="Weights"></a>
</p>

<p align="center">
  <a href="https://kevinywu.github.io/">Kevin Yuanbo Wu</a><sup>1</sup>,
  <a href="https://ztx2021.github.io/">Tianxing Zhou</a><sup>1,2</sup>,
  <a href="https://www.linkedin.com/in/isaactu7/">Isaac Tu</a><sup>1</sup>,
  <a href="https://billy-yibo-yan.github.io/">Billy Yan</a><sup>1</sup>,
  <a href="https://irmakguzey.github.io/">Irmak Guzey</a><sup>1</sup>,
  <br/>
  <a href="https://cs.nyu.edu/~fouhey/">David Fouhey</a><sup>1</sup>,
  <a href="https://ddshan.github.io/">Dandan Shan</a><sup>1,3,‡</sup>,
  <a href="https://www.lerrelpinto.com/">Lerrel Pinto</a><sup>1,‡</sup>
  <br/>
  <sup>1</sup>New York University &nbsp; <sup>2</sup>Tsinghua University &nbsp; <sup>3</sup>University of Michigan
  <br/>
  <sup>‡</sup>Equal advising
</p>

<p align="center">
  <img src="assets/img/hug_demo.gif" alt="HUG demo" width="100%"/>
</p>

## 🗓️ Release

- [x] Paper and website
- [x] Inference + visualization code
- [ ] `1M-HUGs` dataset (planned 2026/06/29)
- [ ] `HUG-Bench` benchmark, assets + sim eval (planned 2026/06/29)
- [ ] Training code (planned 2026/06/29)

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

- **MANO**: [Register](https://mano.is.tue.mpg.de/) → download and unzip the MANO models → copy contents of `mano_v*_*/` to `assets/mano/`
- **DINOv2**: Auto-downloads on first use
- **HUG weights**: `hf download kevinywu/hug hug_full.safetensors --local-dir checkpoints/`

## 🚀 Usage

HUG predicts human grasps in MANO form for selected objects in the camera frame.Currently, only inference is supported. We provide sample inputs of one image from each scene in HUG-Bench.

```bash
CKPT=checkpoints/hug_full.safetensors
DATA=data/hug_bench/

# App: click an object to predict a grasp
# --save-pred saves each clicked prediction to $DATA/grasp_pred/
python -m hug.app --checkpoint-path "$CKPT" --dataset-path "$DATA" --save-pred

# Visualize saved predictions
python -m hug.visualize_predictions --dataset-path "$DATA"
```

### Custom inputs

You can also run inference on your own captures. Put three files in one folder, we provide an example in `data/custom/` for a ZED 2i output.

- **RGB**: 8-bit image ("`rgb.png`"/"`rgb.jpg`"), any H×W, grayscale is also supported.
- **Depth**: 16-bit single-channel PNG ("`depth.png`" in `uint16`), **millimeter** units, same H×W as RGB and registered to it. Use [S2M2](https://junhong-3dv.github.io/s2m2-project/) for best results.
- **Intrinsics**: text file ("`intrinsics.txt`") at the RGB resolution: either four numbers `fx fy cx cy` or a 3×3 K matrix. `.npy`/`.json` also accepted.

```bash
# Prepare inputs writes <stem>.pkl into the folder
python -m hug.prepare_inputs --dataset-path data/custom
python -m hug.app --checkpoint-path "$CKPT" --dataset-path data/custom --save-pred
```

> **Note**: `--dataset-path` is any folder of `.pkl` samples (searched recursively; the `grasp_pred/` output dir is skipped). With `--save-pred`, each click in `app.py` writes a new `grasp_pred/<name>_<datetime_ms>.pkl`, mirroring the input layout. `visualize_predictions.py` then reads those saved predictions; run it after saving at least one.

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
