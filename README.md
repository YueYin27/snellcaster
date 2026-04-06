<h1 align="center" style="font-size: 36px; margin-bottom: 10px;">Refracting Reality: Generating Images with Realistic Transparent Objects</h1>

<div align="center" style="margin-bottom: 20px;">
  <a href="https://yueyin27.github.io">Yue Yin</a> ·
  <a href="https://enze-tao.github.io/">Enze Tao</a> ·
  <a href="https://sites.google.com/view/djcampbell">Dylan Campbell</a>
</div>

<p align="center">
	<a href="https://arxiv.org/abs/2511.17340">
		<img src="https://img.shields.io/badge/Paper-arXiv-red?logo=arxiv&logoColor=white" style="height: 27px; margin: 5px;">
	</a>&nbsp
	<a href="https://huggingface.co/datasets/yinyue27/Snellcaster">
		<img src="https://img.shields.io/badge/Dataset-HuggingFace-yellow?logo=huggingface&logoColor=white" style="height: 27px; margin: 5px;">
	</a>&nbsp
	<a href="https://yueyin27.github.io/snellcaster-page">
		<img src="https://img.shields.io/badge/Project-Website-blue?logo=google-chrome&logoColor=white" style="height: 27px; margin: 5px;">
	</a>
</p>

## ✨ Overview
Refracting Reality (accepted to CVPR 2026) introduces Snellcaster, a generation framework for synthesizing images with transparent objects that obey physically grounded optics. Given a text prompt, Snellcaster synchronizes pixels within the object’s boundary with those outside by warping and merging the pixels using Snell’s Law of Refraction at each step of the generation trajectory. For surfaces that are not directly observed in the image but are visible via refraction or reflection, we recover their appearance by synchronizing the image with a second generated view—a panorama centered at the object—using the same warping and merging procedure.


## 🚀 Quickstart

### 🛠️ Setup the Environment

```bash
conda create -n snell -y python=3.10 cmake=3.14.0
conda activate snell

# Example CUDA-specific installs (CUDA 12.8). Adapt to your system.
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -U xformers --index-url https://download.pytorch.org/whl/cu128

pip install --upgrade pip
pip install -r requirements.txt
```

### 🤖 Inference
1. Inference (with pre-processing):
   ```bash
   prompt="A beautiful landscape with a river and mountains, viewed from a camera positioned directly in front of a stone table and chairs in the immediate foreground, a solid transparent glass sphere on the table."
   python inference.py "$prompt" --scene_name "landscape"
   ```
2. Add shadows (post-processing):
	```bash
	python utils/add_shadows.py --image_name main_image.jpg --output_dir results/
	```

## 📑 Citation

```bibtex
@article{yin2025refracting,
	title={Refracting Reality: Generating Images with Realistic Transparent Objects},
	author={Yin, Yue and Tao, Enze and Campbell, Dylan},
    journal={arXiv preprint arXiv:2511.17340},
    year={2025}
}
```


## ⚖️ License
This project is licensed under the Apache License 2.0. See the [LICENSE](LICENSE) file for details.
