<!-- PROJECT LOGO -->
<h1 align="center">
  <img src="https://visinf.github.io/glass/static/images/favicon.png" width="28" valign="middle">
  GLASS: Guided Latent Slot Diffusion for Object-Centric Learning
</h1>

<p align="center">
  <a href="https://openaccess.thecvf.com/content/cvpr2025/papers/singh_glass_guided_latent_slot_diffusion_for_object-centric_learning_cvpr_2025_paper.pdf"><img src="https://img.shields.io/badge/CVPR-2025-blue"></a>
  <a href="https://arxiv.org/pdf/2407.17929"><img src="https://img.shields.io/badge/arXiv-2407.17929-b31b1b.svg"></a>
  <a href="https://opensource.org/licenses/Apache-2.0"><img src="https://img.shields.io/badge/License-Apache_2.0-green.svg"></a>
  <a href="https://visinf.github.io/glass/"><img src="https://img.shields.io/badge/Project-Page-ff69b4.svg"></a>
  <a href="https://youtube.com/watch?v=n7JCtkmxP3A"><img src="https://img.shields.io/badge/Video-Youtube-red.svg"></a>
</p>

<p align="center">
  <b>Official repository of the CVPR 2025 paper</b><br>
  <a href="https://openaccess.thecvf.com/content/cvpr2025/papers/singh_glass_guided_latent_slot_diffusion_for_object-centric_learning_cvpr_2025_paper.pdf">
  GLASS: Guided Latent Slot Diffusion for Object-Centric Learning</a><br><br>
  <i>Krishnakant Singh, Simone Schaub-Meyer, and Stefan Roth</i><br>
  <a href="https://visinf.tu-darmstadt.de/">Visual Inference Lab, TU Darmstadt</a>
</p>

---

## <img src="https://visinf.github.io/glass/static/images/favicon.png" width="22" valign="middle"> Overview

**GLASS** introduces a diffusion-based framework for **object-centric representation learning**.  
It integrates **slot attention** with a **guided latent diffusion decoder** to learn **compositional, interpretable slot representations** that generalize across visual tasks:

- 🧠 **Unsupervised Object Discovery**  
- 🎨 **Image Generation & Reconstruction**  
- <img src="https://visinf.github.io/glass/static/images/favicon.png" width="18" valign="middle"> **Compositional Image Editing & Inpainting**

---

## 🧭 Quick Links

📄 [**Paper**](https://openaccess.thecvf.com/content/cvpr2025/papers/singh_glass_guided_latent_slot_diffusion_for_object-centric_learning_cvpr_2025_paper.pdf) | 🌐 [**Project Page**](https://visinf.github.io/glass/) | 📚 [**arXiv**](https://arxiv.org/pdf/2407.17929) | 🎥 [**Video**](https://youtube.com/watch?v=n7JCtkmxP3A)

---

<details>
<summary><b>🔧 Dependencies</b></summary>

```
Python >= 3.11  
PyTorch == 2.5.0  
CUDA == 11.8
```

</details>

<details>
<summary><b>⚙️ Environment Setup</b></summary>

```bash
conda create -n glass python==3.11.10
conda activate glass

# Install PyTorch and CUDA
conda install pytorch==2.5.0 torchvision==0.20.0 torchaudio==2.5.0 pytorch-cuda=11.8 -c pytorch -c nvidia

# Install remaining dependencies
pip install -r requirements.txt
```

</details>

<details>
<summary><b>💾 Pretrained Models</b></summary>

Pretrained checkpoints from the paper are available here:  
📥 [Google Drive Folder](https://drive.google.com/drive/folders/17VMkthwjXOtQkfAD5QCqtTxGZ_8Lyag4?usp=sharing)

Please unzip the folder and place the models under a top-level directory named `glass/`.

</details>

<details>
<summary><b>🖼️ Datasets</b></summary>

- **Evaluation:** Download the **COCO dataset** from the [official website](https://cocodataset.org/#home).  
- **Training:** Use [Dataset Diffusion](https://github.com/VinAIResearch/Dataset-Diffusion) to create generated images and pseudo-segmentation maps.
</details>

---

## 🚀 Evaluation

### 🧠 Object-Centric Segmentation
```bash
bash ./src/eval/scripts/coco/eval_oclf_metrics_coco.sh
```
This would create file ```metrics_coco.json``` file in the checkpoint folder.

### 🎨 Image Generation
```bash
bash ./src/eval/scripts/coco/eval_generation.sh
```

### <img src="https://visinf.github.io/glass/static/images/favicon.png" width="18" valign="middle"> Compositional Editing

We provide a very crude implementation for generation compositional images. 

```bash
bash ./src/eval/scripts/coco/eval_composition.sh
```

---

## 🧠 TODO

- [ ] Release full training pipeline  

---

## 📚 Citation

If you find this repository useful, please consider citing:

```bibtex
@inproceedings{singh2025glass,
  author    = {Krishnakant Singh and Simone Schaub-Meyer and Stefan Roth},
  title     = {GLASS: Guided Latent Slot Diffusion for Object-Centric Learning},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year      = {2025},
}
```

---

## 🙏 Acknowledgements

This repository builds upon  
**[LSD: Latent Slot Diffusion](https://github.com/JindongJiang/latent-slot-diffusion)**  and [Dataset Diffusion](https://github.com/VinAIResearch/Dataset-Diffusion).
We thank the authors for open-sourcing their work.

---

## 📜 License

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

---

## ✉️ Contact

**Krishnakant Singh**  
📧 firstname.lastname@visinf.tu-darmstadt.de  
🌐 [https://visinf.github.io/glass](https://visinf.github.io/glass)

---
