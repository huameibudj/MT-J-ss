# MT-J-ss

## Multi-Task Joint Self-Supervised Learning for Histopathological Image Segmentation

**MT-J-ss** is the official repository for the study **“Multi-Task Joint Self-Supervised Learning for Histopathological Image Segmentation.”** The proposed framework is designed to improve Vision Transformer (ViT) representations for dense prediction in computational pathology by jointly modeling local semantic discrimination, masked semantic reconstruction, and multi-scale structural consistency.

**Authors:** Yiping Jiao (first and corresponding author), Hongxiang Lin  
**Affiliation:** Jiangsu Key Laboratory of Intelligent Medical Image Computing, School of Artificial Intelligence, Nanjing University of Information Science and Technology, Nanjing, China.

---

## Overview

Recent pathology foundation models trained with self-supervised learning have shown strong transfer performance, but their behavior on histopathological image segmentation is still insufficiently explored. Generic self-supervised objectives are often optimized for image-level representation learning, while segmentation requires fine-grained local semantics, spatial structure, and boundary-aware representations.

MT-J-ss addresses this gap with a **student–teacher self-supervised pre-training framework** built on a ViT backbone. The framework combines three complementary objectives:

1. **Patch-level student–teacher semantic alignment**  
   Patch representations from strongly and weakly augmented views are aligned through a teacher–student mechanism. Sparse patch sampling and entropy-based confidence weighting are used to reduce noisy supervision.

2. **Masked Patch Reconstruction (MPR)**  
   Instead of reconstructing raw pixels, the student predicts teacher-generated patch features at masked locations, encouraging semantic recovery from surrounding context.

3. **Multi-scale structural consistency**  
   Fine-grained regional consistency, coarse-grained regional consistency, and local gradient consistency are jointly used to preserve spatial organization across scales.

The self-supervised weights are subsequently transferred to an **Encoder-only Mask Transformer (EoMT)** segmentation model for downstream histopathological image segmentation.

---

## Main Contributions

- A multi-task self-supervised framework tailored to histopathological image segmentation.
- Patch-level semantic alignment that explicitly models local relationships rather than only global image representations.
- Feature-level masked patch reconstruction using teacher features as semantic targets.
- Multi-scale structural consistency for preserving local and global tissue organization.
- Entropy-based confidence weighting to suppress uncertain patch-level supervision.
- Evaluation on multiple lung- and prostate-cancer segmentation datasets, including cross-dataset and cross-cancer transfer settings.

---

## Datasets

### Self-supervised pre-training datasets

| Dataset | Source | Whole-slide images | Image patches | Patch size |
|---|---|---:|---:|---:|
| LUNG | TCGA-LUAD + TCGA-LUSC | 1,021 | 105,302 | 224 × 224 |
| PRAD | TCGA-PRAD | 448 | 44,902 | 224 × 224 |

### Downstream segmentation datasets

| Dataset | Task / cancer type | Number of image–mask pairs |
|---|---|---:|
| Lung-cancer | Lung adenocarcinoma / lung squamous-cell carcinoma segmentation | 18,195 |
| GLASS-AI | Lung adenocarcinoma segmentation | 70,736 |
| RINGS | Prostate-cancer segmentation | 1,501 |

> **Note:** Dataset licenses and redistribution conditions should be respected. For third-party datasets, users should follow the original data providers' access and usage requirements. Class definitions should follow the annotations distributed with each dataset.

---


## Method Components

The complete pre-training objective combines:

- **PCL**: patch-level contrastive / student–teacher alignment loss;
- **MPR**: masked patch reconstruction loss;
- **STRUCT**: multi-scale structural consistency loss.

The structural term includes:

- fine-grained regional consistency;
- coarse-grained regional consistency;
- local gradient consistency.

The teacher parameters are updated using an **exponential moving average (EMA)** of the student parameters.

---

## Code and Data Availability

All data and source code associated with this study are publicly available through this repository:

**https://github.com/huameibudj/MT-J-ss**

For datasets originating from external sources, this repository should provide either the permissible data files or the corresponding download / preprocessing instructions, depending on the original dataset license.

---

## Getting Started

Clone the repository:

```bash
git clone https://github.com/huameibudj/MT-J-ss.git
cd MT-J-ss
```

The exact environment, training entry points, configuration files, and evaluation commands should match the released implementation. Please document them in this repository together with the source code, for example through `requirements.txt`, `environment.yml`, configuration files, and executable training / evaluation scripts.

A complete release should document at least:

- environment and package versions;
- dataset preparation and directory structure;
- self-supervised pre-training commands;
- downstream EoMT fine-tuning commands;
- evaluation commands;
- checkpoint locations;
- random seeds and important hyperparameters.

---

## Reproducibility

To make the experiments reproducible, we recommend reporting or releasing the following together with the implementation:

- data split information;
- patch extraction strategy;
- augmentation configuration;
- masking ratio;
- sampled patch number;
- teacher and student temperatures;
- EMA momentum;
- confidence-weighting parameters;
- loss weights and dynamic weighting schedule;
- optimizer, learning rate, batch size, and number of epochs;
- downstream segmentation settings.

---

## Citation

If you use this work, please cite the paper. Before the final bibliographic information is available, the following temporary citation can be used:

```bibtex
@article{jiao2026mtjss,
  title   = {Multi-Task Joint Self-Supervised Learning for Histopathological Image Segmentation},
  author  = {Jiao, Yiping and Lin, Hongxiang},
  year    = {2026},
  note    = {Manuscript submitted to Elsevier}
}
```

Please replace this entry with the final journal citation after publication.

---

## Acknowledgements

This work was supported by the National Key R&D Program of China (No. 2023YFC3402800), the National Natural Science Foundation of China (Nos. 62302228, 82330060, 82302291, 82441029, 62171230, 62101365, 92159301, 62301263, 62301265, 82302352, 62401272), and the Jiangsu Provincial Department of Science and Technology major project on frontier-leading basic research in technology (No. BK2023200).

---

## Contact

For questions regarding this work, please contact:

**Yiping Jiao**  
Corresponding author  
Jiangsu Key Laboratory of Intelligent Medical Image Computing  
School of Artificial Intelligence  
Nanjing University of Information Science and Technology  
Email: **ping@nuist.edu.cn**
