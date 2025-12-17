# References and Acknowledgments

## Inspiration and Related Work

### Kolmogorov-Arnold Networks (KAN)

FluxFlow's Bezier activation functions were **inspired by KAN**, which demonstrated the power of learnable activation functions based on the Kolmogorov-Arnold representation theorem. FluxFlow extends this concept by:
- Using cubic Bezier curves instead of B-splines
- Implementing dynamic parameter generation (parameters derived from input)
- Applying the approach to large-scale text-to-image generation

**Citation:**
```bibtex
@article{liu2024kan,
  title={KAN: Kolmogorov-Arnold Networks},
  author={Liu, Ziming and Wang, Yixuan and Vaidya, Sachin and Ruehle, Fabian and Halverson, James and Solja{\v{c}}i{\'c}, Marin and Hou, Thomas Y and Tegmark, Max},
  journal={arXiv preprint arXiv:2404.19756},
  year={2024},
  url={https://arxiv.org/abs/2404.19756}
}
```

**Resources:**
- Paper: [arXiv:2404.19756](https://arxiv.org/abs/2404.19756)
- GitHub: [KindXiaoming/pykan](https://github.com/KindXiaoming/pykan)

---

## Conditioning Mechanisms

### SPADE (Spatially-Adaptive Normalization)

Used in FluxFlow's VAE decoder for spatial conditioning.

**Citation:**
```bibtex
@inproceedings{park2019SPADE,
  title={Semantic Image Synthesis with Spatially-Adaptive Normalization},
  author={Park, Taesung and Liu, Ming-Yu and Wang, Ting-Chun and Zhu, Jun-Yan},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  year={2019},
  url={https://arxiv.org/abs/1903.07291}
}
```

**Resources:**
- Paper: [arXiv:1903.07291](https://arxiv.org/abs/1903.07291)
- Project: [NVIDIA GauGAN](https://nvlabs.github.io/SPADE/)
- GitHub: [NVlabs/SPADE](https://github.com/NVlabs/SPADE)

### FiLM (Feature-wise Linear Modulation)

Conditioning technique used in FluxFlow for text-based feature modulation.

**Citation:**
```bibtex
@inproceedings{perez2018film,
  title={FiLM: Visual Reasoning with a General Conditioning Layer},
  author={Perez, Ethan and Strub, Florian and De Vries, Harm and Dumoulin, Vincent and Courville, Aaron},
  booktitle={Thirty-Second AAAI Conference on Artificial Intelligence},
  year={2018},
  url={https://arxiv.org/abs/1709.07871}
}
```

**Resources:**
- Paper: [arXiv:1709.07871](https://arxiv.org/abs/1709.07871)

---

## Training Datasets

We gratefully acknowledge the following datasets used for development and testing:

### COCO 2017 & Open Images

Training utilized a combination of COCO 2017 images with captions from both COCO and Open Images datasets.

**COCO 2017:**
```bibtex
@inproceedings{lin2014microsoft,
  title={Microsoft COCO: Common Objects in Context},
  author={Lin, Tsung-Yi and Maire, Michael and Belongie, Serge and Hays, James and Perona, Pietro and Ramanan, Deva and Doll{\'a}r, Piotr and Zitnick, C Lawrence},
  booktitle={European Conference on Computer Vision},
  pages={740--755},
  year={2014},
  organization={Springer},
  url={https://arxiv.org/abs/1405.0312}
}
```

**Open Images V4:**
```bibtex
@article{kuznetsova2018open,
  title={The Open Images Dataset V4: Unified Image Classification, Object Detection, and Visual Relationship Detection at Scale},
  author={Kuznetsova, Alina and Rom, Hassan and Alldrin, Neil and Uijlings, Jasper and Krasin, Ivan and Pont-Tuset, Jordi and Kamali, Shahab and Popov, Stefan and Malloci, Matteo and Duerig, Tom and others},
  journal={arXiv preprint arXiv:1811.00982},
  year={2018}
}
```

**Resources:**
- COCO: [cocodataset.org](https://cocodataset.org/)
- COCO Paper: [arXiv:1405.0312](https://arxiv.org/abs/1405.0312)
- Open Images: [storage.googleapis.com/openimages/web/index.html](https://storage.googleapis.com/openimages/web/index.html)
- Open Images Paper: [arXiv:1811.00982](https://arxiv.org/abs/1811.00982)

**Special Thanks:** COCO 2017 images with mixed COCO/Open Images captions used for testing and validation.

### TTI-2M (Text-to-Image 2 Million)

A large-scale text-to-image dataset with 2 million image-text pairs used for training experiments.

**Resources:**
- HuggingFace: [jackyhate/text-to-image-2M](https://huggingface.co/datasets/jackyhate/text-to-image-2M)

**Special Thanks:** Used for larger-scale training experiments to validate model scalability and generalization.

---

## Architectural Influences

### Stable Diffusion & Latent Diffusion Models

FluxFlow's VAE-based latent diffusion approach builds upon concepts from:

**Citation:**
```bibtex
@inproceedings{rombach2022high,
  title={High-Resolution Image Synthesis with Latent Diffusion Models},
  author={Rombach, Robin and Blattmann, Andreas and Lorenz, Dominik and Esser, Patrick and Ommer, Bj{\"o}rn},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={10684--10695},
  year={2022},
  url={https://arxiv.org/abs/2112.10752}
}
```

**Resources:**
- Paper: [arXiv:2112.10752](https://arxiv.org/abs/2112.10752)
- GitHub: [CompVis/latent-diffusion](https://github.com/CompVis/latent-diffusion)

### Rotary Position Embeddings (RoPE)

Used in FluxFlow's transformer architecture for position encoding.

**Citation:**
```bibtex
@article{su2021roformer,
  title={RoFormer: Enhanced Transformer with Rotary Position Embedding},
  author={Su, Jianlin and Lu, Yu and Pan, Shengfeng and Wen, Bo and Liu, Yunfeng},
  journal={arXiv preprint arXiv:2104.09864},
  year={2021},
  url={https://arxiv.org/abs/2104.09864}
}
```

**Resources:**
- Paper: [arXiv:2104.09864](https://arxiv.org/abs/2104.09864)

---

## Discriminator Architecture

### PatchGAN & Spectral Normalization

**PatchGAN:**
```bibtex
@inproceedings{isola2017image,
  title={Image-to-Image Translation with Conditional Adversarial Networks},
  author={Isola, Phillip and Zhu, Jun-Yan and Zhou, Tinghui and Efros, Alexei A},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  pages={1125--1134},
  year={2017},
  url={https://arxiv.org/abs/1611.07004}
}
```

**Spectral Normalization:**
```bibtex
@inproceedings{miyato2018spectral,
  title={Spectral Normalization for Generative Adversarial Networks},
  author={Miyato, Takeru and Kataoka, Toshiki and Koyama, Masanori and Yoshida, Yuichi},
  booktitle={International Conference on Learning Representations},
  year={2018},
  url={https://arxiv.org/abs/1802.05957}
}
```

---

## Additional Acknowledgments

### Pre-trained Models

- **DistilBERT**: Used as temporary text encoder backbone
  - Hugging Face: [distilbert-base-uncased](https://huggingface.co/distilbert-base-uncased)
  - Paper: [arXiv:1910.01108](https://arxiv.org/abs/1910.01108)

### Frameworks and Tools

- **PyTorch**: Deep learning framework
- **Hugging Face Diffusers**: Pipeline infrastructure
- **Hugging Face Transformers**: Text encoding
- **SafeTensors**: Checkpoint format

---

## Community Contributions

We welcome contributions and acknowledge:
- Bug reports and feature requests from the community
- Testing and validation on various hardware platforms
- Documentation improvements and translations

---

## Citation

If you use FluxFlow in your research, please cite:

```bibtex
@software{fluxflow2024,
  title={FluxFlow: Efficient Text-to-Image Generation with Bezier Activation Functions},
  author={FluxFlow Contributors},
  year={2024},
  note={Inspired by Kolmogorov-Arnold Networks (KAN)},
  url={https://github.com/danny-mio/fluxflow-core}
}
```

---

## License

FluxFlow is released under the MIT License. See [LICENSE](LICENSE) for details.

All referenced works are cited according to their respective licenses and terms of use.
