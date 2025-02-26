# FunctaFormer: Domain-Agnostic Super-Resolution
Super-resolution techniques have made remarkable progress in recent years, especially in the vision domain. However, extending these advances to other data modalities has been challenging due to the need for specialized architectures tailored to each domain. In this work, we introduce FunctaFormer, a novel domain-agnostic super-resolution method capable of upscaling any arbitrary data type. Our approach leverages Implicit Neural Representations (INRs) to model data as continuous functions, providing a unified framework across different modalities. FunctaFormer obtains competitive performance across various domains, including images, audio, videos, 3D shapes, manifold data, and LiDAR scans.

![FunctaFormer Architecture](img/header.png)
Samples of 4x super-resolution with FunctaFormer on low-resolution input (top) resulting in upsampled output (bottom). By
using INRs, FunctaFormer can upsample any data modality (shown here as 2D images, 1D audio waveforms, 3D shapes, polar coordinates
for manifold data, 2+1D videos, and LiDAR point clouds).

📄 Paper: access coming soon
## Overview
FunctaFormer is a novel super-resolution method that works across multiple data modalities, including:

🖼️ Images  
🎵 Audio  
🧊 3D Shapes  
🌍 Manifold Data  
🎥 Videos  
📡 LiDAR Scans  

Unlike traditional super-resolution methods that require domain-specific architectures, FunctaFormer leverages Implicit Neural Representations (INRs) to model data as continuous functions. This enables unified and scalable super-resolution across diverse data types.

## Features
✅ Domain-agnostic: Works on various data types without modification.  
✅ INR-based: Uses continuous function representations for better generalization.  
✅ Performance: Competitive results across multiple domains.  
✅ Flexible & Scalable: Can be extended to new data types easily.  

## Installation
### Data
coming soon
### Dependencies
To install the required dependencies, run:

```bash
git clone https://github.com/flavioschenker/functaformer.git
cd functaformer
pip install -r requirements.txt
```
## Usage
### Inference
To run FunctaFormer on your data, use the following command:
```bash
python -m upscale -t <task> -i <path/to/inputs> -o <path/to/outputs>
```
### Training
To train FunctaFormer from scratch:

```bash
python -m functa -t <task> -i <path/to/dataset>
```
## Citation
If you use FunctaFormer in your research, please cite:

```bibtex
@article{functaformer2024,
  title={FunctaFormer: Domain-Agnostic Super-Resolution using Implicit Neural Representations},
  author={Benjamin Estermann and Flavio Schenker and Luca A. Lanzendörfer and Roger Wattenhofer},
  journal={ArXiv},
  year={2024}
}
```
## License
This project is open-source under the MIT License, meaning you're free to use, modify, and distribute it with proper attribution.

## Contact and Acknowledgements
- Flavio Schenker (maintainer, Model implementation)
- Benjamin Estermann (first authorship)
- Luca A. Lanzendörfer (maintainer)