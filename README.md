# STATS-402 - Interdisciplinary Data Analysis
## Short-Horizon Temporal Optical Flow with Physics-Informed Consistency
<h3 align="center">Sihan Yao, Yuxuan Huang</h3>
<p align="center">
  <a href="mailto:sihan.yao@dukekunshan.edu.cn">sihan.yao@dukekunshan.edu.cn</a> · 
  <a href="mailto:y.huang@dukekunshan.edu.cn">y.huang@dukekunshan.edu.cn</a>
</p>

**Description:**
This project investigates **short-horizon spatiotemporal optical / scene flow estimation** by moving beyond 
the traditional two-frame formulation and treating motion as a temporally evolving spatial field. Instead of estimating
motion independently between image pairs, we leverage multi-frame sequences to model motion dynamics over time.

Our core idea is to bridge classical correspondence-based optical flow with **operator-based learning**, enabling the
model to capture both:
- Local pixel-wise motion (pairwise flow)
- Global spatiotemporal structure (motion evolution)

The pipeline integrates a pairwise flow encoder, a joint visual–motion fusion module, a U-shaped Fourier Neural Operator (UNO), and a PWC-style decoder to model spatiotemporal motion dynamics.A central objective of this work is to investigate how the placement of the neural operator affects model performance.

**What it does**
The framework takes a short sequence of consecutive frames as input and produces optical flow estimations for adjacent frame pairs. The model extracts motion and visual features, fuses them into a spatiotemporal representation, and refines this representation through a neural operator before decoding it into flow predictions.

We implement and compare three architecture variants:

(1) Standalone UNO (Original Formulation)

The UNO module is applied after the visual–motion fusion encoder, directly operating on fused spatiotemporal embeddings. Its output serves as the input to the decoder without additional residual connections. 
![Pipeline Overview](Images/Readme_Supplements/Standalone_UNO.png)

(2) Early Integration

The UNO module is inserted before the fusion stage, immediately after the pairwise flow encoder. In this setting, the operator acts on lower-level motion representations, influencing how motion and visual features are subsequently combined.
![Pipeline Overview](Images/Readme_Supplements/Early_integration.png)

(3) Late Integration (Refinement)

The UNO module is applied after the fusion encoder as a residual refinement module. Instead of replacing the fused representation, the operator outputs a latent update that is added to the original embedding before decoding. 
![Pipeline Overview](Images/Readme_Supplements/Late_Integration.png)

**Quick Start**
After SETUP.md
To train the Standalone Architechure, run:
```
python train.py --config config.json --architecture_type standalone
```
To train the Early Integration Architechure, run:
```
python train.py --config config.json --architecture_type early
```
To train the Late Integration Architechure, run:
```
python train.py --config config.json --architecture_type later
```
To visualize single frame result (example: Late Integration) with EPE map, run:
```
python visualization.py --config config.json --checkpoint checkpoints/fullpipeline_later_best.pth --architecture_type later --mode single
```
To visualize short-horizon multiple frames result, run:
```
python visualization.py --config config.json --checkpoint checkpoints/fullpipeline_later_best.pth --architecture_type later --mode multiple
```
To use other architectures, simply replace `later` with `early` or `standalone` in both `--checkpoint` and `--architecture_type`.

**Example Results**
We evaluate the performance of three architecture variants under the same training and experimental settings on KITTI 2025 training set. The comparison focuses on standard optical flow metrics, including End-Point Error (EPE), per-image EPE (F1-EPE), and outlier ratio (F1-all%).

| Model Variant        | EPE ↓    | F1-EPE ↓ | F1-all% ↓ |
|---------------------|----------|----------|-----------|
| Standalone          | 2.9234   | 2.7889   | **20.59** |
| Early Integration   | **2.6255** | **2.5596** | 22.11     |
| Late Integration    | 2.7647   | 2.6673   | 23.30     |

Below is a optical flow predictions for the Standalone variant. Row 1, figures 1 and 2 (left to right) are target images; row 1, figure 3, is the estimated flow; row 2, figure 1 the ground truth flow; row 2, figure 2 is the EPE map of our estimation.
![Standalone](Images/Readme_Supplements/Seperate.png)
Here is also an example of our short-horizon multiple prediction using the Standalone UNO Architechure: 
![Standalone_multiple](Images/Readme_Supplements/Standalone_Multiple.png)