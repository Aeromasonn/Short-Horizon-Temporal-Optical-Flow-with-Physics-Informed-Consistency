# STATS-402 - Interdisciplinary Data Analysis
## Short-Horizon Temporal Optical Flow with Physics-Informed Consistency
<h3 align="center">Sihan Yao, Yuxuan Huang</h3>
<p align="center">
  <a href="mailto:sihan.yao@dukekunshan.edu.cn">sihan.yao@dukekunshan.edu.cn</a> · 
  <a href="mailto:y.huang@dukekunshan.edu.cn">y.huang@dukekunshan.edu.cn</a>
</p>

This project investigates **short-horizon spatiotemporal optical / scene flow estimation** by moving beyond 
the traditional two-frame formulation and treating motion as a temporally evolving spatial field. Instead of estimating
motion independently between image pairs, we leverage multi-frame sequences to model motion dynamics over time.

Our core idea is to bridge classical correspondence-based optical flow with **operator-based learning**, enabling the
model to capture both:
- Local pixel-wise motion (pairwise flow)
- Global spatiotemporal structure (motion evolution)

**Pipeline**
![Workflow1](Images/Readme_Supplements/Workflow%201.PNG)
![Workflow2](Images/Readme_Supplements/Workflow%202.PNG)

**Outcome as of Milestone 2**
Implementation: Full framework except from the Neural Operator module.
Stage 1: With Sobel operator (enhance edge extraction), without self-supervision; 50 epochs
- Example1:
![sample_1](Images/Readme_Supplements/sample_1.png)
- Example2:
![sample_2](Images/Readme_Supplements/sample_2.png)
- Example3:
![sample_3](Images/Readme_Supplements/sample_3.png)
- Example4:
![sample_4](Images/Readme_Supplements/sample_4.png)
- loss(After 50 epochs)
![loss_50_epoch](Images/Readme_Supplements/loss_50_epoch.png)

Stage 2: Adding self-supervision. Trained in DKUCC for 200 epochs:
- Example1:
![stage2_1](Images/Readme_Supplements/stage2_1.png)
- Example2:
![stage2_2](Images/Readme_Supplements/stage2_2.png)
- Example3:
![stage2_3](Images/Readme_Supplements/stage2_3.png)
- loss after 200 epochs:
![loss_200_epoch_stage3](Images/Readme_Supplements/stage2_loss_200_epoch.png)

Stage3: Adding edge awareness loss. Trained in DKUCC for 200 epochs:
- Example1:
![stage3_1](Images/Readme_Supplements/stage3_1.png)
- sample2:
![stage3_2](Images/Readme_Supplements/stage3_2.png)
- sample3:
![stage3_3](Images/Readme_Supplements/stage3_3.png)
- loss after 200 epochs:
![loss_200_epoch_stage3](Images/Readme_Supplements/stage3_loss_200_epoch.png)

- supervised_predict example:
![supervised_example_1](Images/Readme_Supplements/supervised_example_1.png)