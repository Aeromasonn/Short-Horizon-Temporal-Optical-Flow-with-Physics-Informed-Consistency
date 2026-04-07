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

Pipeline
![Workflow1](Images/Readme_Supplements/Workflow%201.PNG)
![Workflow2](Images/Readme_Supplements/Workflow%202.PNG)

Outcome
Stage 1: With sobel operator, without self-supervised; After 50 epochs
- sample1:
![sample_1](Images/Readme_Supplements/sample_1.png)
- sample2:
![sample_2](Images/Readme_Supplements/sample_2.png)
- sample3:
![sample_3](Images/Readme_Supplements/sample_3.png)
- sample4:
![sample_4](Images/Readme_Supplements/sample_4.png)
- loss(After 50 epochs)
![loss_50_epoch](Images/Readme_Supplements/loss_50_epoch.png)

Stage 2: Adding Self_supervise:
- sample1:
![self_supervised_1](Images/Readme_Supplements/self_supervised_1.png)
- sample2:
![self_supervised_1](Images/Readme_Supplements/self_supervised_1.png)
- loss(After 50 epochs)
![loss_50_epoch](Images/Readme_Supplements/loss_50_epoch_supervised.png)