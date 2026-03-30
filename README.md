# STATS-402 - Interdisciplinary Data Analysis
## Short-Horizon Temporal Optical Flow with Physics-Informed Consistency
### Sihan Yao, Yuxuan Huang
### sihan.yao@dukekunshan.edu.cn, y.huang@dukekunshan.edu.cn

This project investigates **short-horizon spatiotemporal optical / scene flow estimation** by moving beyond 
the traditional two-frame formulation and treating motion as a temporally evolving spatial field. Instead of estimating
motion independently between image pairs, we leverage multi-frame sequences to model motion dynamics over time.

Our core idea is to bridge classical correspondence-based optical flow with **operator-based learning**, enabling the
model to capture both:
- Local pixel-wise motion (pairwise flow)
- Global spatiotemporal structure (motion evolution)

Pipeline
#[!Workflow1](Images/Readme_Supplements/Workflow%201.PNG)
#[!Workflow2](Images/Readme_Supplements/Workflow%202.PNG)