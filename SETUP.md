This document explains how to set up the file structure of this project.

To clone our GitHub Repository, run:
```
git clone https://github.com/Aeromasonn/Short-Horizon-Temporal-Optical-Flow-with-Physics-Informed-Consistency.git
```

Ideal Structure Layout:
```
├─Ckpts
│  └─ YOUR_Model.pt             # Checkpoints, Pretrained Models
├─Data                          # IMPORTANT: The KITTI Flow 2015 Dataset
│  ├─Additional_frames
│  │  ├─testing
│  │  │  ├─image_2
│  │  │  └─image_3
│  │  └─training
│  │      ├─image_2
│  │      └─image_3
│  └─Flow
│      ├─testing
│      │  ├─image_2
│      │  └─image_3
│      └─training
│          ├─disp_noc_0
│          ├─disp_noc_1
│          ├─disp_occ_0
│          ├─disp_occ_1
│          ├─flow_noc
│          ├─flow_occ
│          ├─image_2
│          ├─image_3
│          ├─label_2
│          ├─obj_map
│          └─viz_flow_occ
├─Downstream                    # Scripts for Downstream Task: Motion Detection
├─Images                        # Images
│  └─Readme_Supplements
├─Model                         # FULL Model
│  ├─neuralop_seg               # IMPORTANT: The Neural Operator Library
│  │  ├─__pycache__             # The .py files here are the LEAST ESSENTIAL dependencies 
│  │  ├─config.json             # for UNO Implementation
│  │  ├─DataLoader.py
│  │  ├─Decoders.py
│  │  ├─Detector.py
│  │  ├─Encoders.py
│  │  ├─stats.json
│  │  ├─train.py
│  │  ├─trainer.py
│  │  ├─visualization.py
│  ├─config.json                # Configuration file for running train.py
│  ├─DataLoader.py
│  ├─Detector.py
│  ├─Encoders.py
│  ├─stats.json                 # Optional to keep: Global statistics for RGB normalization
│  ├─train.py
│  ├─trainer.py
│  └─visualization.py
├─Notebooks                     # All Example Scripts For the Model
│  │                            # NOTE: .py files in this directory are best to keep as-is
│  │                            # to prevent version conflict with those in .\Model\.
│  │                            # The files implement the exact same framework nevertheless.
│  ├─__pycache__
│  ├─Bounding_Box_test_KITTI Label.ipynb
│  ├─Encoders.py
│  ├─FullPipeline_Early-Integration.ipynb
│  ├─FullPipeline_Late-Integration.ipynb
│  ├─FullPipeline_Standalone.ipynb
│  ├─stats.json
│  └─Trainers.py
├─Reports
└─utils                         # .json File Templates
│  ├─config.json
│  └─stats.json
```

The scripts and notebooks are established and arranged based on this structure.