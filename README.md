🩹 Wound Dressing AI

An intelligent computer vision system that automatically detects wounds and virtually applies a realistic medical dressing on the affected area using advanced AI models.

This project combines object detection, segmentation, and image synthesis to simulate a professional wound dressing effect.

🚀 Features

✔ Automatic wound detection using YOLO
✔ Precise wound segmentation using Segment Anything Model
✔ Adaptive bandage placement based on wound geometry
✔ Skin-tone aware adhesive color generation
✔ Realistic gauze texture simulation
✔ Arm curvature approximation for natural wrapping
✔ Lighting-aware blending for photorealistic results

## Model Files

Due to GitHub size limits, model files are hosted on Google Drive.

SAM Model:
https://drive.google.com/file/d/1IpCwYcg-plnlNGV3Eb9TsUsTNxfs2TfD/view

YOLO Wound Detection Model:
https://drive.google.com/file/d/12chkMt1UYwSkOCb-obPivMHENkQx5Aq7/view

Place them in the project root directory before running the code.

🧠 Workflow

Select an image containing a wound

Detect wound using YOLO model

Segment wound region using SAM

Estimate skin tone and lighting conditions

Generate synthetic medical dressing

Warp bandage to wound geometry

Blend dressing with original image

Result → AI-generated realistic wound dressing

🛠 Technologies Used

Python

OpenCV

Ultralytics YOLO

Segment Anything Model

NumPy

📂 Requirements
pip install ultralytics
pip install opencv-python
pip install numpy
pip install segment-anything

Download required models:

YOLO wound detection model → wound.pt

SAM checkpoint → sam_vit_b_01ec64.pth

▶ Run the Project
python wound_dressing.py

Select an image when prompted and the system will automatically generate the AI-based medical dressing visualization.

📌 Use Cases

Medical AI research

Automated wound analysis systems

Surgical planning simulations

Healthcare computer vision applications

⚠ Disclaimer

This project is intended for research and visualization purposes only and should not replace professional medical diagnosis or treatment.
