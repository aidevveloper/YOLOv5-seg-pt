# YOLOv5-seg-pt

A streamlined YOLOv5 segmentation implementation with clean code for better readability and maintenance.

## Installation

Install dependencies:
```bash
pip install -r requirements.txt
```

## Dataset 

Download data by the following link and put it in crack-seg:

```bash
# Download the dataset
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/crack-seg.zip

# Extract to crack-seg directory
unzip crack-seg.zip
```

## Quick Start

### Training

Train a segmentation model on your custom dataset:

```bash
python train.py --data crack-seg/crack-seg.yaml --weights yolov5s-seg.pt --img 640 --epochs 300
```

### Inference

```bash
python predict.py --weights runs/train-seg/exp/weights/best.pt --source path/to/video.mp4
```

## Results

<video width="640" height="480" controls>
  <source src="assets\demo.mp4" type="video/mp4">
  Your browser does not support the video tag.
</video>

## License

This project is based on YOLOv5 by Ultralytics. Please refer to the original license terms.

## Acknowledgments

- [Ultralytics YOLOv5](https://github.com/ultralytics/yolov5) - Original implementation
- Clean codebase modifications for improved readability