# Supermarket Shelf Detector

A tool that finds and identifies products and price tags on supermarket shelf images using artificial intelligence.

![image](https://github.com/user-attachments/assets/111cb80e-48ed-4da3-9ae0-a150352590d0)

![image](https://github.com/user-attachments/assets/f6617379-3805-440b-a9fc-ee653852382f)

![image](https://github.com/user-attachments/assets/4eec49ca-7845-4a55-b1e8-476b3b7fb69d)

## What This Tool Does

This app lets you:
- Automatically detect products and price tags in supermarket shelf images
- Upload your own images or use Kaggle's supermarket shelves dataset
- See detailed information about what was found in each image
- Analyze image properties like brightness and contrast

## Requirements

To run this app, you need:

```
streamlit
ultralytics
opencv-python
numpy
pandas
matplotlib
pillow
kagglehub
```

## How to Install

1. Clone this repository or download the app.py file
2. Install the required packages:

```bash
pip install streamlit ultralytics opencv-python numpy pandas matplotlib pillow kagglehub
```

3. Place your trained model in the same folder as the app (if you have one)

## How to Run

Start the app with:

```bash
streamlit run app.py
```

## Features

### Detection Options
- **YOLOv8 Detection**: Uses advanced AI to find products and price tags
- **Basic Contour Detection**: A simpler method that outlines shapes in the image

### Analysis
- Shows the number of products and price tags found
- Calculates image brightness and contrast
- Creates charts showing detection distribution

### Image Sources
- **Upload Your Own**: Test with your own supermarket shelf photos
- **Kaggle Dataset**: Use the built-in dataset of supermarket shelves

## Using Your Own Model

1. Train a YOLOv8 model to detect supermarket items
2. Save your model as "supermarket_detector.pt" in the app folder
3. Or enter a custom path to your model in the app's sidebar

## Getting the Dataset

The app can automatically download the Kaggle supermarket shelves dataset for you:

1. Click "Download Kaggle Dataset" in the sidebar
2. Wait for the download to complete
3. The app will remember the dataset location

## Image Processing

When you select or upload an image:

1. The app shows the original image on the left
2. The processed image with detections on the right
3. A table lists all detected objects
4. Statistics show counts for each object type

## Customizing the App

You can adjust:
- The confidence threshold (how certain the model must be to report a detection)
- Whether to show image metrics
- Which detection method to use

## How It Works

The app uses:
- YOLOv8, a powerful object detection model
- Streamlit for the web interface
- OpenCV for basic image processing
- Matplotlib for creating charts

## Troubleshooting

If you see "No objects detected":
- Try lowering the confidence threshold
- Check that your model is properly trained
- Ensure the image contains products or price tags

If the dataset path is not found:
- Use the "Download Dataset Now" button
- Or manually enter the correct path to your dataset folder


