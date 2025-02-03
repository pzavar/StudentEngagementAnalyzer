# Student Engagement Analysis System

A web-based AI pipeline for analyzing student engagement from recorded Zoom lectures using emotion recognition.

## Overview

This system helps professors analyze student engagement during recorded lectures by:
- Processing video footage to detect and track student faces
- Classifying emotions using a trained CNN model
- Mapping emotions to engagement levels
- Generating comprehensive engagement reports with visualizations

## Project Structure

```
├── src/
│   ├── components/          # Streamlit UI components
│   ├── models/             # ML model implementations
│   ├── training/           # Model training scripts
│   └── utils/              # Utility functions
├── .streamlit/             # Streamlit configuration
├── app.py                  # Main application
└── README.md              # Documentation
```

## Local Setup with Jupyter (MacBook)

### 1. Environment Setup

1. Install Anaconda if you haven't already:
   - Visit [Anaconda Downloads](https://www.anaconda.com/download)
   - Download and install Anaconda for macOS

2. Open Terminal and create a new conda environment:
```bash
conda create -n engagement python=3.8
conda activate engagement
```

3. Install required packages:
```bash
conda install jupyter tensorflow pandas numpy matplotlib seaborn scikit-learn
conda install -c conda-forge opencv
pip install streamlit plotly
```

### 2. Training the Model

1. Download the [Emotion Recognition Dataset](https://www.kaggle.com/datasets/sujaykapadnis/emotion-recognition-dataset/data) from Kaggle
   - You'll need a Kaggle account
   - Download and extract the dataset to a folder on your MacBook

2. Open Jupyter Notebook:
```bash
jupyter notebook
```

3. Navigate to the project directory and open `train_emotion_classifier.ipynb`

4. Follow the step-by-step instructions in the notebook:
   - Update the dataset path in the notebook to match your local path
   - Run each cell in sequence
   - The training process might take a while depending on your MacBook's specifications
   - The trained model will be saved as `best_model.h5`

### 3. Running the Application

1. Copy the generated `best_model.h5` to the project root directory

2. Start the Streamlit application:
```bash
streamlit run app.py
```

3. Open your browser and navigate to http://localhost:8501

## Using the Application

1. Upload a recorded Zoom lecture video
2. Configure analysis parameters:
   - Sampling rate (seconds between frame captures)
   - Confidence threshold for emotion detection
3. View the generated engagement report:
   - Overall class engagement score
   - Individual student engagement timelines
   - Class-wide engagement trends
   - Engagement level distribution

## Model Architecture

The emotion classification model uses MobileNetV2 as the base model, with additional layers:
- Global Average Pooling
- Dropout (0.5)
- Dense layer (256 units, ReLU activation)
- Dropout (0.3)
- Output layer (5 units, softmax activation)

The model is optimized for:
- Multi-face detection in Zoom grid layouts
- Consistent student tracking across frames
- Real-time processing of video frames
- Handling varying lighting conditions

## Video Processing Features

- Enhanced face detection optimized for Zoom grid layouts
- Automatic lighting adjustment for better face detection
- Multi-face tracking with student ID persistence
- Configurable sampling rate and confidence thresholds
- Frame quality enhancement for better detection accuracy

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- [Kaggle Emotion Recognition Dataset](https://www.kaggle.com/datasets/sujaykapadnis/emotion-recognition-dataset/data)
- TensorFlow and Keras teams
- Streamlit community

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.