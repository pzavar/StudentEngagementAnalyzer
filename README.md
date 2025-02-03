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

## Installation

### Local Setup (macOS)

1. Clone the repository:
```bash
git clone https://github.com/pzavar/StudentEngagementAnalyzer.git
cd student-engagement-analysis
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

### Training the Model

1. Download the [Emotion Recognition Dataset](https://www.kaggle.com/datasets/sujaykapadnis/emotion-recognition-dataset/data) from Kaggle

2. Run the training script:
```bash
python src/training/train_model.py
```

The script will:
- Preprocess the dataset
- Train the model using EfficientNetB0 or MobileNetV2
- Generate training visualizations using seaborn
- Save the model in TFLite format

### Running the Application

1. Start the Streamlit server:
```bash
streamlit run app.py
```

2. Open your browser and navigate to `http://localhost:3000`

## Usage

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

The emotion classification model uses transfer learning with either EfficientNetB0 or MobileNetV2 as the base model, with additional layers:
- Global Average Pooling
- Dropout (0.5)
- Dense layer (256 units, ReLU activation)
- Dropout (0.3)
- Output layer (7 units, softmax activation)



## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- [Kaggle Emotion Recognition Dataset](https://www.kaggle.com/datasets/sujaykapadnis/emotion-recognition-dataset/data)
- TensorFlow and Keras teams
- Streamlit community

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
