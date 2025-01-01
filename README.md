# Deep-fake-image-detector Detector App

A Streamlit-based web application that detects AI-generated (deepfake) images using various deep learning models. The app uses state-of-the-art architectures like EfficientNet and provides real-time analysis of uploaded images.

## Features

- Image deepfake detection
- Support for multiple image formats (JPG, JPEG, PNG)
- User-friendly interface built with Streamlit
- Real-time analysis and results display
- Pre-trained models for accurate detection
- Color-coded results for easy interpretation

## Tech Stack

- Python
- Streamlit
- PyTorch
- PIL (Python Imaging Library)
- BlazeFace (for face detection)
- EfficientNet (for classification)

## Prerequisites

Before running the application, make sure you have the following installed:

```bash
- Python 3.7+
- PyTorch
- Streamlit
- Pillow
- scipy
```

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd ai-generated-detector
```

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

3. Download the required model weights:
   - Place the BlazeFace weights in `blazeface/blazeface.pth`
   - Place the anchor file in `blazeface/anchors.npy`

## Usage

1. Start the Streamlit app:
```bash
streamlit run App.py
```

2. Open your web browser and navigate to the provided local URL (typically `http://localhost:8501`)

3. Upload an image using the file uploader

4. Click the "Check for Deepfake" button to analyze the image

5. View the results, which will be displayed with color-coding:
   - Green: Real image
   - Red: Fake/AI-generated image

## Project Structure

- `App.py`: Main Streamlit application file
- `api.py`: API handling and image processing functions
- `image.py`: Core image analysis and model inference logic
- `blazeface/`: Directory containing BlazeFace model files
- `architectures/`: Neural network architecture definitions

## Supported Models

The application supports multiple architectures:
- EfficientNetB4
- EfficientNetB4ST
- EfficientNetAutoAttB4
- EfficientNetAutoAttB4ST
- Xception

## Datasets

The models can be trained on:
- DFDC (Deep Fake Detection Challenge)
- FFPP (FaceForensics++)

## Error Handling

The application includes comprehensive error handling for:
- Invalid file types
- Processing errors
- Model inference issues
- File system operations

## Notes

- The application creates temporary files in an 'uploads' directory during processing
- Face detection is performed using BlazeFace before deepfake analysis
- The default confidence threshold is set to 0.5
- Processing time may vary depending on the image size and hardware capabilities

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.
