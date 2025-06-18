# 🎤 SentiSound - Advanced Audio Emotion Detection System

A sophisticated Flask-based web application that analyzes audio files to detect emotions using machine learning. This application includes cutting-edge features for real-time recording, advanced visualizations, and personalized user experiences.

## 🌟 Advanced Features Implemented

### 1. 🎤 Real-time Voice Recording
- **Browser-based recording**: Users can record their voice directly from the browser
- **WebRTC integration**: Uses MediaRecorder API for high-quality audio capture
- **Instant analysis**: Process recordings immediately after capture
- **Visual feedback**: Recording status with animated button and real-time indicators

### 2. 📈 Advanced Audio Visualizations
- **Waveform display**: Shows amplitude vs. time for raw audio signal
- **Spectrogram**: Frequency vs. time visualization with intensity mapping
- **MFCC heatmap**: Mel-frequency cepstral coefficients visualization
- **High-resolution output**: 300 DPI PNG files for detailed analysis
- **Interactive charts**: Responsive design with hover effects

### 3. 📋 Emotion History Tracking
- **Persistent storage**: CSV-based history with timestamps
- **Comprehensive data**: Filename, emotion, confidence, top 3 probabilities
- **Session management**: Track multiple analyses per session
- **Export capabilities**: Download history data for further analysis

### 4. 🧾 Downloadable PDF Reports
- **Professional reports**: Generated using ReportLab
- **Comprehensive content**: Emotion results, probabilities, timestamps
- **Styled formatting**: Professional tables and typography
- **Instant download**: One-click PDF generation and download

### 5. 📡 RESTful API Endpoint
- **External access**: `/api/predict` endpoint for developers
- **CORS enabled**: Cross-origin resource sharing support
- **JSON responses**: Structured data with confidence scores
- **Error handling**: Comprehensive error responses

### 6. 🎨 Enhanced UI/UX Features

#### Confidence Threshold Slider
- **Adjustable sensitivity**: Users can set confidence thresholds (0-100%)
- **Real-time feedback**: Dynamic threshold value display
- **Filtered results**: Only show emotions above threshold

#### Emotion Icons & Color Coding
- **Visual feedback**: Emoji icons for each emotion
- **Color-coded results**: 
  - 😄 Happy (Green)
  - 😢 Sad (Gray)
  - 😠 Angry (Red)
  - 😮 Surprised (Yellow)
  - 😨 Fear (Purple)
  - 🤢 Disgust (Orange)
  - 😐 Neutral (Blue)

#### Smart Suggestion System
- **Personalized recommendations**: Based on detected emotion
- **Multiple categories**: Music, activities, meditation
- **External links**: Direct YouTube music suggestions
- **Contextual advice**: Emotional wellness recommendations

### 7. 🌍 Multi-language Support
- **Language toggle**: English, Hindi, Gujarati
- **Dynamic translation**: Real-time language switching
- **Cultural adaptation**: Region-specific content

### 8. 📱 Modern Responsive Design
- **Mobile-first**: Optimized for all device sizes
- **Tabbed interface**: Organized feature sections
- **Floating actions**: Quick access buttons
- **Smooth animations**: CSS transitions and hover effects

## 🚀 Installation & Setup

### Prerequisites
```bash
Python 3.8+
pip
```

### Installation Steps

1. **Clone the repository**
```bash
git clone https://github.com/harsh0434/SentiSound.git
cd SentiSound
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Run the application**
```bash
python app.py
```

4. **Access the application**
```
http://localhost:5000
```

## 📁 Project Structure

```
SentiSound/
├── app.py                          # Main Flask application
├── requirements.txt                # Python dependencies
├── README.md                      # Project documentation
├── data/
│   └── emotion_history.csv        # Emotion analysis history
├── models/
│   ├── emotion_model.pkl          # Trained ML model
│   └── scaler.pkl                 # Feature scaler
├── static/
│   ├── audio_uploads/             # Uploaded audio files
│   └── visualizations/            # Generated audio visualizations
└── templates/
    └── index.html                 # Main web interface
```

## 🔧 API Documentation

### Main Prediction Endpoint
```http
POST /predict
Content-Type: multipart/form-data

Parameters:
- file: Audio file (WAV, MP3, M4A)

Response:
{
  "emotion": "happy",
  "probabilities": {
    "happy": 0.85,
    "sad": 0.10,
    "angry": 0.05
  },
  "audio_file": "uploaded_file.wav",
  "visualization": "visualizations/uploaded_file_analysis.png",
  "suggestions": {
    "music": "https://youtube.com/...",
    "activity": "Share your positive mood...",
    "meditation": "Try gratitude meditation..."
  },
  "emotion_config": {
    "emoji": "😄",
    "color": "#28a745",
    "bg_color": "#d4edda"
  }
}
```

### API Endpoint for External Access
```http
POST /api/predict
Content-Type: multipart/form-data

Response:
{
  "emotion": "happy",
  "confidence": 0.85,
  "probabilities": {
    "happy": 0.85,
    "sad": 0.10,
    "angry": 0.05
  },
  "filename": "audio.wav"
}
```

### Real-time Recording Endpoint
```http
POST /record
Content-Type: application/json

Body:
{
  "audio": "data:audio/wav;base64,UklGRnoGAABXQVZFZm10IBAAAAABAAEAQB8AAEAfAAABAAgAZGF0YQoGAACBhYqFbF1fdJivrJBhNjVgodDbq2EcBj+a2/LDciUFLIHO8tiJNwgZaLvt559NEAxQp+PwtmMcBjiR1/LMeSwFJHfH8N2QQAoUXrTp66hVFApGn+DyvmwhBSuBzvLZiTYIG2m98OScTgwOUarm7blmGgU7k9n1unEiBC13yO/eizEIHWq+8+OWT..."
}
```

### History Endpoint
```http
GET /history

Response:
{
  "history": [
    {
      "timestamp": "2024-01-15 14:30:25",
      "filename": "recording_20240115_143025.wav",
      "predicted_emotion": "happy",
      "confidence": 0.85,
      "top_3_probabilities": "{\"happy\": 0.85, \"sad\": 0.10, \"angry\": 0.05}"
    }
  ]
}
```

### PDF Report Download
```http
GET /download-report/{filename}

Response: PDF file download
```

## 🎯 Usage Examples

### 1. File Upload Analysis
1. Navigate to the "Upload Audio" tab
2. Select an audio file (WAV, MP3, M4A)
3. Adjust confidence threshold if needed
4. Click "Analyze Emotion"
5. View results with visualizations and suggestions

### 2. Real-time Recording
1. Navigate to the "Record Voice" tab
2. Click the microphone button to start recording
3. Speak clearly into your microphone
4. Click the stop button when finished
5. Click "Analyze Recording" to process

### 3. View History
1. Navigate to the "History" tab
2. View past analyses with timestamps
3. Download PDF reports for any entry
4. Track emotional patterns over time

### 4. API Integration
```python
import requests

# Upload and analyze audio file
with open('audio.wav', 'rb') as f:
    files = {'file': f}
    response = requests.post('http://localhost:5000/api/predict', files=files)
    result = response.json()
    print(f"Detected emotion: {result['emotion']}")
    print(f"Confidence: {result['confidence']}")
```

## 🛠️ Technical Implementation

### Machine Learning Pipeline
- **Feature Extraction**: MFCC (Mel-frequency cepstral coefficients)
- **Model**: Scikit-learn classifier (Random Forest/SVM)
- **Preprocessing**: Audio normalization and scaling
- **Validation**: Cross-validation for model accuracy

### Audio Processing
- **Format Support**: WAV, MP3, M4A
- **Sample Rate**: Automatic resampling to 22050 Hz
- **Duration**: Handles variable-length audio files
- **Quality**: High-fidelity processing with librosa

### Visualization Generation
- **Matplotlib**: Professional plotting library
- **Seaborn**: Enhanced statistical visualizations
- **Librosa**: Audio-specific visualization tools
- **Export**: High-resolution PNG files

### Web Technologies
- **Frontend**: HTML5, CSS3, JavaScript (ES6+)
- **Backend**: Flask (Python)
- **Audio**: WebRTC MediaRecorder API
- **Styling**: Bootstrap 5 + Custom CSS

## 🔒 Security Features

- **File Validation**: Secure filename handling
- **CORS Configuration**: Controlled cross-origin access
- **Input Sanitization**: Audio file type validation
- **Error Handling**: Comprehensive exception management

## 📊 Performance Optimizations

- **Async Processing**: Non-blocking audio analysis
- **Caching**: Visualization file caching
- **Memory Management**: Efficient audio processing
- **Response Time**: Optimized for real-time use

## 🎨 Customization Options

### Adding New Emotions
1. Retrain the model with new emotion classes
2. Update `EMOTION_CONFIG` in `app.py`
3. Add corresponding emojis and colors
4. Update suggestion system

### Custom Visualizations
1. Modify `create_visualizations()` function
2. Add new matplotlib subplots
3. Customize color schemes and layouts
4. Update frontend display logic

### Language Support
1. Add new language to `translations` object
2. Create language-specific UI elements
3. Update `setLanguage()` function
4. Test with native speakers

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Implement your changes
4. Add tests and documentation
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **Librosa**: Audio processing and analysis
- **Scikit-learn**: Machine learning framework
- **Flask**: Web framework
- **Bootstrap**: UI framework
- **Matplotlib**: Visualization library

## 📞 Support

For questions, issues, or feature requests:
- Create an issue on GitHub
- Contact the development team
- Check the documentation

---

**Made with ❤️ for emotion detection and analysis**

![image](https://github.com/user-attachments/assets/53bb5052-d31e-46b1-901f-f58effac7d8e)
![image](https://github.com/user-attachments/assets/4917f665-2e4c-4a12-b935-faf821927651)
![image](https://github.com/user-attachments/assets/832a76f5-1fe2-4a20-8d6e-c80bc5c6e92a)
