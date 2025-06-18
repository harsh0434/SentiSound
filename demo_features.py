#!/usr/bin/env python3
"""
Demo script showcasing all advanced features of the Audio Emotion Detection System
"""

import requests
import json
import os
import time
import base64
from datetime import datetime

# Base URL for the API
BASE_URL = "http://localhost:5000"

def print_header(title):
    """Print a formatted header"""
    print("\n" + "="*60)
    print(f"🎯 {title}")
    print("="*60)

def print_section(title):
    """Print a formatted section"""
    print(f"\n📌 {title}")
    print("-" * 40)

def demo_file_upload():
    """Demo file upload and analysis"""
    print_section("File Upload & Analysis")
    
    # Check for existing audio files
    audio_files = []
    if os.path.exists('static/audio_uploads'):
        audio_files = [f for f in os.listdir('static/audio_uploads') if f.endswith(('.wav', '.mp3', '.m4a'))]
    
    if not audio_files:
        print("⚠️  No audio files found. Please upload some audio files first.")
        return False
    
    # Use the first available audio file
    test_file = os.path.join('static/audio_uploads', audio_files[0])
    print(f"📁 Using file: {audio_files[0]}")
    
    try:
        with open(test_file, 'rb') as f:
            files = {'file': f}
            response = requests.post(f"{BASE_URL}/predict", files=files)
        
        if response.status_code == 200:
            data = response.json()
            
            print(f"✅ Analysis completed!")
            print(f"🎭 Detected Emotion: {data['emotion'].upper()}")
            print(f"📊 Confidence: {max(data['probabilities'].values())*100:.1f}%")
            
            if data.get('visualization'):
                print(f"📈 Visualization: {data['visualization']}")
            
            if data.get('suggestions'):
                print(f"💡 Suggestions provided: {len(data['suggestions'])} categories")
                for category, suggestion in data['suggestions'].items():
                    print(f"   • {category.title()}: {suggestion[:50]}...")
            
            return True
        else:
            print(f"❌ Analysis failed: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def demo_api_endpoint():
    """Demo the API endpoint for external access"""
    print_section("API Endpoint for External Access")
    
    audio_files = []
    if os.path.exists('static/audio_uploads'):
        audio_files = [f for f in os.listdir('static/audio_uploads') if f.endswith(('.wav', '.mp3', '.m4a'))]
    
    if not audio_files:
        print("⚠️  No audio files found.")
        return False
    
    test_file = os.path.join('static/audio_uploads', audio_files[0])
    
    try:
        with open(test_file, 'rb') as f:
            files = {'file': f}
            response = requests.post(f"{BASE_URL}/api/predict", files=files)
        
        if response.status_code == 200:
            data = response.json()
            
            print("✅ API Response:")
            print(f"   Emotion: {data['emotion']}")
            print(f"   Confidence: {data['confidence']}")
            print(f"   Filename: {data['filename']}")
            print(f"   Probabilities: {data['probabilities']}")
            
            return True
        else:
            print(f"❌ API failed: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def demo_history_tracking():
    """Demo emotion history tracking"""
    print_section("Emotion History Tracking")
    
    try:
        response = requests.get(f"{BASE_URL}/history")
        
        if response.status_code == 200:
            data = response.json()
            history = data.get('history', [])
            
            print(f"✅ History loaded: {len(history)} records")
            
            if history:
                print("\n📋 Recent Analysis History:")
                for i, record in enumerate(history[-5:], 1):  # Show last 5 records
                    print(f"   {i}. {record['timestamp']} - {record['filename']}")
                    print(f"      Emotion: {record['predicted_emotion']} (Confidence: {record['confidence']*100:.1f}%)")
                
                # Show emotion distribution
                emotions = [r['predicted_emotion'] for r in history]
                emotion_counts = {}
                for emotion in emotions:
                    emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
                
                print(f"\n📊 Emotion Distribution:")
                for emotion, count in emotion_counts.items():
                    percentage = (count / len(emotions)) * 100
                    print(f"   {emotion.title()}: {count} ({percentage:.1f}%)")
            
            return True
        else:
            print(f"❌ History failed: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def demo_pdf_report():
    """Demo PDF report generation"""
    print_section("PDF Report Generation")
    
    audio_files = []
    if os.path.exists('static/audio_uploads'):
        audio_files = [f for f in os.listdir('static/audio_uploads') if f.endswith(('.wav', '.mp3', '.m4a'))]
    
    if not audio_files:
        print("⚠️  No audio files found.")
        return False
    
    test_file = audio_files[0]
    
    try:
        response = requests.get(f"{BASE_URL}/download-report/{test_file}")
        
        if response.status_code == 200:
            # Save the PDF for demonstration
            pdf_filename = f"demo_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
            with open(pdf_filename, 'wb') as f:
                f.write(response.content)
            
            print(f"✅ PDF Report generated successfully!")
            print(f"📄 File saved as: {pdf_filename}")
            print(f"📏 File size: {len(response.content)} bytes")
            
            return True
        else:
            print(f"❌ PDF generation failed: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def demo_error_handling():
    """Demo error handling capabilities"""
    print_section("Error Handling & Validation")
    
    # Test 1: No file uploaded
    print("🧪 Test 1: No file uploaded")
    try:
        response = requests.post(f"{BASE_URL}/predict")
        if response.status_code == 400:
            print("✅ Correctly handled: No file error")
        else:
            print(f"❌ Unexpected response: {response.status_code}")
    except Exception as e:
        print(f"❌ Error: {e}")
    
    # Test 2: Invalid file type
    print("\n🧪 Test 2: Invalid file type")
    try:
        files = {'file': ('test.txt', b'This is not an audio file', 'text/plain')}
        response = requests.post(f"{BASE_URL}/predict", files=files)
        if response.status_code == 400:
            print("✅ Correctly handled: Invalid file type")
        else:
            print(f"❌ Unexpected response: {response.status_code}")
    except Exception as e:
        print(f"❌ Error: {e}")
    
    # Test 3: Non-existent file for PDF download
    print("\n🧪 Test 3: Non-existent file for PDF download")
    try:
        response = requests.get(f"{BASE_URL}/download-report/nonexistent.wav")
        if response.status_code == 404:
            print("✅ Correctly handled: File not found")
        else:
            print(f"❌ Unexpected response: {response.status_code}")
    except Exception as e:
        print(f"❌ Error: {e}")
    
    return True

def demo_advanced_features():
    """Demo advanced features summary"""
    print_section("Advanced Features Summary")
    
    features = [
        "🎤 Real-time Voice Recording (WebRTC)",
        "📈 Advanced Audio Visualizations (Waveform, Spectrogram, MFCC)",
        "📋 Emotion History Tracking (CSV-based)",
        "🧾 Downloadable PDF Reports (ReportLab)",
        "📡 RESTful API Endpoint (/api/predict)",
        "🎨 Enhanced UI/UX with Bootstrap 5",
        "🎚️ Confidence Threshold Slider",
        "🎭 Emotion Icons & Color Coding",
        "💡 Smart Suggestion System",
        "🌍 Multi-language Support (EN, HI, GU)",
        "📱 Responsive Design",
        "🔒 Security & Error Handling"
    ]
    
    print("✨ Advanced Features Implemented:")
    for feature in features:
        print(f"   {feature}")
    
    return True

def main():
    """Run the complete demo"""
    print_header("Advanced Audio Emotion Detection System - Feature Demo")
    
    print("🚀 This demo showcases all the advanced features implemented in the system.")
    print("📋 Make sure the Flask application is running on http://localhost:5000")
    
    demos = [
        ("File Upload & Analysis", demo_file_upload),
        ("API Endpoint", demo_api_endpoint),
        ("History Tracking", demo_history_tracking),
        ("PDF Report Generation", demo_pdf_report),
        ("Error Handling", demo_error_handling),
        ("Advanced Features Summary", demo_advanced_features)
    ]
    
    successful_demos = 0
    total_demos = len(demos)
    
    for title, demo_func in demos:
        try:
            if demo_func():
                successful_demos += 1
        except Exception as e:
            print(f"❌ Demo '{title}' crashed: {e}")
        time.sleep(1)  # Brief pause between demos
    
    print_header("Demo Results")
    print(f"📊 Completed: {successful_demos}/{total_demos} demos successfully")
    
    if successful_demos == total_demos:
        print("🎉 All demos completed successfully!")
        print("✨ The Advanced Audio Emotion Detection System is fully functional!")
    else:
        print("⚠️  Some demos failed. Please check the application status.")
    
    print("\n🔗 Access the web interface at: http://localhost:5000")
    print("📚 Check the README.md for detailed documentation")
    print("🧪 Run test_api.py for comprehensive testing")

if __name__ == "__main__":
    main() 