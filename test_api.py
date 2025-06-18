#!/usr/bin/env python3
"""
Test script for the Advanced Audio Emotion Detection API
Tests all endpoints and features
"""

import requests
import json
import os
import time

# Base URL for the API
BASE_URL = "http://localhost:5000"

def test_home_page():
    """Test the home page loads correctly"""
    print("🧪 Testing home page...")
    try:
        response = requests.get(f"{BASE_URL}/")
        if response.status_code == 200:
            print("✅ Home page loads successfully")
            return True
        else:
            print(f"❌ Home page failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Home page error: {e}")
        return False

def test_history_endpoint():
    """Test the history endpoint"""
    print("🧪 Testing history endpoint...")
    try:
        response = requests.get(f"{BASE_URL}/history")
        if response.status_code == 200:
            data = response.json()
            print(f"✅ History endpoint works: {len(data.get('history', []))} records")
            return True
        else:
            print(f"❌ History endpoint failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ History endpoint error: {e}")
        return False

def test_api_predict_endpoint():
    """Test the API predict endpoint with a sample file"""
    print("🧪 Testing API predict endpoint...")
    
    # Check if there are any existing audio files
    audio_files = []
    if os.path.exists('static/audio_uploads'):
        audio_files = [f for f in os.listdir('static/audio_uploads') if f.endswith(('.wav', '.mp3', '.m4a'))]
    
    if not audio_files:
        print("⚠️  No audio files found for testing")
        return False
    
    # Use the first available audio file
    test_file = os.path.join('static/audio_uploads', audio_files[0])
    
    try:
        with open(test_file, 'rb') as f:
            files = {'file': f}
            response = requests.post(f"{BASE_URL}/api/predict", files=files)
            
        if response.status_code == 200:
            data = response.json()
            print(f"✅ API predict works: {data.get('emotion', 'Unknown')} emotion detected")
            print(f"   Confidence: {data.get('confidence', 0):.2f}")
            return True
        else:
            print(f"❌ API predict failed: {response.status_code}")
            print(f"   Response: {response.text}")
            return False
    except Exception as e:
        print(f"❌ API predict error: {e}")
        return False

def test_main_predict_endpoint():
    """Test the main predict endpoint"""
    print("🧪 Testing main predict endpoint...")
    
    # Check if there are any existing audio files
    audio_files = []
    if os.path.exists('static/audio_uploads'):
        audio_files = [f for f in os.listdir('static/audio_uploads') if f.endswith(('.wav', '.mp3', '.m4a'))]
    
    if not audio_files:
        print("⚠️  No audio files found for testing")
        return False
    
    # Use the first available audio file
    test_file = os.path.join('static/audio_uploads', audio_files[0])
    
    try:
        with open(test_file, 'rb') as f:
            files = {'file': f}
            response = requests.post(f"{BASE_URL}/predict", files=files)
            
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Main predict works: {data.get('emotion', 'Unknown')} emotion detected")
            print(f"   Visualization: {data.get('visualization', 'None')}")
            print(f"   Suggestions: {len(data.get('suggestions', {}))} items")
            return True
        else:
            print(f"❌ Main predict failed: {response.status_code}")
            print(f"   Response: {response.text}")
            return False
    except Exception as e:
        print(f"❌ Main predict error: {e}")
        return False

def test_pdf_download():
    """Test PDF report download"""
    print("🧪 Testing PDF download...")
    
    # Check if there are any existing audio files
    audio_files = []
    if os.path.exists('static/audio_uploads'):
        audio_files = [f for f in os.listdir('static/audio_uploads') if f.endswith(('.wav', '.mp3', '.m4a'))]
    
    if not audio_files:
        print("⚠️  No audio files found for testing")
        return False
    
    # Use the first available audio file
    test_file = audio_files[0]
    
    try:
        response = requests.get(f"{BASE_URL}/download-report/{test_file}")
        
        if response.status_code == 200:
            print(f"✅ PDF download works: {len(response.content)} bytes")
            return True
        else:
            print(f"❌ PDF download failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ PDF download error: {e}")
        return False

def test_error_handling():
    """Test error handling with invalid requests"""
    print("🧪 Testing error handling...")
    
    # Test with no file
    try:
        response = requests.post(f"{BASE_URL}/predict")
        if response.status_code == 400:
            print("✅ No file error handled correctly")
        else:
            print(f"❌ No file error not handled: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ No file error test failed: {e}")
        return False
    
    # Test with invalid file type
    try:
        files = {'file': ('test.txt', b'This is not an audio file', 'text/plain')}
        response = requests.post(f"{BASE_URL}/predict", files=files)
        if response.status_code == 400:
            print("✅ Invalid file type error handled correctly")
        else:
            print(f"❌ Invalid file type error not handled: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Invalid file type error test failed: {e}")
        return False
    
    return True

def main():
    """Run all tests"""
    print("🚀 Starting Advanced Audio Emotion Detection API Tests")
    print("=" * 60)
    
    tests = [
        test_home_page,
        test_history_endpoint,
        test_api_predict_endpoint,
        test_main_predict_endpoint,
        test_pdf_download,
        test_error_handling
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"❌ Test {test.__name__} crashed: {e}")
        print()
    
    print("=" * 60)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The application is working correctly.")
    else:
        print("⚠️  Some tests failed. Please check the application.")
    
    return passed == total

if __name__ == "__main__":
    # Wait a moment for the server to start
    print("⏳ Waiting for server to start...")
    time.sleep(3)
    
    success = main()
    exit(0 if success else 1) 