---
title: Drug Classification System
emoji: 🏥
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
pinned: false
license: mit
---

# 🏥 Drug Classification System

An intelligent drug recommendation system that predicts the most suitable drug for patients based on their medical characteristics using machine learning.

## 🚀 Features

- **Real-time Predictions**: Get instant drug recommendations based on patient data
- **Interactive Interface**: User-friendly Gradio web interface
- **High Accuracy**: 97% accuracy with Random Forest algorithm
- **Multiple Drug Classes**: Supports drugA, drugB, drugC, drugX, and drugY classifications

## 📊 Model Performance

- **Accuracy**: 97%
- **F1 Score**: 0.94
- **Algorithm**: Random Forest (100 estimators)
- **Features**: Age, Gender, Blood Pressure, Cholesterol, Na_to_K Ratio

## 🔬 Usage

Enter patient information:
- **Age**: 15-74 years
- **Gender**: Male (M) or Female (F)
- **Blood Pressure**: HIGH, LOW, or NORMAL
- **Cholesterol**: HIGH or NORMAL
- **Na_to_K Ratio**: Sodium to Potassium ratio (6.2-38.2)

The system will provide drug recommendations with confidence scores.

## ⚠️ Disclaimer

This application is for educational and demonstration purposes only. Always consult qualified healthcare professionals for medical decisions and treatment recommendations.

## 🛠️ Technical Details

- **Framework**: Gradio + Scikit-learn
- **Model**: Random Forest Classifier with preprocessing pipeline
- **Deployment**: Automated via GitHub Actions to Hugging Face Spaces
- **Data Processing**: Categorical encoding and numerical scaling

## 📈 Development

The model is automatically retrained and deployed via GitHub Actions when new data or code changes are pushed to the main branch.

Built with ❤️ for healthcare innovation