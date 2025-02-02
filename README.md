# Amdocs-GenAI-Graduate-HACKATHON-2024

This repository contains a prototype for a Misinformation Detection and Fact-Checking system built for the Amdocs GenAI Graduate Hackathon 2024.

The project comprises two main components:
- **Model Training and Preparation** ([`train_model.py`](train_model.py))
- **Flask Web Application for Inference** ([`app.py`](app.py))

## Table of Contents
1. [Project Overview](#project-overview)
2. [File Structure](#file-structure)
3. [Setup and Installation](#setup-and-installation)
4. [Usage](#usage)
5. [Acknowledgements](#acknowledgements)

## Project Overview
This prototype leverages a machine learning pipeline to detect misinformation in text statements. The workflow consists of:
- **Steps 1-2:** Data Preparation and Model Training using TF-IDF vectorization, SMOTE for balancing, and an ensemble classifier.
- **Steps 3-4:** A Flask web application that allows users to input text and receive a misinformation analysis in real time.

## File Structure
```
Amdocs-GenAI-Graduate-HACKATHON-2024/
├── 

README.md

            # Project overview and instructions
├── 

app.py

         # Flask application for text inference (Steps 3-4)
├── 

train_model.py

               # Script for training the ML model (Steps 1-2)
├── 

politifact_factcheck_data.json

                           # Training data file (JSON format)
├── 

model.pkl

                      # Serialized trained model (generated after training)
├── 

vectorizer.pkl

                 # Serialized TF-IDF vectorizer (generated after training)
├── 

requirements.txt

                           # Python dependencies
```

## Setup and Installation

1. **Clone the repository:**
   ```sh
   git clone https://github.com/TheProfessor123/Amdocs-GenAI-Graduate-HACKATHON-2024.git
   cd Amdocs-GenAI-Graduate-HACKATHON-2024
   ```

2. **Create a virtual environment (optional but recommended):**
   ```sh
   python -m venv env
   # On Windows:
   env\Scripts\activate
   # On macOS/Linux:
   source env/bin/activate
   ```

3. **Install the required dependencies:**
   ```sh
   pip install -r requirements.txt
   ```

## Usage

### Model Training
Train the ML model by running:
```sh
python train_model.py
```
This script will:
- Load the training data from `politifact_factcheck_data.json`
- Preprocess the text and apply SMOTE for balancing
- Train an ensemble classifier using GridSearchCV for parameter tuning
- Save the trained model and the TF-IDF vectorizer as `model.pkl` and `vectorizer.pkl`

### Running the Flask Application
Once the model is trained, start the web application by running:
```sh
python app.py
```
Your Flask app will run in debug mode on port 5000 and listen on all interfaces (`0.0.0.0`). Open your browser and navigate to:
```
http://localhost:5000
```
to access the application.

## Acknowledgements
- Developed for the Amdocs GenAI Graduate Hackathon 2024.
- Special thanks to the open-source community for the tools and libraries that made this project possible.