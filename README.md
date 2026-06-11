# NEC_CRT_PROJECT1_GENDER_DETECTION_USING_ML
A high-accuracy Gender Detection System using Flask and Machine Learning (Random Forest).
# Gender Detection System using Hybrid Machine Learning
**Project ID:** NEC_CRT_PROJECT1_GENDER_DETECTION_USING_ML

# Gender Detection Intelligence System
**Live Demo:** [Click Here to Access the Web App](https://nec-crt-project1-gender-detection-using.onrender.com)

##  Project Overview
This project is an advanced web-based intelligence system designed to identify gender (Male/Female) from human names. It employs a **Triple-Layer Hybrid Architecture** that combines traditional database lookup, linguistic heuristic rules, and a trained Machine Learning model to ensure nearly 88% accuracy.

---

##  System Architecture (How it Works)
To ensure the highest reliability, the system processes every name through three distinct layers:

1. **Layer 1: Verified Database Lookup**
   The system first checks the input against a pre-verified `dataset.csv` containing thousands of Indian and Global names. If a match is found, it returns the result with 100% certainty.

2. **Layer 2: Linguistic Heuristic Engine**
   If the name is not in the database, the engine analyzes phonetic suffixes. 
   - **Male Patterns:** Identifies endings like `-ish`, `-esh`, `-kumar`, `-singh`, `-sh`, etc.
   - **Female Patterns:** Identifies endings like `-itha`, `-ana`, `-ni`, `-vathi`, `-shree`, etc.
   - **Exception Handler:** Manages names like *Sai, Shiva, or Krishna* which end in vowels but are male.

3. **Layer 3: Random Forest ML Model**
   If the first two layers fail, the system utilizes a **Random Forest Classifier** trained on character n-grams. It analyzes the sound and structure of the name to predict the most likely gender.

---

##  Technology Stack
- **Language:** Python 3.x
- **Framework:** Flask (Web Micro-framework)
- **Machine Learning:** Scikit-Learn (Random Forest Algorithm)
- **Data Manipulation:** Pandas & NumPy
- **Feature Engineering:** DictVectorizer (Character Suffix Mapping)
- **UI/UX:** Modern Dark Dashboard with Glassmorphism (HTML5/CSS3)

---

##  Installation & Execution

### 1. Environment Setup
Install the required Python libraries:
```bash
pip install flask pandas scikit-learn joblib
2. Dataset Generation
Prepare the training data:
python dataset_generator.py
3. Training the Model
Train the Machine Learning model:
python train_model.py
4. Deploying the Application
Launch the Flask server:
python app.py
Access the dashboard at: http://127.0.0.1:5000

Project Structure
app.py: The central controller for the hybrid prediction logic.
train_ai.py: Script to extract features and train the Random Forest model.
dataset_generator.py: Generates the initial CSV name repository.
model/: Directory storing serialized ML models (.pkl files).
static/: Professional CSS styles for the dark-themed UI.
templates/: HTML structure for the identification dashboard.

Performance & Accuracy
By integrating Rule-based Logic with Machine Learning, the system effectively handles:
Known Names: 88% Accuracy
Unknown Names: ~75% ML Accuracy
Phonetic Patterns: Effectively managed via Suffix Analysis.


Developed by: SHAIK AHAMMAD BI
Project Category: Machine Learning / Web Intelligence

