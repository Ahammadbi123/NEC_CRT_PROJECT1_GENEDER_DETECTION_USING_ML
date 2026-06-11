from flask import Flask, render_template, request
import pandas as pd
import joblib
import os

app = Flask(__name__)

# --- STEP 1: LOAD DATABASE & AI MODEL ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 1. Load CSV for 100% Accuracy on known names
try:
    df_db = pd.read_csv('dataset.csv')
    # Dictionary lookup is much faster than searching dataframe every time
    DB_LOOKUP = dict(zip(df_db['name'].str.lower(), df_db['gender']))
except Exception as e:
    print(f"Error loading dataset.csv: {e}")
    DB_LOOKUP = {}

# 2. Load Trained AI Model and Vectorizer
try:
    ai_model = joblib.load(os.path.join(BASE_DIR, 'model', 'expert_model.pkl'))
    vectorizer = joblib.load(os.path.join(BASE_DIR, 'model', 'vectorizer.pkl'))
except Exception as e:
    print(f"Error loading ML models. Did you run train_ai.py first? Error: {e}")

# --- STEP 2: FEATURE EXTRACTION FOR ML ---
def get_features(name):
    """
    Transforms a name into numeric patterns that the ML model can understand.
    This must match the training script exactly.
    """
    name = name.lower().strip()
    return {
        'last_1': name[-1], 
        'last_2': name[-2:] if len(name) > 1 else name[-1], 
        'last_3': name[-3:] if len(name) > 2 else name[-1], 
        'last_4': name[-4:] if len(name) > 3 else name[-1],
        'first_2': name[:2], 
        'first_3': name[:3], 
        'vowels': sum(1 for c in name if c in 'aeiou'),
        'length': len(name)
    }

# --- STEP 3: HYBRID PREDICTION LOGIC ---
def get_prediction(name):
    name = name.lower().strip()
    
    if not name:
        return "Unknown", "No Input"

    # LAYER 1: DATABASE CHECK (Priority 1 - 100% Correct)
    if name in DB_LOOKUP:
        return DB_LOOKUP[name].upper(), "Verified Dataset"

    # LAYER 2: LINGUISTIC SHIELD (Priority 2 - Indian Suffix Logic)
    m_end = ('ish', 'sh', 'esh', 'kumar', 'rao', 'reddy', 'naidu', 'singh', 'an', 'ul', 'ar', 'at', 'ak', 'ay', 'aj', 'am', 'as')
    f_end = ('itha', 'ana', 'ya', 'ni', 'ti', 'thi', 'shree', 'sri', 'ini', 'ika', 'vathi', 'deepika', 'lavanya')
    
    # Exception handling for Male names that end in vowels (e.g., Sai, Shiva)
    male_exceptions = ['sai', 'nani', 'vamsi', 'shiva', 'krishna', 'rama', 'surya', 'chaitanya', 'rishi', 'mani']
    if any(name == ex for ex in male_exceptions):
        return "MALE", "Heuristic Logic"

    if name.endswith(m_end): 
        return "MALE", "Heuristic Logic"
    
    if name.endswith(f_end) or name.endswith(('a', 'i', 'e')): 
        return "FEMALE", "Heuristic Logic"

    # LAYER 3: ML FALLBACK (Priority 3 - For unseen/unique names)
    try:
        feat = vectorizer.transform([get_features(name)])
        pred = ai_model.predict(feat)[0]
        return pred.upper(), "Neural Analysis (AI)"
    except Exception as e:
        return "MALE", "Default (ML Error)"

# --- STEP 4: FLASK ROUTES ---

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        name_input = request.form['name'].strip()
        
        if len(name_input) < 2:
            return render_template('index.html', prediction="ERROR", method="Invalid Name", name=name_input)
            
        result, method = get_prediction(name_input)
        
        return render_template('index.html', 
                               prediction=result, 
                               method=method, 
                               name=name_input.title())

if __name__ == '__main__':
    # Using debug=True helps in development to see errors instantly
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
