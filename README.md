# Email Spam Classifier

A production-structured spam classification system using **TF-IDF + scikit-learn** (Naive Bayes, Logistic Regression, Random Forest), served via a **FastAPI** REST API.

---

## Project Structure

```
spam-classification/
├── api/
│   ├── main.py                        # FastAPI app entry point
│   ├── schemas.py                     # Pydantic request/response models
│   └── routers/
│       └── predict.py                 # POST /api/v1/predict endpoint
│
├── data/
│   ├── raw/                           # Original dataset (not tracked in git)
│   │   └── combined_data.csv
│   ├── processed/                     # Preprocessed outputs
│   └── external/                      # Third-party supplemental data
│
├── models/
│   ├── artifacts/                     # Serialized model + vectorizer (not tracked in git)
│   │   ├── model.joblib
│   │   └── vectorizer.joblib
│   └── registry/                      # Experiment tracking metadata
│
├── src/
│   └── spam_classifier/
│       ├── data/
│       │   ├── loader.py              # CSV loading + schema validation
│       │   └── preprocessor.py       # clean_text(), tokenize_with_spacy()
│       ├── features/
│       │   └── vectorizer.py         # TF-IDF fit/transform wrappers
│       ├── models/
│       │   ├── trainer.py            # Model selection + GridSearchCV tuning
│       │   ├── evaluator.py          # Metrics, classification report, confusion matrix
│       │   └── predictor.py          # predict_message(), predict_batch()
│       ├── pipelines/
│       │   ├── train_pipeline.py     # Full train flow orchestration
│       │   └── inference_pipeline.py # Load artifacts + predict
│       └── utils/
│           ├── logging.py            # Structured logger setup
│           └── io.py                 # joblib save/load helpers
│
├── notebooks/
│   └── code.ipynb                    # EDA and experimentation only
│
├── dataset/
│   └── read_dataset.py
│
├── requirements.txt
└── README.md
```

---

## Prerequisites

- Python 3.10+
- `combined_data.csv` placed at `data/raw/combined_data.csv`

---

## 1. Clone the Repository

```bash
git clone https://github.com/kousikgvs/email-spam-classifier.git
cd email-spam-classifier
```

---

## 2. Create and Activate Virtual Environment

**Windows**
```bash
python -m venv venv
venv\Scripts\activate
```

**macOS / Linux**
```bash
python -m venv venv
source venv/bin/activate
```

---

## 3. Install Dependencies

```bash
pip install -r requirements.txt
pip install spacy fastapi uvicorn httpx
```

Download the required NLTK data:
```bash
python -c "import nltk; nltk.download('wordnet'); nltk.download('omw-1.4')"
```

---

## 4. Prepare the Dataset

Place your dataset at:
```
data/raw/combined_data.csv
```

The CSV must have exactly two columns:

| Column | Type    | Description                  |
|--------|---------|------------------------------|
| `text`  | string  | Raw email / SMS message text |
| `label` | integer | `0` = Ham, `1` = Spam        |

---

## 5. Train the Model

Run the training pipeline from the project root:

```bash
PYTHONPATH=src python -c "
from spam_classifier.utils.logging import setup_logging
from spam_classifier.pipelines.train_pipeline import run_train_pipeline
setup_logging()
run_train_pipeline(
    data_path='data/raw/combined_data.csv',
    artifact_dir='models/artifacts',
)
"
```

**Windows (PowerShell)**
```powershell
$env:PYTHONPATH = "src"
python -c "
from spam_classifier.utils.logging import setup_logging
from spam_classifier.pipelines.train_pipeline import run_train_pipeline
setup_logging()
run_train_pipeline(
    data_path='data/raw/combined_data.csv',
    artifact_dir='models/artifacts',
)
"
```

What happens during training:
1. Loads and validates `combined_data.csv`
2. Cleans and tokenizes text using spaCy + NLTK lemmatization
3. Splits data (80% train / 20% test, stratified)
4. Fits a TF-IDF vectorizer (`max_features=30000`, unigrams + bigrams)
5. Compares Naive Bayes, Logistic Regression, and Random Forest
6. Tunes the best model with `GridSearchCV` (3-fold CV, F1 scoring)
7. Evaluates on the test set and logs a classification report
8. Saves `model.joblib` and `vectorizer.joblib` to `models/artifacts/`

**Example output:**
```
Model comparison:
              model       f1  accuracy
      random_forest 0.999673  0.999655
logistic_regression 0.988984  0.988346
        naive_bayes 0.976868  0.975868

Best params for random_forest: {'max_depth': None, 'min_samples_split': 2, 'n_estimators': 200}

Final Metrics
Accuracy : 0.9862
Precision: 0.9849
Recall   : 0.9890
F1       : 0.9869
```

---

## 6. Run Inference (Programmatic)

```bash
PYTHONPATH=src python -c "
from spam_classifier.utils.logging import setup_logging
from spam_classifier.pipelines.inference_pipeline import run_inference, run_batch_inference
setup_logging()

# Single message
result = run_inference('Win a free iPhone now!', 'models/artifacts')
print('Prediction:', result)

# Batch
messages = [
    'Win a free iPhone now!',
    'Can we reschedule our meeting to tomorrow?',
    'Limited offer: Buy medicine at 70% discount.',
]
results = run_batch_inference(messages, 'models/artifacts')
for msg, label in zip(messages, results):
    print(f'  [{label}] {msg}')
"
```

---

## 7. Start the API Server

```bash
ARTIFACT_DIR=models/artifacts PYTHONPATH=src uvicorn api.main:app --host 127.0.0.1 --port 8000
```

**Windows (PowerShell)**
```powershell
$env:PYTHONPATH = "src"
$env:ARTIFACT_DIR = "models/artifacts"
uvicorn api.main:app --host 127.0.0.1 --port 8000
```

Expected startup output:
```
INFO:     Started server process [xxxxx]
INFO:     Waiting for application startup.
INFO:     Inference artifacts loaded from models/artifacts
INFO:     Application startup complete.
INFO:     Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)
```

---

## 8. Call the API

### Using curl

```bash
curl -X POST http://127.0.0.1:8000/api/v1/predict \
     -H "Content-Type: application/json" \
     -d '{"message": "Win a free iPhone now! Click here."}'
```

### Using Python (httpx)

```python
import httpx

r = httpx.post(
    "http://127.0.0.1:8000/api/v1/predict",
    json={"message": "Win a free iPhone now! Click here."}
)
print(r.json())
```

### Response

```json
{
  "message": "Win a free iPhone now! Click here.",
  "prediction": "Spam"
}
```

### API Endpoints

| Method | Endpoint             | Description                  |
|--------|----------------------|------------------------------|
| `POST` | `/api/v1/predict`    | Classify a single message    |
| `GET`  | `/docs`              | Interactive Swagger UI       |
| `GET`  | `/redoc`             | ReDoc API documentation      |

---

## 9. Interactive API Docs

Open your browser and navigate to:

```
http://127.0.0.1:8000/docs
```

This opens the Swagger UI where you can test the `/api/v1/predict` endpoint directly from the browser.

---

## Environment Variables

| Variable       | Default             | Description                              |
|----------------|---------------------|------------------------------------------|
| `ARTIFACT_DIR` | `models/artifacts`  | Path to the saved model and vectorizer   |
| `PYTHONPATH`   | _(must be set)_     | Must include `src` for imports to resolve|

---

## Notes

- `models/artifacts/` and `data/raw/` are excluded from git (files exceed GitHub's 100MB limit). Re-run training after cloning.
- The `notebooks/code.ipynb` file is for EDA only — all production logic lives in `src/spam_classifier/`.
