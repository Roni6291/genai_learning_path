# SMS Spam Classifier - ANN vs Naive Bayes

## Project Overview

A comprehensive SMS spam classification system comparing Artificial Neural Networks (TensorFlow/Keras) with Naive Bayes baseline. Includes complete CLI interface, data preprocessing pipeline, model training/evaluation, and error analysis.

**Key Results:**
- **ANN**: 97.49% accuracy, 89.97% F1-score, 87.25% recall
- **Naive Bayes**: 97.31% accuracy, 100% precision, 79.87% recall

---

## Features

✅ **Modular CLI Architecture** - Command groups for data, model, and comparison pipelines
✅ **Data Processing** - Text cleaning, TF-IDF vectorization, stratified splitting
✅ **Neural Network** - 2-layer ANN with dropout regularization
✅ **Baseline Comparison** - Multinomial Naive Bayes for benchmarking
✅ **Comprehensive Evaluation** - Accuracy, Precision, Recall, F1-score, Confusion Matrix
✅ **Error Analysis** - Detailed misclassification investigation with explanations
✅ **Visualization** - Training curves, confusion matrices, performance plots

---

## Installation

### Prerequisites
- Python 3.13+
- [uv](https://github.com/astral-sh/uv) package manager (recommended)

### Setup

```bash
# Clone repository
git clone https://github.com/Roni6291/ANN_SMS_spam_classifier.git
cd ANN_SMS_spam_classifier

# Install dependencies
uv sync

# Or with pip
pip install -e .
```

### Dependencies

- **tensorflow** 2.20.0 - Neural network framework
- **scikit-learn** 1.7.2 - Naive Bayes, metrics, train/test split
- **matplotlib** 3.10.0 - Visualization
- **seaborn** 0.13.2 - Confusion matrix heatmaps
- **click** - CLI framework
- **loguru** - Structured logging
- **nltk** - Text preprocessing
- **numpy** - Numerical operations

---

## Quick Start

### Complete Workflow

```bash
# 1. Clean raw data
uv run spam-classifier data clean

# 2. Preprocess and vectorize
uv run spam-classifier data preprocess

# 3. Train neural network
uv run spam-classifier model train --epochs 20

# 4. Evaluate on test set
uv run spam-classifier model evaluate --model-path artifacts/spam_classifier_model.keras

# 5. Analyze errors
uv run spam-classifier model analyze-errors -m artifacts/spam_classifier_model.keras -c data/clean/sms_spam_cleaned.txt

# 6. Compare with Naive Bayes
uv run spam-classifier compare naive-bayes
```

---

## CLI Commands

### Data Pipeline

#### Clean Data
```bash
uv run spam-classifier data clean [OPTIONS]

Options:
  -i, --input-path PATH    Raw data file (default: data/raw/SMSSpamCollection)
  -o, --output-path PATH   Cleaned output file (default: data/clean/sms_spam_cleaned.txt)
```

**What it does:**
- Converts text to lowercase
- Removes URLs, digits, punctuation, special characters
- Tokenizes and removes stopwords
- Lemmatizes words to base forms

#### Inspect Data
```bash
uv run spam-classifier data inspect [OPTIONS]

Options:
  -i, --input-path PATH    Data file to inspect (default: data/raw/SMSSpamCollection)
```

**What it does:**
- Shows dataset statistics (total messages, ham/spam counts)
- Displays class distribution
- Shows sample messages

#### Preprocess Data
```bash
uv run spam-classifier data preprocess [OPTIONS]

Options:
  -i, --input-path PATH    Cleaned data file (default: data/clean/sms_spam_cleaned.txt)
  -o, --output-dir PATH    Output directory (default: data/)
```

**What it does:**
- TF-IDF vectorization (5,000 features, unigrams + bigrams)
- Train/test split (80/20 with stratification)
- Saves vectorizer, label encoder, and numpy arrays

---

### Model Pipeline

#### Train Neural Network
```bash
uv run spam-classifier model train [OPTIONS]

Options:
  --train-dir PATH         Training data directory (default: data/train)
  --output-dir PATH        Model save directory (default: artifacts)
  --epochs INTEGER         Training epochs (default: 10)
  --batch-size INTEGER     Batch size (default: 32)
  --validation-split FLOAT Validation split ratio (default: 0.2)
  --hidden-layers TEXT     Hidden layer sizes (default: "128,64")
  --dropout FLOAT          Dropout rate (default: 0.5)
```

**Model Architecture:**
```
Input (5000) → Dense(128, ReLU) → Dropout(0.5)
            → Dense(64, ReLU) → Dropout(0.5)
            → Dense(1, Sigmoid)
```

**Training Details:**
- Optimizer: Adam
- Loss: Binary crossentropy
- Metrics: Accuracy, Precision, Recall
- Total parameters: 648,449

#### Evaluate Model
```bash
uv run spam-classifier model evaluate [OPTIONS]

Options:
  -m, --model-path PATH    Trained model file (required)
  --test-dir PATH          Test data directory (default: data/test)
  --output-dir PATH        Results save directory (default: artifacts)
  --threshold FLOAT        Classification threshold (default: 0.5)
```

**Outputs:**
- Accuracy, Precision, Recall, F1-score
- Confusion matrix visualization (PNG)
- Detailed metrics logged to `logs/evaluation.log`

#### Analyze Errors
```bash
uv run spam-classifier model analyze-errors [OPTIONS]

Options:
  -m, --model-path PATH       Trained model file (required)
  -t, --test-dir PATH         Test data directory (default: data/test)
  -c, --cleaned-data PATH     Cleaned messages file (required)
  -o, --output-dir PATH       Results directory (default: artifacts)
  --threshold FLOAT           Classification threshold (default: 0.5)
  -n, --num-examples INTEGER  Examples per category (default: 5)
```

**Outputs:**
- False positive examples with confidence scores
- False negative examples with confidence scores
- Detailed analysis report: `artifacts/misclassification_analysis.txt`

---

### Comparison Pipeline

#### Naive Bayes Baseline
```bash
uv run spam-classifier compare naive-bayes [OPTIONS]

Options:
  --train-dir PATH     Training data directory (default: data/train)
  --test-dir PATH      Test data directory (default: data/test)
  --output-dir PATH    Results directory (default: artifacts)
  --alpha FLOAT        Laplace smoothing parameter (default: 1.0)
```

**Outputs:**
- Complete evaluation metrics
- Confusion matrix: `artifacts/naive_bayes_confusion_matrix.png`
- Saved model: `artifacts/naive_bayes_model.pkl`
- Logs: `logs/naive-bayes.log`

---

## Project Structure

```
1.1-ANN_spam_classifier/
├── data/
│   ├── raw/
│   │   └── SMSSpamCollection          # Original dataset
│   ├── clean/
│   │   └── sms_spam_cleaned.txt       # Cleaned text
│   ├── train/
│   │   ├── X_train.npy                # Training features (TF-IDF)
│   │   └── y_train.npy                # Training labels
│   └── test/
│       ├── X_test.npy                 # Test features (TF-IDF)
│       └── y_test.npy                 # Test labels
├── spam_classifier/
│   ├── cli/
│   │   ├── __init__.py                # Main CLI entry point
│   │   ├── data.py                    # Data pipeline commands
│   │   ├── model.py                   # Model commands
│   │   └── compare.py                 # Comparison commands
│   ├── data_processor/
│   │   ├── clean.py                   # TextCleaner class
│   │   └── processor.py               # Data processing utilities
│   ├── pipelines/
│   │   ├── data/
│   │   │   ├── cleaning.py            # Data cleaning pipeline
│   │   │   ├── inspection.py          # Data inspection
│   │   │   └── preprocessing.py       # TF-IDF vectorization
│   │   ├── train/
│   │   │   └── training.py            # ANN training pipeline
│   │   ├── evaluation/
│   │   │   ├── eval.py                # Model evaluation
│   │   │   └── analyze_errors.py      # Error analysis
│   │   └── compare/
│   │       └── naive_bayes_classification.py
│   ├── classifier.py                  # SMSSpamClassifier class
│   └── __main__.py                    # Module entry point
├── artifacts/
│   ├── spam_classifier_model.keras    # Trained ANN model
│   ├── naive_bayes_model.pkl          # Trained Naive Bayes
│   ├── vectorizer.pkl                 # TF-IDF vectorizer
│   ├── label_encoder.pkl              # Label encoder
│   ├── confusion_matrix.png           # ANN confusion matrix
│   ├── naive_bayes_confusion_matrix.png
│   ├── training_history.png           # Training curves
│   └── misclassification_analysis.txt
├── logs/
│   ├── training.log                   # Training logs
│   ├── evaluation.log                 # Evaluation logs
│   └── naive-bayes.log                # Naive Bayes logs
├── reports/
│   └── final_report.md                # Comprehensive analysis report
├── pyproject.toml                     # Project configuration
├── ruff.toml                          # Linting configuration
└── README.md
```

---

## Model Performance

### Results Summary

| Metric | ANN | Naive Bayes | Winner |
|--------|-----|-------------|--------|
| **Accuracy** | 97.49% | 97.31% | ANN (+0.18%) |
| **Precision** | 92.86% | **100.00%** | Naive Bayes |
| **Recall** | **87.25%** | 79.87% | ANN (+7.38%) |
| **F1-Score** | **89.97%** | 88.81% | ANN (+1.16%) |

### Confusion Matrices

#### ANN
|  | Predicted Ham | Predicted Spam |
|---|---|---|
| **Actual Ham** | 955 | 10 |
| **Actual Spam** | 19 | 130 |

**Error Rate**: 2.60% (29 errors / 1,114 samples)

#### Naive Bayes
|  | Predicted Ham | Predicted Spam |
|---|---|---|
| **Actual Ham** | 965 | 0 |
| **Actual Spam** | 30 | 119 |

**Error Rate**: 2.69% (30 errors / 1,114 samples)

### Key Insights

1. **Precision**: Naive Bayes achieves perfect 100% precision (zero false positives) but sacrifices recall
2. **Recall**: ANN catches 7.38% more spam messages through non-linear pattern learning
3. **F1-Score**: ANN provides better balance between precision and recall
4. **Trade-offs**: Choose Naive Bayes for zero false alarms, ANN for better spam detection

---

## Example Usage

### Python API

```python
from spam_classifier.classifier import SMSSpamClassifier
import numpy as np

# Load test data
X_test = np.load('data/test/X_test.npy')
y_test = np.load('data/test/y_test.npy')

# Initialize and load model
classifier = SMSSpamClassifier(input_dim=5000)
classifier.load('artifacts/spam_classifier_model.keras')

# Make predictions
predictions = classifier.predict(X_test, threshold=0.5)
probabilities = classifier.predict_proba(X_test)

# Evaluate
metrics = classifier.evaluate(X_test, y_test)
print(f"Test Accuracy: {metrics['accuracy']:.4f}")
print(f"Test Loss: {metrics['loss']:.4f}")

# Plot training history (after training)
classifier.plot_training_history()
```

### Custom Training

```python
from spam_classifier.classifier import SMSSpamClassifier
import numpy as np

# Load training data
X_train = np.load('data/train/X_train.npy')
y_train = np.load('data/train/y_train.npy')

# Initialize with custom architecture
classifier = SMSSpamClassifier(
    input_dim=5000,
    hidden_layers=[256, 128, 64],  # 3 hidden layers
    dropout_rate=0.3
)

# Train
history = classifier.train(
    X_train, y_train,
    epochs=30,
    batch_size=64,
    validation_split=0.2
)

# Save model
classifier.save('artifacts/custom_model.keras')
```

---

## Data Processing Details

### Text Cleaning Pipeline

The `TextCleaner` class performs the following operations:

1. **Lowercase conversion** - Normalize case
2. **URL removal** - Remove HTTP/HTTPS links
3. **Digit removal** - Remove numbers
4. **Punctuation removal** - Remove punctuation marks
5. **Special character removal** - Keep only letters and spaces
6. **Tokenization** - Split into words
7. **Stopword removal** - Remove common English words
8. **Lemmatization** - Reduce words to base forms

**Example:**
```python
from spam_classifier.data_processor.clean import clean_text

text = "WINNER!! Call 09061701461. Visit http://example.com NOW!"
cleaned = clean_text(text)
print(cleaned)
# Output: "winner call visit example"
```

### TF-IDF Vectorization

- **Max features**: 5,000
- **N-gram range**: (1, 2) - unigrams and bigrams
- **Min document frequency**: 1
- **Sublinear TF**: True (log scaling)

---

## Troubleshooting

### TensorFlow Warnings

If you see oneDNN or CPU optimization warnings, they're suppressed in the code but you can also set:

```bash
set TF_CPP_MIN_LOG_LEVEL=2
set TF_ENABLE_ONEDNN_OPTS=0
```

### NLTK Data

If stopwords or lemmatizer data is missing:

```python
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')
```

### Memory Issues

If training fails due to memory constraints:
- Reduce batch size: `--batch-size 16`
- Reduce hidden layer sizes: `--hidden-layers "64,32"`
- Use smaller TF-IDF features (modify preprocessing.py)

---

## Development

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=spam_classifier --cov-report=html
```

### Linting

```bash
# Check with ruff
ruff check .

# Auto-fix
ruff check --fix .

# Format code
ruff format .
```

### Adding New Commands

1. Create command function in appropriate CLI module (`cli/data.py`, `cli/model.py`, etc.)
2. Register command with Click decorator
3. Add to command group in `cli/__init__.py`
4. Update `pyproject.toml` entry points

---

## References

- **Dataset**: [UCI SMS Spam Collection](https://archive.ics.uci.edu/ml/datasets/SMS+Spam+Collection)
- **TensorFlow/Keras**: https://www.tensorflow.org/
- **Scikit-learn**: https://scikit-learn.org/
- **Click**: https://click.palletsprojects.com/

---

## License

MIT License - see LICENSE file for details

---

## Contributors

- **Roni6291** - Initial implementation and development

---

## Citation

If you use this project in your research, please cite:

```bibtex
@software{sms_spam_classifier_2025,
  author = {Roni6291},
  title = {SMS Spam Classifier: ANN vs Naive Bayes},
  year = {2025},
  url = {https://github.com/Roni6291/ANN_SMS_spam_classifier}
}
```

---

**Last Updated**: November 24, 2025
