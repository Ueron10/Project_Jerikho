# News Classification System - NLP

Implementasi Natural Language Processing pada Klasifikasi Artikel Berita Bahasa Inggris

## 📋 Overview

This project implements a Natural Language Processing (NLP) system for classifying English news articles into four categories: World, Sports, Business, and Sci/Tech. The system uses a complete NLP pipeline including preprocessing, feature extraction, and machine learning classification.

## 🎯 Categories

- **World**: International news and global events
- **Sports**: Sports news and athletic events
- **Business**: Business, finance, and economic news
- **Sci/Tech**: Science and technology news

## 🏗️ Project Structure

```
DL/
├── app/
│   ├── __init__.py
│   ├── app.py                      # Flask web application
│   ├── templates/
│   │   └── index.html              # Web interface
│   └── static/
│       ├── css/
│       │   └── style.css           # Styling
│       └── js/
│           └── script.js           # Frontend logic
├── data/
│   ├── raw/                        # Raw dataset files
│   └── processed/                  # Processed data
├── models/                         # Saved models
│   ├── news_classifier.pkl         # Trained classifier
│   └── tfidf_vectorizer.pkl        # TF-IDF vectorizer
├── src/
│   ├── __init__.py
│   ├── data_loader.py              # Dataset management
│   ├── preprocessing/
│   │   ├── __init__.py
│   │   └── text_preprocessor.py    # Text preprocessing
│   ├── features/
│   │   ├── __init__.py
│   │   └── tfidf_extractor.py      # TF-IDF feature extraction
│   └── models/
│       ├── __init__.py
│       └── classifier.py           # ML classifier
├── Docs/                           # Documentation
├── requirements.txt                 # Python dependencies
├── train_model.py                  # Training script
└── README.md                       # This file
```

## 🔧 NLP Pipeline

The system implements the following NLP pipeline:

1. **Case Folding**: Convert all text to lowercase
2. **Special Character Removal**: Remove numbers and special characters
3. **Tokenization**: Split text into individual words
4. **Stopword Removal**: Remove common words (the, a, an, etc.)
5. **Stemming**: Reduce words to their root form using Porter Stemmer
6. **TF-IDF**: Convert text to numerical features using Term Frequency-Inverse Document Frequency
7. **Classification**: Use machine learning to predict category

## 📦 Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. The system will automatically download required NLTK data on first run.

## 🚀 Usage

### Option 1: Train and Run via Web Interface

1. **Train the model**:
```bash
python train_model.py
```

This will:
- Create a sample dataset (or load existing one)
- Preprocess the text
- Train the classifier
- Save the model and vectorizer
- Display evaluation results

2. **Run the web application**:
```bash
python app/app.py
```

3. Open your browser and navigate to:
```
http://localhost:5000
```

4. Use the web interface to:
   - Train the model (click "Train Model" button)
   - Load an existing model (click "Load Model" button)
   - Classify news articles by entering title and description

### Option 2: Use in Python Code

```python
from src.preprocessing.text_preprocessor import TextPreprocessor
from src.features.tfidf_extractor import TfidfExtractor
from src.models.classifier import NewsClassifier
from src.data_loader import DataLoader

# Initialize components
preprocessor = TextPreprocessor()
tfidf_extractor = TfidfExtractor()
classifier = NewsClassifier(model_type='naive_bayes')
data_loader = DataLoader()

# Load dataset
df = data_loader.create_sample_dataset()
df = data_loader.combine_title_description(df)

# Preprocess
processed_texts = preprocessor.preprocess_batch(df['Combined_Text'].tolist())

# Train
tfidf_extractor.fit(processed_texts)
X_tfidf = tfidf_extractor.transform(processed_texts)
classifier.train(X_tfidf, df['Class Id'].tolist())

# Predict new text
new_text = "New breakthrough in quantum computing research"
processed = preprocessor.preprocess(new_text)
features = tfidf_extractor.transform([processed])
prediction = classifier.predict(features)[0]
category = classifier.get_category_name(prediction)
print(f"Category: {category}")
```

## 🤖 Available Models

The system supports multiple classification algorithms:

- **Naive Bayes** (default): Fast and effective for text classification
- **Logistic Regression**: Interpretable and performs well on text data
- **SVM (Linear)**: Good for high-dimensional text features
- **Random Forest**: Ensemble method for robust classification

To change the model, modify the `model_type` parameter:
```python
classifier = NewsClassifier(model_type='logistic_regression')
```

## 📊 Dataset

The system uses a news dataset with the following structure:

| Column | Description |
|--------|-------------|
| Class Id | Category identifier (0-3) |
| Title | News article title |
| Description | News article content |

### Sample Dataset

A sample dataset with 20 articles (5 per category) is included for testing. To use your own dataset:

1. Place your CSV file in `data/raw/`
2. Ensure it has columns: Class Id, Title, Description
3. Use `data_loader.load_csv('your_file.csv')`

## 📈 Model Evaluation

The system provides comprehensive evaluation metrics:

- **Accuracy**: Overall classification accuracy
- **Precision**: True positive rate for each category
- **Recall**: Coverage of each category
- **F1-Score**: Harmonic mean of precision and recall
- **Confusion Matrix**: Detailed prediction breakdown

## 🔍 Preprocessing Details

### Text Preprocessor Methods

- `case_folding()`: Convert to lowercase
- `remove_special_characters()`: Remove non-alphabetic characters
- `tokenization()`: Split into words using NLTK
- `remove_stopwords()`: Filter out common English stopwords
- `stemming()`: Apply Porter stemming algorithm
- `preprocess()`: Complete pipeline in one call

### TF-IDF Configuration

- **Max Features**: 5000 (most frequent words)
- **N-gram Range**: (1, 2) (unigrams and bigrams)
- **Stop Words**: English stopwords removed automatically

## 🌐 Web Application Features

- **Model Training**: Train the model directly from the web interface
- **Model Loading**: Load pre-trained models
- **Real-time Classification**: Classify news articles instantly
- **Probability Display**: See confidence scores for each category
- **Responsive Design**: Works on desktop and mobile devices

## 📝 API Endpoints

### POST /train
Train the model with sample data.

**Response:**
```json
{
  "success": true,
  "accuracy": 0.95,
  "classification_report": "...",
  "message": "Model trained successfully"
}
```

### POST /load_model
Load a pre-trained model from disk.

### POST /predict
Classify a news article.

**Request:**
```json
{
  "title": "News title",
  "description": "News description"
}
```

**Response:**
```json
{
  "category": "Sports",
  "probabilities": {
    "World": 0.1,
    "Sports": 0.8,
    "Business": 0.05,
    "Sci/Tech": 0.05
  }
}
```

## 🛠️ Troubleshooting

### NLTK Data Download Error

If you encounter NLTK download errors, run:
```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
```

### Model Not Loading

Ensure the model files exist in the `models/` directory:
- `news_classifier.pkl`
- `tfidf_vectorizer.pkl`

If missing, run `python train_model.py` to create them.

### Port Already in Use

If port 5000 is in use, modify the port in `app/app.py`:
```python
app.run(debug=True, host='0.0.0.0', port=5001)
```

## 📚 References

This project implements standard NLP techniques for text classification:

- **NLTK**: Natural Language Toolkit for text processing
- **Scikit-learn**: Machine learning library for classification
- **TF-IDF**: Term Frequency-Inverse Document Frequency for feature extraction
- **Porter Stemmer**: Algorithm for reducing words to root form

## 👨‍💻 Author

Jeriko Ictus Seo

## 📄 License

This project is for educational purposes.

## 🤝 Contributing

To extend this project:

1. Add new preprocessing steps in `src/preprocessing/text_preprocessor.py`
2. Implement new feature extraction methods in `src/features/`
3. Add new classification algorithms in `src/models/classifier.py`
4. Enhance the web interface in `app/templates/` and `app/static/`

## 📞 Support

For issues or questions, please refer to the code documentation or create an issue in the project repository.
