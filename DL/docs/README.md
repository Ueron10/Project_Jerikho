# News Classification System - NLP

Implementasi Natural Language Processing pada Klasifikasi Artikel Berita Bahasa Inggris

## 📋 Overview

This project implements a Natural Language Processing (NLP) system for classifying English news articles into four categories: World, Sports, Business, and Sci/Tech. The system uses a complete NLP pipeline including preprocessing, feature extraction, and machine learning classification.

**Model Accuracy**: 88.55% with Naive Bayes classifier

## 🎯 Categories

- **World** (Class ID: 1): International news and global events
- **Sports** (Class ID: 2): Sports news and athletic events
- **Business** (Class ID: 3): Business, finance, and economic news
- **Sci/Tech** (Class ID: 4): Science and technology news

## 🏗️ Project Structure

```
DL/
├── app/
│   ├── templates/
│   │   └── index.html              # Web interface
│   └── static/
│       ├── css/
│   │   └── style.css               # Styling
│   └── js/
│       └── script.js               # Frontend logic
├── data/
│   └── dataset.csv                 # Dataset (7600 samples)
├── models/                         # Saved models
│   ├── naive_bayes_model.pkl       # Trained classifier
│   ├── tfidf_vectorizer.pkl        # TF-IDF vectorizer
│   └── processed_data.pkl          # Processed training data
├── src/
│   ├── __init__.py
│   ├── data_processing.py          # Data loading, preprocessing, TF-IDF
│   ├── training.py                 # Model training script
│   ├── evaluasi.py                 # Evaluation script
│   └── aplikasi.py                 # Flask web application
├── docs/
│   ├── README.md                   # This file
│   └── RUN_ORDER.md                # Execution order
└── requirements.txt                # Python dependencies
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

### Training the Model

1. **Process the dataset**:
```bash
cd src
python data_processing.py
```

This will:
- Load dataset.csv (7600 samples)
- Preprocess text (case folding, tokenization, stopword removal, stemming)
- Extract TF-IDF features
- Save processed data and vectorizer to models/

2. **Train the model**:
```bash
python training.py
```

This will:
- Load processed data
- Train Naive Bayes classifier
- Evaluate model (accuracy: 88.55%)
- Save trained model to models/

3. **Alternative: Run complete pipeline**:
```bash
python evaluasi.py
```

This runs the complete pipeline from data loading to model training and evaluation.

### Running the Web Application

1. **Start the Flask app**:
```bash
python aplikasi.py
```

2. Open your browser and navigate to:
```
http://localhost:5000
```

3. Use the web interface to:
   - Train the model (click "Train Model" button)
   - Load an existing model (click "Load Model" button)
   - Classify news articles by entering title and description

### Using in Python Code

```python
from data_processing import DataLoader, TextPreprocessor, TfidfExtractor
from training import NewsClassifier

# Initialize components
preprocessor = TextPreprocessor()
tfidf_extractor = TfidfExtractor()
classifier = NewsClassifier(model_type='naive_bayes')
data_loader = DataLoader(data_path='../data')

# Load dataset
df = data_loader.load_csv('dataset.csv')
df = data_loader.combine_title_description(df)

# Preprocess
processed_texts = preprocessor.preprocess_batch(df['Combined_Text'].tolist())

# Train
tfidf_extractor.fit(processed_texts)
X_tfidf = tfidf_extractor.transform(processed_texts)
classifier.train(X_tfidf, df['Class Index'].tolist())

# Predict new text
new_text = "New breakthrough in quantum computing research"
processed = preprocessor.preprocess(new_text)
features = tfidf_extractor.transform([processed])
prediction = classifier.predict(features)[0]
category = classifier.get_category_name(prediction)
print(f"Category: {category}")
```

## 🤖 Model

The system uses **Naive Bayes (MultinomialNB)** classifier:
- Fast and effective for text classification
- Works well with TF-IDF features
- Achieves 88.55% accuracy on the test set

## 📊 Dataset

The system uses a news dataset with the following structure:

| Column | Description |
|--------|-------------|
| Class Index | Category identifier (1-4) |
| Title | News article title |
| Description | News article content |

**Dataset Size**: 7600 news articles (1900 per category)

**Category Mapping**:
- 1: World
- 2: Sports
- 3: Business
- 4: Sci/Tech

To use your own dataset:
1. Place your CSV file in `data/`
2. Ensure it has columns: Class Index, Title, Description
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
Train the model with dataset.csv.

**Response:**
```json
{
  "success": true,
  "accuracy": 0.8855,
  "classification_report": "...",
  "message": "Model trained successfully"
}
```

### POST /load_model
Load a pre-trained model from disk.

**Response:**
```json
{
  "success": true,
  "message": "Model loaded successfully"
}
```

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
  },
  "processed_text": "sport champion final weekend..."
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
- `naive_bayes_model.pkl`
- `tfidf_vectorizer.pkl`

If missing, run:
```bash
cd src
python data_processing.py
python training.py
```

### Port Already in Use

If port 5000 is in use, modify the port in `src/aplikasi.py`:
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

1. Add new preprocessing steps in `src/data_processing.py` (TextPreprocessor class)
2. Implement new feature extraction methods in `src/data_processing.py` (TfidfExtractor class)
3. Add new classification algorithms in `src/training.py` (NewsClassifier class)
4. Enhance the web interface in `app/templates/` and `app/static/`

## 📞 Support

For issues or questions, please refer to the code documentation or create an issue in the project repository.
