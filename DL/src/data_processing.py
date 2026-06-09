import pandas as pd
import os
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib

# Download required NLTK data
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('punkt_tab', quiet=True)


class DataLoader:
    def __init__(self, data_path='data'):
        self.data_path = data_path
        self.category_mapping = {
            'World': 1,
            'Sports': 2,
            'Business': 3,
            'Sci/Tech': 4
        }
    
    def load_csv(self, filename):
        """Load data from CSV file"""
        filepath = os.path.join(self.data_path, filename)
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File not found: {filepath}")
        
        df = pd.read_csv(filepath)
        return df
    
    def combine_title_description(self, df):
        """Combine title and description into a single text field"""
        df['Combined_Text'] = df['Title'] + ' ' + df['Description']
        return df


class TextPreprocessor:
    def __init__(self):
        self.stemmer = PorterStemmer()
        self.stop_words = set(stopwords.words('english'))
    
    def case_folding(self, text):
        """Convert text to lowercase"""
        return text.lower()
    
    def remove_special_characters(self, text):
        """Remove special characters and numbers"""
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        return text
    
    def tokenization(self, text):
        """Tokenize text into words"""
        tokens = word_tokenize(text)
        return tokens
    
    def remove_stopwords(self, tokens):
        """Remove stopwords from tokens"""
        filtered_tokens = [token for token in tokens if token not in self.stop_words]
        return filtered_tokens
    
    def stemming(self, tokens):
        """Apply stemming to tokens"""
        stemmed_tokens = [self.stemmer.stem(token) for token in tokens]
        return stemmed_tokens
    
    def preprocess(self, text):
        """Complete preprocessing pipeline"""
        # Case folding
        text = self.case_folding(text)
        
        # Remove special characters
        text = self.remove_special_characters(text)
        
        # Tokenization
        tokens = self.tokenization(text)
        
        # Remove stopwords
        tokens = self.remove_stopwords(tokens)
        
        # Stemming
        tokens = self.stemming(tokens)
        
        # Join tokens back to text
        processed_text = ' '.join(tokens)
        
        return processed_text
    
    def preprocess_batch(self, texts):
        """Preprocess a list of texts"""
        processed_texts = [self.preprocess(text) for text in texts]
        return processed_texts


def split_data(X, y, test_size=0.2, random_state=42, stratify=None):
    """
    Split data into training and testing sets
    
    Args:
        X: Features (list of texts or TF-IDF features)
        y: Labels (list of class IDs)
        test_size: Proportion of data for testing (default: 0.2)
        random_state: Random seed for reproducibility (default: 42)
        stratify: Labels for stratified splitting (default: None)
    
    Returns:
        X_train, X_test, y_train, y_test: Split data
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify
    )
    
    print(f"Data split completed:")
    print(f"  - Training samples: {len(X_train)}")
    print(f"  - Testing samples: {len(X_test)}")
    
    return X_train, X_test, y_train, y_test


class TfidfExtractor:
    def __init__(self, max_features=5000, ngram_range=(1, 2)):
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            stop_words='english'
        )
        self.is_fitted = False
    
    def fit(self, texts):
        """Fit the TF-IDF vectorizer on the training data"""
        self.vectorizer.fit(texts)
        self.is_fitted = True
        return self
    
    def transform(self, texts):
        """Transform texts to TF-IDF features"""
        if not self.is_fitted:
            raise ValueError("Vectorizer must be fitted before transform")
        return self.vectorizer.transform(texts)
    
    def fit_transform(self, texts):
        """Fit and transform in one step"""
        result = self.vectorizer.fit_transform(texts)
        self.is_fitted = True
        return result
    
    def get_feature_names(self):
        """Get feature names"""
        return self.vectorizer.get_feature_names_out()
    
    def save(self, filepath):
        """Save the fitted vectorizer"""
        if not self.is_fitted:
            raise ValueError("Cannot save unfitted vectorizer")
        joblib.dump(self.vectorizer, filepath)
    
    def load(self, filepath):
        """Load a fitted vectorizer"""
        self.vectorizer = joblib.load(filepath)
        self.is_fitted = True
        return self


if __name__ == "__main__":
    print("=" * 50)
    print("Data Processing Pipeline")
    print("=" * 50)
    
    # Initialize components
    loader = DataLoader(data_path='../data')
    preprocessor = TextPreprocessor()
    tfidf_extractor = TfidfExtractor(max_features=5000, ngram_range=(1, 2))
    
    # Load dataset
    print("\n1. Loading dataset...")
    df = loader.load_csv('dataset.csv')
    print(f"   Loaded {len(df)} samples from dataset.csv")
    
    # Combine title and description
    print("\n2. Combining title and description...")
    df = loader.combine_title_description(df)
    
    # Preprocess text
    print("\n3. Preprocessing text...")
    processed_texts = preprocessor.preprocess_batch(df['Combined_Text'].tolist())
    df['Processed_Text'] = processed_texts
    print(f"   Preprocessed {len(processed_texts)} texts")
    
    # Split data
    print("\n4. Splitting data...")
    X_train, X_test, y_train, y_test = split_data(
        processed_texts,
        df['Class Index'].tolist(),
        test_size=0.2,
        random_state=42,
        stratify=df['Class Index'].tolist()
    )
    
    # Extract TF-IDF features
    print("\n5. Extracting TF-IDF features...")
    X_train_tfidf = tfidf_extractor.fit_transform(X_train)
    X_test_tfidf = tfidf_extractor.transform(X_test)
    print(f"   TF-IDF features shape: {X_train_tfidf.shape}")
    
    # Save vectorizer
    print("\n6. Saving TF-IDF vectorizer...")
    os.makedirs('../models', exist_ok=True)
    tfidf_extractor.save('../models/tfidf_vectorizer.pkl')
    print("   Vectorizer saved to models/tfidf_vectorizer.pkl")
    
    # Save processed data
    print("\n7. Saving processed data...")
    joblib.dump({
        'X_train': X_train_tfidf,
        'X_test': X_test_tfidf,
        'y_train': y_train,
        'y_test': y_test
    }, '../models/processed_data.pkl')
    print("   Processed data saved to models/processed_data.pkl")
    
    print("\n" + "=" * 50)
    print("Data processing completed successfully!")
    print("=" * 50)
