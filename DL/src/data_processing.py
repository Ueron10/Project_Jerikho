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
            'World': 0,
            'Sports': 1,
            'Business': 2,
            'Sci/Tech': 3
        }
    
    def load_csv(self, filename):
        """Load data from CSV file"""
        filepath = os.path.join(self.data_path, filename)
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File not found: {filepath}")
        
        df = pd.read_csv(filepath)
        return df
    
    def create_sample_dataset(self, num_samples=100):
        """Create a sample dataset for testing"""
        sample_data = {
            'Class Id': [],
            'Title': [],
            'Description': []
        }
        
        # Sample news data for each category
        samples = {
            'World': [
                ("Global Summit Addresses Climate Change", "World leaders gathered today to discuss urgent measures to combat climate change and reduce carbon emissions."),
                ("Earthquake Strikes Pacific Region", "A powerful earthquake hit the Pacific region, causing widespread damage and prompting emergency response teams."),
                ("UN Peacekeeping Mission Extended", "The United Nations Security Council voted to extend the peacekeeping mission in the conflict zone for another six months."),
                ("New Trade Agreement Signed", "Multiple countries signed a comprehensive trade agreement aimed at boosting economic cooperation and reducing tariffs."),
                ("Diplomatic Talks Resume", "After months of tension, diplomatic talks between the two nations have resumed with hopes of resolving longstanding disputes.")
            ],
            'Sports': [
                ("Championship Finals This Weekend", "The highly anticipated championship finals will take place this weekend, with top teams competing for the title."),
                ("Star Player Signs Record Contract", "A renowned athlete has signed a record-breaking contract worth millions, marking a historic moment in sports."),
                ("Olympic Games Preparation Underway", "Athletes from around the world are intensifying their training as the Olympic Games approach."),
                ("Tournament Upset Shocks Fans", "In a stunning turn of events, the underdog team defeated the favorites in a thrilling tournament match."),
                ("New Stadium Construction Begins", "Construction has begun on a state-of-the-art stadium that will host major sporting events in the future.")
            ],
            'Business': [
                ("Stock Market Reaches New High", "The stock market surged to record levels as investor confidence grows amid positive economic indicators."),
                ("Tech Company Reports Strong Earnings", "A leading technology company announced better-than-expected quarterly earnings, driving stock prices up."),
                ("Merger Creates Industry Giant", "Two major corporations have merged to form the largest company in their sector, reshaping the industry landscape."),
                ("Interest Rates Remain Unchanged", "The central bank decided to keep interest rates steady, maintaining the current monetary policy."),
                ("Startup Raises Massive Funding Round", "A promising startup successfully raised significant funding from venture capitalists to expand operations.")
            ],
            'Sci/Tech': [
                ("Breakthrough in Quantum Computing", "Scientists have achieved a major breakthrough in quantum computing, solving complex problems faster than ever."),
                ("New Space Mission Launched", "A rocket carrying advanced satellites was successfully launched, marking a new era in space exploration."),
                ("AI System Achieves Human-Level Performance", "An artificial intelligence system has demonstrated performance comparable to human experts in various tasks."),
                ("Renewable Energy Innovation", "Researchers have developed a more efficient solar panel technology that could revolutionize clean energy production."),
                ("Medical Research Shows Promise", "Clinical trials for a new treatment show promising results, offering hope for patients with rare diseases.")
            ]
        }
        
        # Generate samples
        for category, news_list in samples.items():
            for title, description in news_list:
                sample_data['Class Id'].append(self.category_mapping[category])
                sample_data['Title'].append(title)
                sample_data['Description'].append(description)
        
        df = pd.DataFrame(sample_data)
        return df
    
    def save_dataset(self, df, filename):
        """Save dataset to CSV"""
        filepath = os.path.join(self.data_path, filename)
        os.makedirs(self.data_path, exist_ok=True)
        df.to_csv(filepath, index=False)
        print(f"Dataset saved to: {filepath}")
    
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
        return self.vectorizer.fit_transform(texts)
    
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
