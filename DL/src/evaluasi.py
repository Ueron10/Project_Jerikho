import sys
import os

from data_processing import DataLoader, TextPreprocessor, TfidfExtractor, split_data
from training import NewsClassifier


def main():
    print("=" * 60)
    print("News Classification Model Training")
    print("=" * 60)
    
    # Initialize components
    print("\n[1/6] Initializing components...")
    preprocessor = TextPreprocessor()
    tfidf_extractor = TfidfExtractor(max_features=5000, ngram_range=(1, 2))
    classifier = NewsClassifier(model_type='naive_bayes')
    data_loader = DataLoader(data_path='../data')
    
    # Load dataset
    print("\n[2/6] Loading dataset...")
    df = data_loader.load_csv('dataset.csv')
    print(f"Loaded {len(df)} samples from CSV file")
    
    # Combine title and description
    print("\n[3/6] Combining title and description...")
    df = data_loader.combine_title_description(df)
    print(f"Combined text length: {len(df['Combined_Text'].iloc[0])} characters")
    
    # Preprocess texts
    print("\n[4/6] Preprocessing texts...")
    print("   - Case folding")
    print("   - Removing special characters")
    print("   - Tokenization")
    print("   - Stopword removal")
    print("   - Stemming")
    processed_texts = preprocessor.preprocess_batch(df['Combined_Text'].tolist())
    print(f"Preprocessed {len(processed_texts)} texts")
    
    # Split data
    print("\n[5/6] Splitting data...")
    X_train, X_test, y_train, y_test = split_data(
        processed_texts,
        df['Class Index'].tolist(),
        test_size=0.2,
        random_state=42,
        stratify=df['Class Index'].tolist()
    )
    
    # Fit TF-IDF
    print("\n[6/6] Training model...")
    print("   - Fitting TF-IDF vectorizer...")
    tfidf_extractor.fit(X_train)
    
    print("   - Transforming training data...")
    X_train_tfidf = tfidf_extractor.transform(X_train)
    
    print("   - Transforming test data...")
    X_test_tfidf = tfidf_extractor.transform(X_test)
    
    print(f"   - Training {classifier.model_type} classifier...")
    classifier.train(X_train_tfidf, y_train)
    
    # Evaluate
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    results = classifier.evaluate(X_test_tfidf, y_test)
    
    print(f"\nAccuracy: {results['accuracy']:.4f} ({results['accuracy']*100:.2f}%)")
    print("\nClassification Report:")
    print(results['classification_report'])
    
    print("\nConfusion Matrix:")
    print(results['confusion_matrix'])
    
    # Save model and vectorizer
    print("\n" + "=" * 60)
    print("SAVING MODEL")
    print("=" * 60)
    os.makedirs('models', exist_ok=True)
    
    classifier.save('models/news_classifier.pkl')
    print("Classifier saved to models/news_classifier.pkl")
    
    tfidf_extractor.save('models/tfidf_vectorizer.pkl')
    print("TF-IDF vectorizer saved to models/tfidf_vectorizer.pkl")
    
    print("\n" + "=" * 60)
    print("Training completed successfully!")
    print("=" * 60)
    print("\nTo run the web application:")
    print("  python aplikasi.py")
    print("\nThen open http://localhost:5000 in your browser")


if __name__ == '__main__':
    main()
