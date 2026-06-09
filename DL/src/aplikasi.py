from flask import Flask, render_template, request, jsonify
import sys
import os

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_processing import TextPreprocessor, TfidfExtractor, DataLoader, split_data
from training import NewsClassifier

app = Flask(__name__, template_folder='../app/templates', static_folder='../app/static')

# Initialize components
preprocessor = TextPreprocessor()
tfidf_extractor = TfidfExtractor()
classifier = NewsClassifier(model_type='naive_bayes')

# Global variables to track model state
model_loaded = False


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        title = data.get('title', '')
        description = data.get('description', '')
        
        if not title and not description:
            return jsonify({'error': 'Please provide either title or description'}), 400
        
        # Combine title and description
        combined_text = f"{title} {description}"
        
        # Preprocess
        processed_text = preprocessor.preprocess(combined_text)
        
        # Check if model is loaded
        if not model_loaded:
            return jsonify({'error': 'Model not trained yet. Please train the model first.'}), 400
        
        # Transform using TF-IDF
        tfidf_features = tfidf_extractor.transform([processed_text])
        
        # Predict
        prediction = classifier.predict(tfidf_features)[0]
        category_name = classifier.get_category_name(prediction)
        
        # Get probabilities if available
        try:
            probabilities = classifier.predict_proba(tfidf_features)[0]
            prob_dict = {
                'World': float(probabilities[0]),
                'Sports': float(probabilities[1]),
                'Business': float(probabilities[2]),
                'Sci/Tech': float(probabilities[3])
            }
        except:
            prob_dict = None
        
        return jsonify({
            'category': category_name,
            'probabilities': prob_dict,
            'processed_text': processed_text
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/train', methods=['POST'])
def train():
    global model_loaded
    
    try:
        from data_processing import DataLoader, split_data
        
        # Load dataset
        data_loader = DataLoader(data_path='../data')
        df = data_loader.load_csv('dataset.csv')
        df = data_loader.combine_title_description(df)
        
        # Preprocess all texts
        processed_texts = preprocessor.preprocess_batch(df['Combined_Text'].tolist())
        
        # Split data
        X_train, X_test, y_train, y_test = split_data(
            processed_texts,
            df['Class Index'].tolist(),
            test_size=0.2,
            random_state=42
        )
        
        # Fit TF-IDF
        tfidf_extractor.fit(X_train)
        
        # Transform training and test data
        X_train_tfidf = tfidf_extractor.transform(X_train)
        X_test_tfidf = tfidf_extractor.transform(X_test)
        
        # Train classifier
        classifier.train(X_train_tfidf, y_train)
        
        # Evaluate
        results = classifier.evaluate(X_test_tfidf, y_test)
        
        # Save model and vectorizer
        os.makedirs('../models', exist_ok=True)
        classifier.save('../models/naive_bayes_model.pkl')
        tfidf_extractor.save('../models/tfidf_vectorizer.pkl')
        
        model_loaded = True
        
        return jsonify({
            'success': True,
            'accuracy': results['accuracy'],
            'classification_report': results['classification_report'],
            'message': 'Model trained successfully'
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/load_model', methods=['POST'])
def load_model():
    global model_loaded
    
    try:
        # Load model and vectorizer
        classifier.load('../models/naive_bayes_model.pkl')
        tfidf_extractor.load('../models/tfidf_vectorizer.pkl')
        
        model_loaded = True
        
        return jsonify({
            'success': True,
            'message': 'Model loaded successfully'
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
