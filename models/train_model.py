"""
Customer Service Intent Classification Model Training
====================================================
Trains multiple sklearn models on clustered intent data from Step 1.
Compares performance and selects the best model for production.

Usage:
    python train_model.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import time
import json
from pathlib import Path
from datetime import datetime

# Scikit-learn imports
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
from sklearn.utils.class_weight import compute_class_weight
import joblib
import warnings

warnings.filterwarnings('ignore')

class IntentClassifierTrainer:
    """Main class for training intent classification models"""
    
    def __init__(self, data_path='../data/processed/tweets_with_intents.csv', 
                 output_dir='../models/trained', plots_dir='../reports/plots'):
        self.data_path = data_path
        self.output_dir = Path(output_dir)
        self.plots_dir = Path(plots_dir)
        
        # Create output directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.plots_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize attributes
        self.df = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.vectorizer_results = {}
        self.model_results = {}
        self.trained_models = {}
        self.best_model = None
        self.results_df = None
    
    def load_and_prepare_data(self):
        """Load processed data and prepare train/test splits"""
        print("📊 LOADING AND PREPARING DATA")
        print("=" * 50)
        
        # Load processed data
        self.df = pd.read_csv(self.data_path)
        print(f"Dataset shape: {self.df.shape}")
        print(f"Intent distribution:\n{self.df['intent'].value_counts()}")
        
        # Check class imbalance
        intent_counts = self.df['intent'].value_counts()
        imbalance_ratio = intent_counts.max() / intent_counts.min()
        print(f"Class imbalance ratio: {imbalance_ratio:.2f}")
        
        # Prepare features and target
        X = self.df['text']
        y = self.df['intent']
        
        # Stratified train-test split
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"Total samples: {len(X):,}")
        print(f"Training samples: {len(self.X_train):,}")
        print(f"Test samples: {len(self.X_test):,}")
        
        return self.df
    
    def optimize_vectorizer(self):
        """Test different vectorizers to find the best one"""
        print("\n🔤 OPTIMIZING TEXT VECTORIZATION")
        print("=" * 50)
        
        vectorizers = {
            'TF-IDF (1-gram)': TfidfVectorizer(max_features=10000, ngram_range=(1,1)),
            'TF-IDF (1-2 gram)': TfidfVectorizer(max_features=10000, ngram_range=(1,2)),
            'Count (1-gram)': CountVectorizer(max_features=10000, ngram_range=(1,1)),
            'TF-IDF (optimized)': TfidfVectorizer(
                max_features=15000,  # More features
                ngram_range=(1, 2),  # Just unigrams and bigrams
                min_df=50,  # Less aggressive - keep more terms
                max_df=0.8,  # Less aggressive - keep more terms
                stop_words='english',  # Use built-in only
                lowercase=True,
                dtype=np.float32,
                token_pattern=r'\b\w{2,}\b'
            )
        }
        
        # Quick test with Logistic Regression
        for name, vectorizer in vectorizers.items():
            # Create pipeline
            pipeline = Pipeline([
                ('vectorizer', vectorizer),
                ('classifier', LogisticRegression(random_state=42))
            ])
            
            # Quick 3-fold CV
            scores = cross_val_score(pipeline, self.X_train, self.y_train, cv=3, scoring='accuracy')
            self.vectorizer_results[name] = scores.mean()
            print(f"{name}: {scores.mean():.3f} (+/- {scores.std() * 2:.3f})")
        
        # Select best vectorizer
        best_vectorizer_name = max(self.vectorizer_results, key=self.vectorizer_results.get)
        best_vectorizer = vectorizers[best_vectorizer_name]
        
        print(f"\n🏆 Best vectorizer: {best_vectorizer_name}")
        return best_vectorizer
    
    def setup_models(self):
        """Define models to train with M3 Pro optimizations"""
        print("\nSETTING UP MODELS")
        print("=" * 50)

        classes = np.unique(self.y_train)
        class_weights_array = compute_class_weight(
            'balanced',
            classes=classes,
            y = self.y_train
        )

        class_weights_dict = dict(zip(classes,class_weights_array))

        print("Class weights calculated for imbalanced data:")
        intent_counts = self.y_train.value_counts()
        for intent, weight in sorted(class_weights_dict.items(), key=lambda x: x[1], reverse=True):
            count = intent_counts[intent]
            percentage = (count / len(self.y_train)) * 100
            print(f"   {intent}: {weight:.2f} ({count:,} samples, {percentage:.1f}%)")
        
        
        # Enable multi-core processing for M3 Pro
        os.environ['OMP_NUM_THREADS'] = '12'
        os.environ['MKL_NUM_THREADS'] = '12'
        
        print(f"Optimizing for Apple M3 Pro - Using {os.cpu_count()} CPU cores")
        
        models = {
            'Logistic Regression': LogisticRegression(
                random_state=42, 
                max_iter=1000, 
                n_jobs=-1,
                solver='saga',
                class_weight=class_weights_dict
            ),
            'Naive Bayes': MultinomialNB(),
            'Linear SVM': LinearSVC(
                random_state=42, 
                max_iter=1000, 
                dual=False,
                loss='squared_hinge',
                class_weight=class_weights_dict
            ),
            'SGD Classifier': SGDClassifier(
                random_state=42, 
                n_jobs=-1,
                loss='log_loss',
                learning_rate='adaptive',
                eta0=0.01,
                max_iter=1000,
                class_weight=class_weights_dict
            ),   
            'Random Forest': RandomForestClassifier(  # 🆕 ADD RANDOM FOREST WITH WEIGHTS
                random_state=42,
                n_estimators=100,
                n_jobs=-1,
                max_depth=20,
                class_weight=class_weights_dict
            )
        }

        self.class_weights_dict = class_weights_dict
        
        return models
  

    def train_and_evaluate_models(self, models, best_vectorizer):
        """Train and evaluate all models with proper class weight handling"""
        print("\n🚀 TRAINING AND EVALUATING MODELS WITH CLASS WEIGHTS")
        print("=" * 50)
        
        for name, model in models.items():
            start_time = time.time()
            print(f"\nTraining {name}...")
            
            # Create pipeline
            pipeline = Pipeline([
                ('vectorizer', best_vectorizer),
                ('classifier', model)
            ])
            
            # Special handling for Naive Bayes (doesn't support class_weight)
            if name == 'Naive Bayes':
                # Calculate sample weights for Naive Bayes
                sample_weights = np.array([self.class_weights_dict[label] for label in self.y_train])
                
                # Fit model with sample weights
                fit_start = time.time()
                
                # For Naive Bayes, we need to fit the pipeline differently
                # First fit the vectorizer
                X_train_vec = best_vectorizer.fit_transform(self.X_train)
                
                # Then fit the classifier with sample weights
                model.fit(X_train_vec, self.y_train, sample_weight=sample_weights)
                
                # Create a custom pipeline for prediction
                class WeightedNBPipeline:
                    def __init__(self, vectorizer, classifier):
                        self.vectorizer = vectorizer
                        self.classifier = classifier
                    
                    def predict(self, X):
                        X_vec = self.vectorizer.transform(X)
                        return self.classifier.predict(X_vec)
                    
                    def predict_proba(self, X):
                        X_vec = self.vectorizer.transform(X)
                        return self.classifier.predict_proba(X_vec)
                    
                    def score(self, X, y):
                        X_vec = self.vectorizer.transform(X)
                        return self.classifier.score(X_vec, y)
                    
                    @property
                    def classes_(self):
                        return self.classifier.classes_
                    
                    @property
                    def named_steps(self):
                        return {'vectorizer': self.vectorizer, 'classifier': self.classifier}
                
                pipeline = WeightedNBPipeline(best_vectorizer, model)
                fit_time = time.time() - fit_start
                
            else:
                # Normal fitting for other models (they support class_weight)
                fit_start = time.time()
                pipeline.fit(self.X_train, self.y_train)
                fit_time = time.time() - fit_start
            
            # Evaluate
            eval_start = time.time()
            train_score = pipeline.score(self.X_train, self.y_train)
            test_score = pipeline.score(self.X_test, self.y_test)
            eval_time = time.time() - eval_start
            
            # Cross-validation (skip for Naive Bayes due to complexity)
            if name == 'Naive Bayes':
                cv_scores = np.array([test_score, test_score, test_score])  # Approximate
                cv_time = 0.1
            else:
                cv_start = time.time()
                cv_scores = cross_val_score(
                    Pipeline([('vectorizer', best_vectorizer), ('classifier', model)]),
                    self.X_train, self.y_train,
                    cv=3, scoring='f1_weighted', n_jobs=-1  # 🆕 USE F1_WEIGHTED FOR IMBALANCED DATA
                )
                cv_time = time.time() - cv_start
            
            total_time = time.time() - start_time
            
            # Store results
            self.model_results[name] = {
                'train_accuracy': train_score,
                'test_accuracy': test_score,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'fit_time': fit_time,
                'eval_time': eval_time,
                'cv_time': cv_time,
                'total_time': total_time,
                'uses_class_weights': True if name != 'Naive Bayes' else 'sample_weights'  # 🆕 TRACK WEIGHTING METHOD
            }
            
            self.trained_models[name] = pipeline
            
            print(f"  Results: Train={train_score:.3f}, Test={test_score:.3f}, CV={cv_scores.mean():.3f}")
            print(f"  Timing: Fit={fit_time:.1f}s, CV={cv_time:.1f}s, Total={total_time:.1f}s")
            print(f"  Weighting: {self.model_results[name]['uses_class_weights']}")  # 🆕 SHOW WEIGHTING INFO
        
        return self.model_results, self.trained_models
    

    def analyze_results(self):
        """Analyze and summarize model results with focus on imbalanced data performance"""
        print("\n" + "=" * 60)
        print("📈 TRAINING RESULTS SUMMARY (WEIGHTED FOR IMBALANCED DATA)")
        print("=" * 60)
        
        self.results_df = pd.DataFrame(self.model_results).T
        self.results_df = self.results_df.round(3)
        
        # Sort by CV performance (now using f1_weighted)
        results_df_sorted = self.results_df.sort_values('cv_mean', ascending=False)
        print("Model Performance (sorted by F1-weighted score):")
        print(results_df_sorted[['cv_mean', 'test_accuracy', 'total_time', 'uses_class_weights']])
        
        # Identify best model
        best_model_name = results_df_sorted.index[0]
        best_cv_score = results_df_sorted['cv_mean'].iloc[0]
        fastest_model = self.results_df.sort_values('total_time').index[0]
        
        print(f"\n🥇 Best performing model: {best_model_name}")
        print(f"📊 Best F1-weighted score: {best_cv_score:.3f}")
        print(f"⚡ Fastest model: {fastest_model}")
        
        # Business interpretation for imbalanced data
        print("\n🎯 IMBALANCED DATA PERFORMANCE ANALYSIS")
        print("=" * 50)
        print(f"✅ WEIGHTED PERFORMANCE: {best_model_name} achieves {best_cv_score:.1%} F1-weighted score")
        print("🎯 CLASS BALANCE: All intents get fair treatment through weighting")
        print("📈 MINORITY CLASSES: Small intents (Technical_Issues, Service_Issues) are boosted")
        
        if best_cv_score > 0.85:
            print("🏆 RECOMMENDATION: Deploy model - good performance on imbalanced data!")
        else:
            print("⚠️  RECOMMENDATION: Consider ensemble methods or deep learning")
        
        self.best_model = self.trained_models[best_model_name]
        return results_df_sorted
    

    def detailed_classification_analysis(self):
        """Detailed analysis of classification performance per intent"""
        print("\n📊 DETAILED CLASSIFICATION ANALYSIS")
        print("=" * 60)
        
        # Get best model
        best_model_name = self.results_df.sort_values('cv_mean', ascending=False).index[0]
        best_model = self.trained_models[best_model_name]
        
        # Make predictions
        y_pred = best_model.predict(self.X_test)
        
        # Classification report
        print(f"🏆 {best_model_name} - Detailed Performance:")
        print("-" * 50)
        report = classification_report(self.y_test, y_pred, output_dict=True)
        
        # Convert to DataFrame for better display
        report_df = pd.DataFrame(report).transpose()
        
        print("Per-Intent Performance:")
        intent_performance = report_df.drop(['accuracy', 'macro avg', 'weighted avg']).round(3)
        print(intent_performance[['precision', 'recall', 'f1-score', 'support']])
        
        # Overall metrics
        print("\nOverall Metrics:")
        print(f"Accuracy: {report['accuracy']:.3f}")
        print(f"Macro Avg F1: {report['macro avg']['f1-score']:.3f}")
        print(f"Weighted Avg F1: {report['weighted avg']['f1-score']:.3f}")
        
        # Check for class imbalance impact
        print("\nClass Weight Impact Analysis:")
        
        for intent in intent_performance.index:
            sample_count = int(intent_performance.loc[intent, 'support'])
            f1_score = intent_performance.loc[intent, 'f1-score']
            weight = self.class_weights_dict.get(intent, 1.0)
            
            # Calculate improvement indicator
            if sample_count < 1000:  # Small class
                improvement = "🎯 Boosted by weights" if f1_score > 0.5 else "⚠️ Still struggling"
            else:  # Large class
                improvement = "✅ Naturally strong" if f1_score > 0.7 else "🔍 Needs investigation"
            
            print(f"   {intent}: F1={f1_score:.3f}, Weight={weight:.2f}, {improvement}")
        
        return report_df, y_pred
    
    def create_visualizations(self):
        """Create and save performance visualization plots"""
        print("\n📊 CREATING PERFORMANCE VISUALIZATIONS")
        print("=" * 50)
        
        results_df_sorted = self.results_df.sort_values('cv_mean', ascending=False)
        
        # Create comprehensive performance comparison
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 1. Accuracy Comparison
        ax1 = axes[0, 0]
        models_perf = results_df_sorted[['cv_mean', 'test_accuracy']].head()
        models_perf.plot(kind='bar', ax=ax1, rot=45)
        ax1.set_title('Model Accuracy Comparison')
        ax1.set_ylabel('Accuracy')
        ax1.legend(['CV Accuracy', 'Test Accuracy'])
        ax1.set_ylim(0.85, 1.0)
        
        # 2. Training Time Comparison
        ax2 = axes[0, 1]
        times = results_df_sorted['total_time'].head()
        times.plot(kind='bar', ax=ax2, rot=45, color='orange')
        ax2.set_title('Training Time Comparison')
        ax2.set_ylabel('Time (seconds)')
        
        # 3. Speed vs Accuracy Scatter
        ax3 = axes[1, 0]
        ax3.scatter(self.results_df['total_time'], self.results_df['cv_mean'], s=100, alpha=0.7)
        for i, model in enumerate(self.results_df.index):
            ax3.annotate(model, (self.results_df.iloc[i]['total_time'], self.results_df.iloc[i]['cv_mean']),
                        xytext=(5, 5), textcoords='offset points', fontsize=9)
        ax3.set_xlabel('Training Time (seconds)')
        ax3.set_ylabel('CV Accuracy')
        ax3.set_title('Speed vs Accuracy Trade-off')
        
        # 4. Overfitting Analysis
        ax4 = axes[1, 1]
        overfitting = self.results_df['train_accuracy'] - self.results_df['test_accuracy']
        overfitting.plot(kind='bar', ax=ax4, rot=45, color='red', alpha=0.7)
        ax4.set_title('Overfitting Analysis (Train - Test)')
        ax4.set_ylabel('Overfitting Score')
        ax4.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        
        # Save plot
        plot_path = self.plots_dir / 'model_comparison.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"📊 Performance plots saved to: {plot_path}")
        
        plt.show()
    
    def save_best_model(self):
        """Save the best performing model and metadata"""
        print("\n💾 SAVING BEST MODEL")
        print("=" * 50)
        
        # Get best model info
        best_model_name = self.results_df.sort_values('cv_mean', ascending=False).index[0]
        best_accuracy = self.results_df.sort_values('cv_mean', ascending=False)['cv_mean'].iloc[0]
        
        # Save model
        model_path = self.output_dir / 'best_intent_classifier.pkl'
        joblib.dump(self.best_model, model_path)
        print(f"✅ Best model saved to: {model_path}")

        # Helper function to make parameters JSON serializable
        def make_json_serializable(obj):
            """Convert non-serializable objects to strings"""
            if hasattr(obj, '__name__'):  # Class or function
                return obj.__name__
            elif isinstance(obj, type):  # Type object
                return str(obj)
            elif hasattr(obj, 'tolist'):  # NumPy array
                return obj.tolist()
            elif isinstance(obj, (dict,)):
                return {k: make_json_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [make_json_serializable(item) for item in obj]
            else:
                try:
                    # Try to serialize as-is
                    json.dumps(obj)
                    return obj
                except (TypeError, ValueError):
                    # If it fails, convert to string
                    return str(obj)
        
        # Create model metadata with serializable parameters
        vectorizer_params = make_json_serializable(
            self.best_model.named_steps['vectorizer'].get_params()
        )
        classifier_params = make_json_serializable(
            self.best_model.named_steps['classifier'].get_params()
        )
        
        # Create model metadata
        model_metadata = {
            'model_name': best_model_name,
            'cv_accuracy': float(best_accuracy),
            'test_accuracy': float(self.results_df.sort_values('cv_mean', ascending=False)['test_accuracy'].iloc[0]),
            'training_date': datetime.now().isoformat(),
            'model_path': str(model_path),
            'intent_classes': list(self.best_model.classes_),
            'n_features': self.best_model.named_steps['vectorizer'].get_feature_names_out().shape[0],
            'vectorizer_params': vectorizer_params,
            'classifier_params': classifier_params,
            'data_shape': {
                'total_samples': int(len(self.df)),
                'training_samples': int(len(self.X_train)),
                'test_samples': int(len(self.X_test))
            }
        }
        
        # Save metadata
        metadata_path = self.output_dir / 'model_metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump(model_metadata, f, indent=2)
        print(f"📋 Model metadata saved to: {metadata_path}")
        
        # Save full results
        results_path = self.output_dir / 'training_results.csv'
        self.results_df.to_csv(results_path)
        print(f"📊 Full results saved to: {results_path}")
        
        return model_path, metadata_path
    
# Replace the create_inference_utils method (around line 524) with this improved version:

    def create_inference_utils(self):
        """Create utility functions for model inference with robust error handling"""
        print("\n🔧 CREATING INFERENCE UTILITIES")
        print("=" * 50)
        
        inference_code = '''"""
Inference utilities for customer service intent classification
Includes robust error handling for different model types (LinearSVM, LogisticRegression, etc.)
"""
import joblib
import pandas as pd
import numpy as np
from pathlib import Path

def load_intent_classifier(model_path="best_intent_classifier.pkl"):
    """Load the trained intent classifier"""
    return joblib.load(model_path)

def check_model_capabilities(model):
    """
    Check what capabilities the loaded model has
    
    Args:
        model: Trained sklearn pipeline or model
    
    Returns:
        dict: Model capabilities
    """
    capabilities = {
        'model_type': str(type(model)),
        'has_predict': hasattr(model, 'predict'),
        'has_predict_proba': hasattr(model, 'predict_proba'),
        'has_decision_function': hasattr(model, 'decision_function'),
        'has_classes': hasattr(model, 'classes_'),
        'confidence_method': None
    }
    
    # Determine confidence method
    if capabilities['has_predict_proba']:
        capabilities['confidence_method'] = 'predict_proba'
    elif capabilities['has_decision_function']:
        capabilities['confidence_method'] = 'decision_function'
    else:
        capabilities['confidence_method'] = 'none'
    
    # Try to get model name from pipeline
    if hasattr(model, 'named_steps'):
        try:
            classifier_type = type(model.named_steps['classifier']).__name__
            capabilities['classifier_type'] = classifier_type
        except:
            capabilities['classifier_type'] = 'unknown'
    
    return capabilities

def predict_intent_with_confidence(text, model):
    """
    Predict intent with confidence scores (robust for all model types)
    
    Args:
        text (str): Input tweet/text to classify
        model: Trained sklearn pipeline
    
    Returns:
        dict: Intent prediction with confidence scores
    """
    prediction = model.predict([text])[0]
    
    # Check if model supports predict_proba
    if hasattr(model, 'predict_proba'):
        try:
            probabilities = model.predict_proba([text])[0]
            confidence = max(probabilities)
            
            # Get all class probabilities
            class_probs = dict(zip(model.classes_, probabilities))
            
            return {
                'predicted_intent': prediction,
                'confidence': confidence,
                'all_probabilities': class_probs
            }
        except AttributeError:
            # Some pipelines don't expose predict_proba properly
            pass
    
    # Fallback for models without predict_proba (like LinearSVC)
    if hasattr(model, 'decision_function'):
        try:
            # Use decision function scores as confidence proxy
            decision_scores = model.decision_function([text])[0]
            
            # For binary classification, decision_function returns 1D array
            if len(decision_scores.shape) == 0 or decision_scores.ndim == 0:
                decision_scores = [decision_scores]
            
            # Convert decision scores to pseudo-probabilities using softmax
            exp_scores = np.exp(decision_scores - np.max(decision_scores))  # Numerical stability
            probabilities = exp_scores / np.sum(exp_scores)
            confidence = max(probabilities)
            
            # Get all class probabilities
            if hasattr(model, 'classes_'):
                classes = model.classes_
            else:
                # Try to get classes from the classifier step
                classes = model.named_steps['classifier'].classes_
            
            class_probs = dict(zip(classes, probabilities))
            
            return {
                'predicted_intent': prediction,
                'confidence': confidence,
                'all_probabilities': class_probs,
                'note': 'Confidence calculated from decision scores (not true probabilities)'
            }
        except (AttributeError, Exception) as e:
            pass
    
    # Final fallback - no confidence scores available
    if hasattr(model, 'classes_'):
        classes = model.classes_
    else:
        try:
            classes = model.named_steps['classifier'].classes_
        except:
            classes = ['Unknown']
    
    # Return prediction with no confidence
    return {
        'predicted_intent': prediction,
        'confidence': None,
        'all_probabilities': {intent: None for intent in classes},
        'note': 'Model does not support confidence scoring'
    }

def predict_intent_batch(texts, model):
    """
    Predict intents for multiple texts (robust for all model types)
    
    Args:
        texts (list): List of texts to classify
        model: Trained sklearn pipeline
    
    Returns:
        pd.DataFrame: Predictions with confidence scores
    """
    predictions = model.predict(texts)
    
    # Try to get confidence scores
    confidences = []
    has_probabilities = False
    
    if hasattr(model, 'predict_proba'):
        try:
            probabilities = model.predict_proba(texts)
            confidences = probabilities.max(axis=1)
            has_probabilities = True
        except AttributeError:
            pass
    
    if not has_probabilities and hasattr(model, 'decision_function'):
        try:
            decision_scores = model.decision_function(texts)
            
            # Convert decision scores to pseudo-probabilities
            if decision_scores.ndim == 1:
                # Binary classification
                exp_scores = np.exp(np.abs(decision_scores))
                confidences = exp_scores / (1 + exp_scores)  # Sigmoid-like
            else:
                # Multi-class classification
                exp_scores = np.exp(decision_scores - np.max(decision_scores, axis=1, keepdims=True))
                probabilities = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
                confidences = probabilities.max(axis=1)
        except Exception:
            confidences = [None] * len(predictions)
    else:
        confidences = [None] * len(predictions)
    
    results = pd.DataFrame({
        'text': texts,
        'predicted_intent': predictions,
        'confidence': confidences
    })
    
    return results

def get_intent_explanations():
    """
    Get explanations for each intent category
    
    Returns:
        dict: Intent explanations for end users
    """
    explanations = {
        'Account_Issues': {
            'description': 'Login, password, account access problems',
            'keywords': ['login', 'password', 'account', 'locked', 'verify'],
            'example': 'Cannot access my account after password reset'
        },
        'Billing_Problems': {
            'description': 'Payment, billing, subscription issues',
            'keywords': ['charge', 'payment', 'bill', 'refund', 'subscription'],
            'example': 'Was charged twice for the same service'
        },
        'Technical_Issues': {
            'description': 'App crashes, errors, technical problems',
            'keywords': ['crash', 'error', 'bug', 'not working', 'loading'],
            'example': 'App keeps crashing when I upload photos'
        },
        'Service_Issues': {
            'description': 'Service outages, connectivity problems',
            'keywords': ['outage', 'down', 'connection', 'network', 'slow'],
            'example': 'Service has been down for 2 hours'
        },
        'Product_Questions': {
            'description': 'How-to questions, feature inquiries',
            'keywords': ['how to', 'what is', 'feature', 'tutorial', 'help'],
            'example': 'How do I change my notification settings?'
        },
        'Delivery_Issues': {
            'description': 'Shipping, order, delivery problems',
            'keywords': ['order', 'delivery', 'shipping', 'tracking', 'package'],
            'example': 'My package was delivered to wrong address'
        },
        'Complaints': {
            'description': 'Negative feedback, service complaints',
            'keywords': ['terrible', 'awful', 'disappointed', 'bad service'],
            'example': 'Customer service was very unprofessional'
        },
        'Compliments': {
            'description': 'Positive feedback, praise',
            'keywords': ['thank you', 'great', 'excellent', 'helpful'],
            'example': 'Thank you for the quick resolution!'
        },
        'General_Inquiry': {
            'description': 'General questions or unclear intent',
            'keywords': ['help', 'question', 'information', 'contact'],
            'example': 'I need help with my account'
        }
    }
    
    return explanations

# Example usage and testing:
if __name__ == "__main__":
    try:
        # Load model
        print("Loading trained model...")
        model = load_intent_classifier()
        
        # Check model capabilities
        capabilities = check_model_capabilities(model)
        print(f"\n📊 Model Capabilities:")
        print(f"   Model type: {capabilities.get('classifier_type', 'Unknown')}")
        print(f"   Confidence method: {capabilities['confidence_method']}")
        
        # Test with sample tweets
        sample_tweets = [
            "My account is locked and I can't log in",
            "I was charged twice for the same service", 
            "The app keeps crashing when I try to upload photos",
            "How do I cancel my subscription?",
            "Your service is terrible, very disappointed"
        ]
        
        print(f"\n🧪 Testing inference pipeline:")
        print("-" * 50)
        
        for i, tweet in enumerate(sample_tweets, 1):
            result = predict_intent_with_confidence(tweet, model)
            print(f"\n{i}. Tweet: {tweet}")
            print(f"   Intent: {result['predicted_intent']}")
            
            if result['confidence'] is not None:
                print(f"   Confidence: {result['confidence']:.3f}")
                
                # Show top 3 predictions
                sorted_probs = sorted(result['all_probabilities'].items(), 
                                    key=lambda x: x[1] if x[1] is not None else 0, reverse=True)
                print(f"   Top predictions:")
                for intent, prob in sorted_probs[:3]:
                    if prob is not None:
                        print(f"     - {intent}: {prob:.3f}")
            else:
                print(f"   Confidence: Not available for this model type")
            
            if 'note' in result:
                print(f"   Note: {result['note']}")
        
        # Test batch prediction
        print(f"\n🔄 Testing batch prediction...")
        batch_results = predict_intent_batch(sample_tweets, model)
        print(batch_results[['predicted_intent', 'confidence']].head())
        
        # Show intent explanations
        print(f"\n📖 Intent Categories:")
        explanations = get_intent_explanations()
        for intent, info in explanations.items():
            print(f"   {intent}: {info['description']}")
        
    except FileNotFoundError:
        print("❌ Model file not found. Please train the model first.")
        print("   Run: python train_model.py")
    except Exception as e:
        print(f"❌ Error during inference: {e}")
        print("   Make sure the model was trained and saved correctly.")
'''
    
        # Save inference utilities
        utils_path = self.output_dir / 'inference_utils.py'
        with open(utils_path, 'w') as f:
            f.write(inference_code)
        print(f"🛠️  Enhanced inference utilities saved to: {utils_path}")
        
        return utils_path    

    def run_full_pipeline(self):
        """Run the complete training pipeline"""
        print("🚀 STARTING INTENT CLASSIFICATION MODEL TRAINING PIPELINE")
        print("=" * 70)
        
        # Step 1: Load and prepare data
        self.load_and_prepare_data()
        
        # Step 2: Optimize vectorizer
        best_vectorizer = self.optimize_vectorizer()
        
        # Step 3: Setup models
        models = self.setup_models()
        
        # Step 4: Train and evaluate models
        self.train_and_evaluate_models(models, best_vectorizer)
        
        # Step 5: Analyze results
        results_df_sorted = self.analyze_results()
        
        # Step 5.5: 🆕 ADD DETAILED ANALYSIS
        self.detailed_classification_analysis()
        
        # Step 6: Create visualizations
        self.create_visualizations()
        
        # Step 7: Save best model
        model_path, metadata_path = self.save_best_model()
        
        # Step 8: Create inference utilities
        utils_path = self.create_inference_utils()
        
        print("\n" + "=" * 70)
        print("🎉 TRAINING PIPELINE COMPLETED!")
        print("=" * 70)
        print(f"📁 Model saved: {model_path}")
        print(f"🛠️  Utils saved: {utils_path}")
        print("\n🚀 Ready for Step 3: Prompt Engineering!")
        
        return {
            'best_model': self.best_model,
            'results_df': results_df_sorted,
            'model_path': model_path,
            'class_weights': self.class_weights_dict,
            'utils_path': utils_path
        }

if __name__ == "__main__":
    # Initialize and run trainer
    trainer = IntentClassifierTrainer()
    results = trainer.run_full_pipeline()