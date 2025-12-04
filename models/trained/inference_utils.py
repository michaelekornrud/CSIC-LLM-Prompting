
"""
Inference utilities for customer service intent classification
Includes robust error handling for different model types (LinearSVM, LogisticRegression, etc.)
"""

import joblib
import pandas as pd
import numpy as np


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
        except Exception:
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
        except (AttributeError, Exception):
            pass
    
    # Final fallback - no confidence scores available
    if hasattr(model, 'classes_'):
        classes = model.classes_
    else:
        try:
            classes = model.named_steps['classifier'].classes_
        except Exception:
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
        print("\n📊 Model Capabilities:")
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
        
        print("\n🧪 Testing inference pipeline:")
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
                print("   Top predictions:")
                for intent, prob in sorted_probs[:3]:
                    if prob is not None:
                        print(f"     - {intent}: {prob:.3f}")
            else:
                print("   Confidence: Not available for this model type")
            
            if 'note' in result:
                print(f"   Note: {result['note']}")
        
        # Test batch prediction
        print("\n🔄 Testing batch prediction...")
        batch_results = predict_intent_batch(sample_tweets, model)
        print(batch_results[['predicted_intent', 'confidence']].head())
        
        # Show intent explanations
        print("\n📖 Intent Categories:")
        explanations = get_intent_explanations()
        for intent, info in explanations.items():
            print(f"   {intent}: {info['description']}")
        
    except FileNotFoundError:
        print("❌ Model file not found. Please train the model first.")
        print("   Run: python train_model.py")
    except Exception as e:
        print(f"❌ Error during inference: {e}")
        print("   Make sure the model was trained and saved correctly.")

