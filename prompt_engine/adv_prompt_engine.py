import sys
from pathlib import Path

# Ensure the project root is on sys.path so the top-level "models" package can be imported
project_root = Path('..').resolve()
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Also add the models/trained directory to sys.path to be safe
trained_model_dir = (project_root / 'models' / 'trained').resolve()
if str(trained_model_dir) not in sys.path:
    sys.path.insert(0, str(trained_model_dir))


from models.trained.inference_utils import predict_intent_with_confidence, load_intent_classifier  # NOQA: E402

# Advanced features for prompt engineering
class AdvancedPromptFeatures:
    """Additional prompt engineering capabilities"""
    
    @staticmethod
    def create_conversation_flow(prompt_engine, conversation_history):
        """Create multi-turn conversation awareness"""
        context_summary = ""
        if conversation_history:
            context_summary = f"\nConversation Context: Previous messages indicate ongoing {conversation_history[-1]['intent']} issue."
        
        return context_summary
    
    @staticmethod
    def generate_follow_up_questions(intent, customer_message):
        """Generate intent-specific follow-up questions"""
        follow_ups = {
            'Technical_Issues': [
                "What device and operating system are you using?",
                "When did this issue first start occurring?",
                "Are you seeing any specific error messages?"
            ],
            'Account_Issues': [
                "What email address is associated with your account?",
                "When did you last successfully access your account?",
                "Are you trying to log in from a new device?"
            ],
            'Billing_Problems': [
                "Which charges are you questioning?",
                "What date did the incorrect charge appear?",
                "Would you like me to walk through your recent billing history?"
            ]
        }
        
        return follow_ups.get(intent, ["Is there anything else I can help clarify?"])
    
    @staticmethod
    def assess_escalation_criteria(intent_result, customer_message):
        """Determine if human escalation is needed"""
        escalation_triggers = {
            'low_confidence': intent_result.get('confidence', 1.0) < 0.6,
            'complex_language': len(customer_message.split()) > 30,
            'negative_sentiment': any(word in customer_message.lower() 
                                   for word in ['lawsuit', 'attorney', 'fraud', 'stolen']),
            'multiple_issues': customer_message.lower().count(' and ') > 2
        }
        
        return escalation_triggers
    

if __name__ == "__main__":

    model = load_intent_classifier('../models/trained/best_intent_classifier.pkl')


    # Test advanced features
    customer_msg = "The app crashed and I lost my data, plus I was charged for premium features I didn't want"
    intent_result = predict_intent_with_confidence(customer_msg, model)

    escalation_check = AdvancedPromptFeatures.assess_escalation_criteria(intent_result, customer_msg)
    follow_ups = AdvancedPromptFeatures.generate_follow_up_questions(intent_result['predicted_intent'], customer_msg)

    print("🔍 ADVANCED PROMPT ANALYSIS")
    print(f"Message: {customer_msg}")
    print(f"Intent: {intent_result['predicted_intent']}")
    print(f"Escalation Triggers: {escalation_check}")
    print(f"Follow-up Questions: {follow_ups}")