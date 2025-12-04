import sys
import numpy as np
from pathlib import Path


# Ensure the project root is on sys.path so the top-level "models" package can be imported
project_root = Path('..').resolve()
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Also add the models/trained directory to sys.path to be safe
trained_model_dir = (project_root / 'models' / 'trained').resolve()
if str(trained_model_dir) not in sys.path:
    sys.path.insert(0, str(trained_model_dir))


from models.trained.inference_utils import predict_intent_with_confidence, load_intent_classifier # NOQA: E402


# Add this cell to your notebook - Prompt Engineering Framework

class CustomerServicePromptEngine:
    """
    Prompt engineering system for customer service intents
    Creates context-aware, intent-specific responses
    """
    
    def __init__(self, model):
        self.model = model
        self.intent_prompts = self._initialize_intent_prompts()
        self.conversation_context = []
        
    def _initialize_intent_prompts(self):
        """Define intent-specific prompt templates"""
        return {
            'Account_Issues': {
                'system_prompt': """You are a helpful customer service representative specializing in account issues.
Your role is to provide clear, step-by-step solutions for login, password, and account access problems.
Always be empathetic and offer multiple solution paths when possible.""",
                
                'response_template': """I understand you're having trouble with your account access. Let me help you resolve this quickly.

Based on your issue: "{customer_message}"

Here are the steps I recommend:

1. **Immediate Action**: {immediate_solution}
2. **If that doesn't work**: {backup_solution}
3. **Additional Help**: {additional_resources}

Is there any specific error message you're seeing? This will help me provide more targeted assistance.

**Security Note**: For your protection, I'll never ask for your password in this chat.""",
                
                'solutions': {
                    'default': {
                        'immediate_solution': 'Try resetting your password using the "Forgot Password" link on the login page',
                        'backup_solution': 'Clear your browser cache and cookies, then try logging in again',
                        'additional_resources': 'If issues persist, I can escalate this to our technical team for priority support'
                    },
                    'locked_account': {
                        'immediate_solution': 'I can unlock your account right now - it appears to be temporarily locked for security',
                        'backup_solution': 'Verify your identity through the email/SMS verification process',
                        'additional_resources': 'I\'ll also review your account for any unusual activity'
                    },
                    'password_reset': {
                        'immediate_solution': 'Check your email (including spam folder) for the password reset link',
                        'backup_solution': 'Try using the alternative recovery options (phone number or security questions)',
                        'additional_resources': 'I can manually trigger a new reset email if needed'
                    }
                }
            },
            
            'Billing_Problems': {
                'system_prompt': """You are a customer service representative specializing in billing and payment issues.
You have access to account information and can resolve payment disputes, explain charges, and process refunds.
Always be transparent about fees and timelines.""",
                
                'response_template': """I sincerely apologize for the billing issue you're experiencing. Let me investigate this immediately.

Regarding: "{customer_message}"

**What I'm doing right now:**
✓ {investigation_action}
✓ Reviewing your recent transactions
✓ Checking for any system errors

**Resolution Plan:**
- {primary_resolution}
- Timeline: {resolution_timeline}
- {refund_information}

**Your Reference Number:** CS-{reference_number}

I'll personally monitor this to ensure it's resolved properly. Is there anything else about your billing I can clarify?""",
                
                'solutions': {
                    'default': {
                        'investigation_action': 'Pulling up your complete billing history',
                        'primary_resolution': 'I will reverse any incorrect charges and provide a detailed explanation',
                        'resolution_timeline': '24-48 hours for full processing',
                        'refund_information': 'Any refunds will appear in 3-5 business days on your original payment method',
                    }
                }
            },
            
            'Technical_Issues': {
                'system_prompt': """You are a technical support specialist for customer service.
You excel at troubleshooting app crashes, errors, and technical problems.
Provide clear, step-by-step technical solutions that non-technical users can follow.""",
                
                'response_template': """I understand how frustrating technical issues can be. Let me help you get this working properly.

**Issue**: "{customer_message}"
**Device/Platform**: {device_info}

**Quick Troubleshooting Steps:**
1. {step_1}
2. {step_2}
3. {step_3}

**Advanced Solutions** (if needed):
- {advanced_solution}

**Prevention**: {prevention_tip}

If these steps don't resolve the issue, I can:
- Schedule a screen-sharing session for hands-on help
- Escalate to our development team if it's a known bug
- Provide a temporary workaround while we investigate

What device and operating system are you using? This helps me give more specific guidance.""",
                
                'solutions': {
                    'app_crash': {
                        'step_1': 'Force close the app completely and restart it',
                        'step_2': 'Check for app updates in your app store',
                        'step_3': 'Restart your device to clear memory',
                        'advanced_solution': 'Uninstall and reinstall the app (your data will be preserved)',
                        'prevention_tip': 'Keep your app updated to avoid known crashes'
                    },
                    'loading_error': {
                        'step_1': 'Check your internet connection stability',
                        'step_2': 'Switch between WiFi and mobile data to test',
                        'step_3': 'Clear the app cache from your device settings',
                        'advanced_solution': 'Reset network settings on your device',
                        'prevention_tip': 'Strong internet connection prevents most loading issues'
                    }
                }
            },
            
            'Product_Questions': {
                'system_prompt': """You are a knowledgeable product specialist and customer education expert.
Your goal is to help customers understand features, learn how to use products, and discover capabilities they might not know about.
Always provide educational value beyond just answering the immediate question.""",
                
                'response_template': """Great question! I love helping customers get the most out of our products.

**Your Question**: "{customer_message}"

**Direct Answer**: {direct_answer}

**Step-by-Step Guide**:
{step_by_step}

**Pro Tips**: 
{pro_tips}

**Related Features** you might find useful:
{related_features}

Would you like me to walk you through any of these features, or do you have questions about other capabilities?""",
                
                'solutions': {
                    'how_to': {
                        'direct_answer': 'Here\'s exactly how to accomplish what you\'re looking for',
                        'step_by_step': '1. Navigate to [location]\n2. Click/tap [action]\n3. Configure [settings]\n4. Save your changes',
                        'pro_tips': 'You can also use keyboard shortcuts or voice commands for faster access',
                        'related_features': 'Automation settings, notification preferences, and sharing options'
                    }
                }
            },
            
            'Complaints': {
                'system_prompt': """You are a senior customer service specialist trained in complaint resolution and service recovery.
Your priority is to acknowledge concerns, take ownership, and turn negative experiences into positive outcomes.
Show genuine empathy and provide concrete action plans.""",
                
                'response_template': """I sincerely apologize for the experience you've had. Your feedback is extremely valuable, and I want to make this right.

**Your Concern**: "{customer_message}"

**My Commitment to You**:
✓ {acknowledgment}
✓ {ownership}
✓ {action_plan}

**What I'm Doing Right Now**:
1. {immediate_action}
2. {follow_up_action}
3. {prevention_action}

**Compensation**: {compensation}

**Personal Follow-Up**: I will personally check in with you in {follow_up_timeline} to ensure everything is resolved to your satisfaction.

**My Direct Contact**: {contact_info}

Your business means everything to us, and I'm committed to exceeding your expectations moving forward.""",
                
                'solutions': {
                    'service_complaint': {
                        'acknowledgment': 'I completely understand your frustration and take full responsibility',
                        'ownership': 'This should never have happened, and I\'m personally going to fix it',
                        'action_plan': 'Implementing immediate changes to prevent this from happening to you or others',
                        'immediate_action': 'Reviewing your entire account experience to identify all issues',
                        'follow_up_action': 'Coordinating with relevant teams to implement improvements',
                        'prevention_action': 'Adding your case to our quality assurance review process',
                        'compensation': 'I\'m applying a service credit and upgrading your account benefits',
                        'follow_up_timeline': '48 hours',
                        'contact_info': 'You can reach me directly at [specialist email/ID]'
                    }
                }
            },
            
            'Compliments': {
                'system_prompt': """You are a customer service representative who excels at acknowledging positive feedback.
Your role is to show genuine appreciation, share the praise with relevant teams, and encourage continued engagement.
Be warm but professional.""",
                
                'response_template': """Thank you so much for taking the time to share this wonderful feedback! 

**Your Message**: "{customer_message}"

This absolutely made my day! {personalized_response}

**Sharing the Love**: {team_sharing}

**As a Thank You**: {appreciation_gesture}

**Keep Exploring**: {additional_value}

Customers like you inspire us to keep improving and providing exceptional service. Thank you for being part of our community!""",
                
                'solutions': {
                    'positive_feedback': {
                        'personalized_response': 'It\'s customers like you who make our work meaningful',
                        'team_sharing': 'I\'m sharing your feedback with the team you mentioned - they\'ll be thrilled!',
                        'appreciation_gesture': 'I\'ve added loyalty points to your account as a small token of our appreciation',
                        'additional_value': 'Have you explored our premium features? I\'d love to show you what else we can do for you'
                    }
                }
            },

            'General_Inquiry': {
                'system_prompt': """You are a friendly and helpful customer service representative.
            Your role is to understand customer needs and either provide direct assistance or route them to the right specialist.
            Always be welcoming and ask clarifying questions to better help the customer.""",
                
                'response_template': """Thank you for contacting us! I'm here to help you with: "{customer_message}"

            To ensure I provide you with the best possible assistance, could you help me understand a bit more about what you need?

            I can help you with:
            • Account and login issues
            • Billing and payment questions  
            • Technical problems with our services
            • Product information and how-to guidance
            • Order and delivery status
            • General questions about our services

            What specific area would you like assistance with today?""",
                
                'solutions': {
                    'default': {}
                }
            },

            'Service_Issues': {
                'system_prompt': """You are a customer service representative specializing in service outages and connectivity issues.
            Your role is to quickly identify service problems and provide status updates and workarounds when possible.""",
                
                'response_template': """I understand you're experiencing service issues. Let me check on this right away.

            **Your Issue**: "{customer_message}"

            **Current Status Check**: {status_check}

            **What I'm Doing**:
            ✓ Checking our service status dashboard
            ✓ Reviewing any reported outages in your area
            ✓ Testing connectivity to our servers

            **Next Steps**: {next_steps}

            I'll keep you updated on the resolution progress. Is there anything else I can help you with in the meantime?""",
                
                'solutions': {
                    'default': {
                        'status_check': 'Investigating service status in your region',
                        'next_steps': 'I\'ll provide updates every 15 minutes until this is resolved'
                    }
                }
            },

            'Delivery_Issues': {
                'system_prompt': """You are a customer service representative specializing in shipping and delivery issues.
            You have access to tracking information and can coordinate with logistics teams to resolve delivery problems.""",
                
                'response_template': """I understand your concern about your delivery. Let me track that down for you immediately.

            **Your Issue**: "{customer_message}"

            **What I'm Checking**:
            ✓ Your order status and tracking information
            ✓ Current location of your package
            ✓ Any delivery exceptions or delays

            **Investigation Results**: {investigation_results}

            **Resolution Plan**: {resolution_plan}

            **Tracking Reference**: {tracking_reference}

            I'll personally monitor this delivery and keep you updated. Would you like me to set up delivery notifications for future orders?""",
                
                'solutions': {
                    'default': {
                        'investigation_results': 'Reviewing your shipment details and current status',
                        'resolution_plan': 'I will coordinate with our logistics team to expedite resolution',
                        'tracking_reference': 'DEL' + str(np.random.randint(100000, 999999))
                    }
                }
            }
        }
    
    def get_device_context(self, message):
        """Extract device context from message"""
        device_indicators = {
            'iphone': 'iPhone/iOS',
            'android': 'Android',
            'ipad': 'iPad',
            'mobile': 'Mobile Device',
            'desktop': 'Desktop/Computer',
            'browser': 'Web Browser'
        }
        
        message_lower = message.lower()
        for indicator, device in device_indicators.items():
            if indicator in message_lower:
                return device
        return 'Please specify your device'
    

    def classify_and_respond(self, customer_message, include_context=True):
        """
        Main function: Classify intent and generate appropriate response
        """
        # Step 1: Classify the intent
        intent_result = predict_intent_with_confidence(customer_message, self.model)
        intent = intent_result['predicted_intent']
        confidence = intent_result.get('confidence', 0)
        
        # Step 2: Get intent-specific prompt (with fallback handling)
        if intent not in self.intent_prompts:
            print(f"⚠️  Unknown intent '{intent}', using General_Inquiry fallback")
            intent = 'General_Inquiry'
            
            # If General_Inquiry also doesn't exist, create a basic fallback
            if intent not in self.intent_prompts:
                return self._create_basic_fallback_response(customer_message, intent_result)
        
        prompt_config = self.intent_prompts[intent]
        
        # Step 3: Generate response based on intent
        try:
            if intent == 'Technical_Issues':
                device_info = self.get_device_context(customer_message)
                
                # Determine specific technical issue type
                if 'crash' in customer_message.lower():
                    solutions = prompt_config['solutions']['app_crash']
                elif 'loading' in customer_message.lower() or 'load' in customer_message.lower():
                    solutions = prompt_config['solutions']['loading_error']
                else:
                    solutions = prompt_config['solutions']['app_crash']  # Default
                
                response = prompt_config['response_template'].format(
                    customer_message=customer_message,
                    device_info=device_info,
                    **solutions
                )
            
            elif intent == 'Billing_Problems':
                solutions = prompt_config['solutions']['default'].copy()
                reference_num = np.random.randint(100000, 999999)
                
                response = prompt_config['response_template'].format(
                    customer_message=customer_message,
                    reference_number=f"BIL{reference_num}",
                    **solutions
                )
            
            elif intent == 'Account_Issues':
                # Determine specific account issue type
                if 'password' in customer_message.lower():
                    solutions = prompt_config['solutions']['password_reset']
                elif 'locked' in customer_message.lower() or 'lock' in customer_message.lower():
                    solutions = prompt_config['solutions']['locked_account']
                else:
                    solutions = prompt_config['solutions']['default']
                
                response = prompt_config['response_template'].format(
                    customer_message=customer_message,
                    **solutions
                )
            
            elif intent == 'Product_Questions':
                solutions = prompt_config['solutions']['how_to']
                
                response = prompt_config['response_template'].format(
                    customer_message=customer_message,
                    **solutions
                )
            
            elif intent == 'Complaints':
                solutions = prompt_config['solutions']['service_complaint']
                
                response = prompt_config['response_template'].format(
                    customer_message=customer_message,
                    **solutions
                )
            
            elif intent == 'Compliments':
                solutions = prompt_config['solutions']['positive_feedback']
                
                response = prompt_config['response_template'].format(
                    customer_message=customer_message,
                    **solutions
                )
            
            elif intent == 'Service_Issues':
                solutions = prompt_config['solutions']['default']
                
                response = prompt_config['response_template'].format(
                    customer_message=customer_message,
                    **solutions
                )
            
            elif intent == 'Delivery_Issues':
                solutions = prompt_config['solutions']['default']
                
                response = prompt_config['response_template'].format(
                    customer_message=customer_message,
                    **solutions
                )
            
            elif intent == 'General_Inquiry':
                solutions = prompt_config['solutions']['default']
                
                response = prompt_config['response_template'].format(
                    customer_message=customer_message,
                    **solutions
                )
            
            else:
                # Final fallback for any unhandled intents
                response = self._create_basic_fallback_response(customer_message, intent_result)['response']
        
        except Exception as e:
            print(f"⚠️  Error generating response for intent '{intent}': {e}")
            return self._create_basic_fallback_response(customer_message, intent_result)
        
        return {
            'intent': intent,
            'confidence': confidence,
            'response': response,
            'system_prompt': prompt_config.get('system_prompt', 'You are a helpful customer service representative.'),
            'metadata': {
                'prompt_version': '1.0',
                'response_type': 'structured',
                'escalation_needed': confidence < 0.7
            }
        }

    def _create_basic_fallback_response(self, customer_message, intent_result):
        """Create a basic fallback response when templates are missing"""
        intent = intent_result['predicted_intent']
        confidence = intent_result.get('confidence', 0)
        
        response = f"""Thank you for contacting us! I understand you need help with: "{customer_message}"

    I want to make sure I give you the best possible assistance. Let me connect you with a specialist who can help you with this {intent.replace('_', ' ').lower()}.

    In the meantime, here are some quick options:
    • For urgent issues, please call our support line
    • For account questions, try our self-service portal  
    • For technical problems, check our troubleshooting guide

    How would you prefer to proceed?"""
        
        return {
            'intent': intent,
            'confidence': confidence,
            'response': response,
            'system_prompt': 'You are a helpful customer service representative providing general assistance.',
            'metadata': {
                'prompt_version': '1.0',
                'response_type': 'fallback',
                'escalation_needed': True
            }
        }

if __name__ == "__main__":

    model = load_intent_classifier('../models/trained/best_intent_classifier.pkl')


    # Initialize the prompt engine
    print("🎯 Initializing Customer Service Prompt Engine...")
    prompt_engine = CustomerServicePromptEngine(model)
    print("✅ Prompt Engine ready!")


    # Test the prompt engineering system
    test_scenarios = [
        "I can't log into my account after password reset",
        "I was charged twice for the same service", 
        "The app keeps crashing when I try to upload photos",
        "How do I cancel my subscription?",
        "Your service is terrible, very disappointed",
        "Thank you so much for the excellent support!",
        "Where is my package, I ordered last week with next day delivery"
    ]

    print("🧪 TESTING PROMPT ENGINEERING SYSTEM")
    print("=" * 60)

    for i, scenario in enumerate(test_scenarios, 1):
        print(f"\n📝 Test {i}: {scenario}")
        print("-" * 40)
        
        result = prompt_engine.classify_and_respond(scenario)
        
        print(f"🎯 Intent: {result['intent']} (Confidence: {result['confidence']:.1%})")
        print("🤖 Response:")
        print(result['response'])
        
        if result['metadata']['escalation_needed']:
            print("⚠️  Low confidence - consider human escalation")
        
        print()