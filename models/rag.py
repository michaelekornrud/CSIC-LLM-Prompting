"""
RAG System for Customer Service Intent Classification

This module implements a complete Retrieval-Augmented Generation (RAG) system that combines:
1. Document Retrieval: FAISS-based semantic search for relevant customer service knowledge
2. Intent Classification: ML-based intent detection from customer messages
3. Prompt Engineering: Context-aware response generation

Architecture:
    Customer Message → Intent Classification → Document Retrieval → Prompt Generation → Response
"""

import json
import pickle
import warnings
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional

import numpy as np
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer

from models.trained.inference_utils import predict_intent_with_confidence

warnings.filterwarnings('ignore')


class CustomerServiceKnowledgeBase:
    """
    Knowledge base for customer service responses
    Stores intent-specific examples and solutions
    """
    
    def __init__(self, df: pd.DataFrame):
        """
        Initialize knowledge base from a DataFrame
        
        Args:
            df: DataFrame with 'text' and 'intent' columns
        """
        self.df = df
        self.intent_examples = self._build_intent_examples()
        self.intent_solutions = self._build_solution_templates()
        
    def _build_intent_examples(self) -> Dict[str, List[str]]:
        """Build representative examples for each intent"""
        examples = {}
        for intent in self.df['intent'].unique():
            intent_data = self.df[self.df['intent'] == intent]
            # Get diverse examples (by length)
            intent_data['text_len'] = intent_data['text'].str.len()
            
            # Sample from different length quantiles
            sample_examples = []
            for q in [0.25, 0.5, 0.75]:
                quantile_val = intent_data['text_len'].quantile(q)
                closest_idx = (intent_data['text_len'] - quantile_val).abs().idxmin()
                sample_examples.append(intent_data.loc[closest_idx, 'text'])
            
            examples[intent] = sample_examples
            
        return examples
    
    def _build_solution_templates(self) -> Dict[str, Dict[str, str]]:
        """Define solution templates for each intent category"""
        return {
            'Account_Issues': {
                'immediate_solution': 'Try resetting your password using the "Forgot Password" link on the login page',
                'backup_solution': 'Clear your browser cache and cookies, then try logging in again',
                'additional_resources': 'Our account recovery team is available 24/7 at support@company.com'
            },
            'Billing_Payment': {
                'immediate_solution': 'Check your payment method is up to date in Account Settings',
                'backup_solution': 'Verify the charge on your bank statement matches our billing cycle',
                'additional_resources': 'Contact billing support for detailed invoice breakdown'
            },
            'Service_Outage': {
                'immediate_solution': 'Check our status page at status.company.com for real-time updates',
                'backup_solution': 'Try accessing from a different network or device',
                'additional_resources': 'Follow @CompanySupport for live outage notifications'
            },
            'Product_Feature': {
                'immediate_solution': 'Visit our help center at help.company.com for feature tutorials',
                'backup_solution': 'Check if your app/software is updated to the latest version',
                'additional_resources': 'Join our community forum for tips from other users'
            },
            'Complaint_Feedback': {
                'immediate_solution': 'I\'ve escalated your feedback to our quality assurance team',
                'backup_solution': 'We\'ll follow up with you within 24 hours via email',
                'additional_resources': f'Your feedback case number is #CS-{np.random.randint(10000, 99999)}'
            },
            'Delivery_Shipping': {
                'immediate_solution': 'Track your order using the tracking number sent to your email',
                'backup_solution': 'Contact the shipping carrier directly for delivery updates',
                'additional_resources': 'Orders typically arrive within 3-5 business days'
            },
            'Refund_Cancellation': {
                'immediate_solution': 'Initiate a refund request in your order history section',
                'backup_solution': 'Refunds are processed within 5-7 business days to your original payment method',
                'additional_resources': 'Review our refund policy at company.com/refunds'
            },
            'General_Inquiry': {
                'immediate_solution': 'Browse our FAQ section for answers to common questions',
                'backup_solution': 'Describe your specific need so I can direct you to the right department',
                'additional_resources': 'Our customer service team is here to help 24/7'
            }
        }
    
    def get_examples(self, intent: str, n: int = 3) -> List[str]:
        """Get example messages for a given intent"""
        return self.intent_examples.get(intent, [])[:n]
    
    def get_solutions(self, intent: str) -> Dict[str, str]:
        """Get solution templates for a given intent"""
        return self.intent_solutions.get(intent, self.intent_solutions['General_Inquiry'])


class FAISSVectorStore:
    """
    FAISS-based vector store for semantic search
    Uses sentence transformers for embeddings
    """
    
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        """
        Initialize with a sentence transformer model
        
        Args:
            model_name: Name of the sentence transformer model to use
        """
        print(f"Loading embedding model: {model_name}...")
        self.encoder = SentenceTransformer(model_name)
        self.dimension = self.encoder.get_sentence_embedding_dimension()
        self.index = None
        self.documents = []
        self.metadata = []
        print(f"✓ Embedding model loaded (dimension: {self.dimension})")
    
    def build_index(self, texts: List[str], metadata: Optional[List[dict]] = None):
        """
        Build FAISS index from texts
        
        Args:
            texts: List of text documents to index
            metadata: Optional list of metadata dicts for each document
        """
        print(f"\nBuilding FAISS index for {len(texts)} documents...")
        
        # Store documents and metadata
        self.documents = texts
        self.metadata = metadata if metadata else [{} for _ in texts]
        
        # Generate embeddings
        print("Generating embeddings...")
        embeddings = self.encoder.encode(
            texts,
            show_progress_bar=True,
            batch_size=32
        )
        
        # Create FAISS index
        self.index = faiss.IndexFlatL2(self.dimension)
        self.index.add(embeddings.astype('float32'))
        
        print(f"✓ FAISS index built with {self.index.ntotal} vectors")
    
    def search(self, query: str, k: int = 5) -> List[Tuple[str, dict, float]]:
        """
        Search for similar documents
        
        Args:
            query: Query string to search for
            k: Number of results to return
            
        Returns:
            List of (document, metadata, distance) tuples
        """
        if self.index is None:
            raise ValueError("Index not built. Call build_index() first.")
        
        # Encode query
        query_embedding = self.encoder.encode([query])
        
        # Search
        distances, indices = self.index.search(
            query_embedding.astype('float32'), k
        )
        
        # Return results
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            results.append((
                self.documents[idx],
                self.metadata[idx],
                float(dist)
            ))
        
        return results
    
    def save_index(self, path: str):
        """Save FAISS index to disk"""
        faiss.write_index(self.index, path)
        print(f"✓ Index saved to {path}")
    
    def load_index(self, path: str):
        """Load FAISS index from disk"""
        self.index = faiss.read_index(path)
        print(f"✓ Index loaded from {path}")


class CustomerServiceRAG:
    """
    Complete RAG system for customer service
    Combines intent classification, document retrieval, and response generation
    """
    
    def __init__(self, intent_model, vector_store: FAISSVectorStore, knowledge_base: CustomerServiceKnowledgeBase):
        """
        Initialize RAG system
        
        Args:
            intent_model: Trained intent classification model
            vector_store: FAISS vector store for document retrieval
            knowledge_base: Knowledge base with solutions and examples
        """
        self.intent_model = intent_model
        self.vector_store = vector_store
        self.kb = knowledge_base
        self.conversation_history = []
    
    def process_message(self, customer_message: str, retrieve_k: int = 3) -> Dict[str, Any]:
        """
        Process a customer message through the RAG pipeline
        
        Args:
            customer_message: The customer's message
            retrieve_k: Number of similar documents to retrieve
        
        Returns:
            dict: Complete RAG response with all components
        """
        # Step 1: Intent Classification
        prediction = predict_intent_with_confidence(
            customer_message, self.intent_model
        )

        intent = prediction['predicted_intent']
        confidence = prediction['confidence']
        
        # Step 2: Document Retrieval
        retrieved_docs = self.vector_store.search(customer_message, k=retrieve_k)
        
        # Step 3: Get knowledge base solutions
        solutions = self.kb.get_solutions(intent)
        examples = self.kb.get_examples(intent)
        
        # Step 4: Generate response
        response = self._generate_response(
            customer_message, intent, confidence, retrieved_docs, solutions, examples
        )
        
        # Store in conversation history
        self.conversation_history.append({
            'customer_message': customer_message,
            'intent': intent,
            'confidence': confidence,
            'response': response
        })
        
        return {
            'customer_message': customer_message,
            'intent': intent,
            'confidence': confidence,
            'retrieved_documents': retrieved_docs,
            'solutions': solutions,
            'response': response
        }
    
    def _generate_response(self, message: str, intent: str, confidence: float, 
                          retrieved_docs: List[Tuple], solutions: Dict[str, str], 
                          examples: List[str]) -> str:
        """Generate a context-aware response"""
        
        # Build context from retrieved documents
        similar_cases = "\n".join([
            f"- {doc[0][:100]}..." for doc in retrieved_docs[:2]
        ])
        
        # Generate response based on intent
        response = f"""Thank you for reaching out! I've analyzed your message and identified this as a **{intent.replace('_', ' ')}** issue.

**Your message**: "{message}"

**Recommended Solutions**:

1. **Immediate Action**: {solutions['immediate_solution']}

2. **Alternative Solution**: {solutions['backup_solution']}

3. **Additional Help**: {solutions['additional_resources']}

**Similar cases we've resolved**:
{similar_cases}

**Confidence**: {confidence:.1%} - {'High confidence match' if confidence > 0.7 else 'Please provide more details if this doesn\'t match your issue'}

Is there anything specific about your situation I should know to provide better assistance?
"""
        
        return response
    
    def get_conversation_summary(self) -> str:
        """Get summary of conversation history"""
        if not self.conversation_history:
            return "No conversation history yet."
        
        summary = f"Conversation Summary ({len(self.conversation_history)} interactions):\n"
        for i, conv in enumerate(self.conversation_history, 1):
            summary += f"\n{i}. Customer: {conv['customer_message'][:50]}...\n"
            summary += f"   Intent: {conv['intent']} (confidence: {conv['confidence']:.1%})\n"
        
        return summary


class AdvancedRAGFeatures:
    """
    Advanced features for production RAG systems
    """
    
    @staticmethod
    def rerank_results(query: str, retrieved_docs: List[Tuple], intent: str) -> List[Tuple]:
        """
        Re-rank retrieved documents by intent match and semantic similarity
        
        Args:
            query: Original query string
            retrieved_docs: List of (document, metadata, distance) tuples
            intent: Predicted intent
            
        Returns:
            Re-ranked list of documents
        """
        scored_docs = []
        for doc, meta, dist in retrieved_docs:
            # Boost score if intent matches
            intent_boost = 0.5 if meta.get('intent') == intent else 0.0
            # Lower distance is better, so invert it
            final_score = (1.0 / (1.0 + dist)) + intent_boost
            scored_docs.append((doc, meta, dist, final_score))
        
        # Sort by final score (descending)
        scored_docs.sort(key=lambda x: x[3], reverse=True)
        
        # Return in original format
        return [(doc, meta, dist) for doc, meta, dist, score in scored_docs]
    
    @staticmethod
    def generate_followup_questions(intent: str) -> List[str]:
        """
        Generate intelligent follow-up questions based on intent
        
        Args:
            intent: The detected intent
            
        Returns:
            List of follow-up questions
        """
        followups = {
            'Account_Issues': [
                "What error message are you seeing when you try to log in?",
                "When was the last time you were able to access your account?",
                "Have you tried using the password reset feature?"
            ],
            'Billing_Payment': [
                "Can you provide the transaction ID or date of the charge?",
                "Which payment method did you use?",
                "Have you checked your bank statement for the charge?"
            ],
            'Service_Outage': [
                "What specific service or feature is not working?",
                "Are you seeing any error messages?",
                "Have you tried accessing from a different device or network?"
            ],
            'Product_Feature': [
                "Which specific feature would you like help with?",
                "Have you checked our tutorial videos?",
                "What are you trying to accomplish?"
            ]
        }
        
        return followups.get(intent, [
            "Can you provide more details about your issue?",
            "What have you tried so far?",
            "Is there anything else I should know?"
        ])
    
    @staticmethod
    def calculate_response_urgency(intent: str, confidence: float) -> str:
        """
        Calculate urgency level for routing to appropriate support tier
        
        Args:
            intent: The detected intent
            confidence: Confidence score of the intent prediction
            
        Returns:
            Urgency level string
        """
        urgent_intents = ['Service_Outage', 'Account_Issues', 'Complaint_Feedback']
        
        if intent in urgent_intents and confidence > 0.7:
            return "HIGH - Escalate to human agent"
        elif confidence < 0.5:
            return "MEDIUM - Request clarification"
        else:
            return "LOW - Automated response acceptable"


class RAGSystemManager:
    """
    Manager class for loading and saving RAG system components
    """
    
    @staticmethod
    def build_and_save(df: pd.DataFrame, intent_model_path: str, output_dir: str, 
                       sample_size: Optional[int] = None) -> CustomerServiceRAG:
        """
        Build RAG system from data and save all components
        
        Args:
            df: DataFrame with customer service data
            intent_model_path: Path to trained intent classification model
            output_dir: Directory to save RAG system components
            sample_size: Optional sample size for vector store (None = use all data)
            
        Returns:
            Initialized CustomerServiceRAG instance
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Load intent model
        from models.trained.inference_utils import load_intent_classifier
        print(f"Loading intent model from {intent_model_path}...")
        intent_model = load_intent_classifier(intent_model_path)
        print("✓ Intent model loaded")
        
        # Build knowledge base
        print("\nBuilding knowledge base...")
        kb = CustomerServiceKnowledgeBase(df)
        print(f"✓ Knowledge base built with {len(kb.intent_solutions)} intents")
        
        # Build vector store
        if sample_size and sample_size < len(df):
            df_sample = df.sample(n=sample_size, random_state=42)
            print(f"\nSampling {sample_size} documents for vector store...")
        else:
            df_sample = df
            print(f"\nUsing all {len(df)} documents for vector store...")
        
        documents = df_sample['text'].tolist()
        metadata = df_sample[['intent', 'author_id']].to_dict('records')
        
        vector_store = FAISSVectorStore()
        vector_store.build_index(documents, metadata)
        
        # Initialize RAG system
        rag_system = CustomerServiceRAG(intent_model, vector_store, kb)
        print("\n✓ RAG system initialized")
        
        # Save components
        print(f"\nSaving RAG system to {output_path}...")
        
        # Save FAISS index
        vector_store.save_index(str(output_path / 'faiss_index.bin'))
        
        # Save documents and metadata
        with open(output_path / 'documents.pkl', 'wb') as f:
            pickle.dump({
                'documents': vector_store.documents,
                'metadata': vector_store.metadata
            }, f)
        print("✓ Documents saved")
        
        # Save knowledge base
        with open(output_path / 'knowledge_base.pkl', 'wb') as f:
            pickle.dump(kb, f)
        print("✓ Knowledge base saved")
        
        # Save configuration
        config = {
            'model_name': 'all-MiniLM-L6-v2',
            'num_documents': len(vector_store.documents),
            'embedding_dimension': vector_store.dimension,
            'intents': list(kb.intent_solutions.keys()),
            'version': '1.0.0'
        }
        
        with open(output_path / 'config.json', 'w') as f:
            json.dump(config, f, indent=2)
        print("✓ Configuration saved")
        
        print(f"\n✓ RAG system successfully saved to {output_path}")
        return rag_system
    
    @staticmethod
    def load(system_dir: str, intent_model_path: str) -> CustomerServiceRAG:
        """
        Load a saved RAG system
        
        Args:
            system_dir: Directory containing saved RAG components
            intent_model_path: Path to intent classification model
            
        Returns:
            Loaded CustomerServiceRAG instance
        """
        system_path = Path(system_dir)
        
        # Load configuration
        with open(system_path / 'config.json', 'r') as f:
            config = json.load(f)
        print(f"Loading RAG system v{config['version']}...")
        
        # Load intent model
        from models.trained.inference_utils import load_intent_classifier
        intent_model = load_intent_classifier(intent_model_path)
        print("✓ Intent model loaded")
        
        # Load knowledge base
        with open(system_path / 'knowledge_base.pkl', 'rb') as f:
            kb = pickle.load(f)
        print("✓ Knowledge base loaded")
        
        # Load vector store
        vector_store = FAISSVectorStore(model_name=config['model_name'])
        vector_store.load_index(str(system_path / 'faiss_index.bin'))
        
        # Load documents and metadata
        with open(system_path / 'documents.pkl', 'rb') as f:
            doc_data = pickle.load(f)
            vector_store.documents = doc_data['documents']
            vector_store.metadata = doc_data['metadata']
        print("✓ Vector store loaded")
        
        # Initialize RAG system
        rag_system = CustomerServiceRAG(intent_model, vector_store, kb)
        print(f"✓ RAG system loaded successfully ({config['num_documents']} documents)")
        
        return rag_system


def evaluate_retrieval_quality(rag_system: CustomerServiceRAG, 
                               test_cases: List[str], k: int = 5) -> pd.DataFrame:
    """
    Evaluate retrieval quality by checking if retrieved documents match the predicted intent
    
    Args:
        rag_system: The RAG system to evaluate
        test_cases: List of test messages
        k: Number of documents to retrieve
        
    Returns:
        DataFrame with evaluation results
    """
    results = []
    
    for message in test_cases:
        result = rag_system.process_message(message, retrieve_k=k)
        
        # Check how many retrieved docs match the predicted intent
        predicted_intent = result['intent']
        matching_docs = sum(
            1 for doc, meta, dist in result['retrieved_documents']
            if meta.get('intent') == predicted_intent
        )
        
        precision_at_k = matching_docs / k
        
        results.append({
            'message': message,
            'intent': predicted_intent,
            'confidence': result['confidence'],
            'matching_docs': matching_docs,
            'precision_at_k': precision_at_k
        })
    
    return pd.DataFrame(results)


if __name__ == "__main__":
    """
    Example usage of the RAG system
    """
    from pathlib import Path
    
    # Setup paths
    project_root = Path(__file__).parent.parent
    data_path = project_root / 'data' / 'processed' / 'tweets_with_intents.csv'
    model_path = project_root / 'models' / 'trained' / 'best_intent_classifier.pkl'
    output_dir = project_root / 'models' / 'rag_system'
    
    # Load data
    print("Loading dataset...")
    df = pd.read_csv(data_path)
    print(f"✓ Loaded {len(df):,} records")
    
    # Build and save RAG system
    rag_system = RAGSystemManager.build_and_save(
        df=df,
        intent_model_path=str(model_path),
        output_dir=str(output_dir),
        sample_size=10000  # Use 10k samples for faster indexing
    )
    
    # Test the system
    test_messages = [
        "I can't access my account, forgot my password",
        "Why was I charged twice this month?",
        "Your service has been down for 2 hours!",
    ]
    
    print("\n" + "="*80)
    print("TESTING RAG SYSTEM")
    print("="*80)
    
    for message in test_messages:
        print(f"\n{'='*80}")
        result = rag_system.process_message(message)
        print(f"Message: {result['customer_message']}")
        print(f"Intent: {result['intent']} ({result['confidence']:.1%})")
        print("\nResponse:")
        print(result['response'])
