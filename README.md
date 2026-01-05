# Customer Service Intent Classification & RAG System

A complete machine learning pipeline for customer service automation using intent classification, retrieval-augmented generation (RAG), and prompt engineering. Built with scikit-learn, FastAPI, FAISS, and modern NLP techniques.

## 🎯 Project Overview

This project analyzes customer service tweets to:
1. **Discover intents** through unsupervised clustering (KMeans)
2. **Train ML models** to classify customer intents (9 categories)
3. **Build a RAG system** with FAISS vector search for context-aware responses
4. **Serve predictions** via a FastAPI REST API
5. **Generate dynamic prompts** using Jinja2 templates

### Intent Categories
- Account Issues
- Billing/Payment Problems
- Service Outages
- Product Features/Questions
- Complaints/Feedback
- Delivery/Shipping
- Refund/Cancellation
- General Inquiry

## 📁 Project Structure

```
csic-llm-prompting/
├── api/                        # FastAPI REST API
│   └── rag_chat.py            # RAG-powered chat endpoint
├── data/                       # Data processing & EDA
│   ├── explore_data.py        # Clustering-based intent discovery
│   ├── raw/                   # Raw customer service tweets
│   └── processed/             # Labeled dataset with intents
├── models/                     # ML models & training
│   ├── train_model.py         # Train intent classifiers
│   ├── rag.py                 # RAG system implementation
│   ├── trained/               # Saved models & metadata
│   └── rag_system/            # FAISS index & knowledge base
├── notebooks/                  # Jupyter notebooks
│   ├── explore_data.ipynb     # Interactive EDA
│   ├── train_models.ipynb     # Model training experiments
│   ├── rag.ipynb              # RAG system development
│   └── promt_engineering.ipynb # Prompt template design
├── prompt_engine/              # Jinja2 prompt templates
├── reports/                    # Visualizations & analysis
└── main.py                     # End-to-end pipeline orchestration
```

## 🚀 Getting Started

### Prerequisites

- **Python 3.13+** (required)
- **uv** package manager (recommended) or pip
- ~4GB disk space for dependencies
- ~2GB for datasets and models

### Installation

#### 1. Clone the Repository
```bash
git clone <your-repo-url>
cd CSIC-LLM-Prompting
```

#### 2. Install Dependencies

**Using uv (recommended):**
```bash
# Install uv if you don't have it
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install project dependencies
uv sync
```

**Using pip:**
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -e .
```

#### 3. Download the Dataset

The project uses the Twitter Customer Support Dataset. Place it in `data/raw/twcs.csv`.

**Option A: Download manually**
- Dataset: [Kaggle - Customer Support on Twitter](https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter)
- Place the file at: `data/raw/twcs.csv`

**Option B: Use sample data**
```bash
# A sample dataset is provided for quick testing
# The pipeline will work with data/raw/sample.csv
```

### 📊 Usage

#### Step 1: Explore Data & Discover Intents

Run unsupervised clustering to discover customer intents:

```bash
python data/explore_data.py
```

**Output:**
- `data/processed/tweets_with_intents.csv` - Labeled dataset
- `data/reports/clustering_analysis.png` - Elbow curve & silhouette analysis
- `data/reports/eda_summary_report.md` - Summary statistics

**What it does:**
- Loads ~1.4M customer service tweets
- Performs TF-IDF vectorization
- Runs KMeans clustering to discover 8-9 intent categories
- Assigns semantic labels to clusters
- Generates comprehensive EDA report

#### Step 2: Train Intent Classification Models

Train and compare multiple ML models:

```bash
cd models
python train_model.py
```

**Output:**
- `models/trained/best_intent_classifier.pkl` - Best performing model
- `models/trained/model_metadata.json` - Model performance metrics
- `models/trained/training_results.csv` - Detailed comparison results
- `reports/plots/model_comparison.png` - Performance visualization

**Models trained:**
- Logistic Regression
- Random Forest
- Linear SVM
- Multinomial Naive Bayes
- SGD Classifier

**Typical accuracy:** 85-92% on test set

#### Step 3: Build RAG System

Build the FAISS vector store and knowledge base:

```bash
python models/rag.py
```

**Output:**
- `models/rag_system/faiss_index.bin` - FAISS vector index
- `models/rag_system/documents.pkl` - Document embeddings
- `models/rag_system/knowledge_base.pkl` - Intent solutions & examples
- `models/rag_system/config.json` - System configuration

**What it does:**
- Creates embeddings for 10k+ customer service examples
- Builds FAISS index for fast semantic search
- Generates solution templates for each intent
- Tests the system with sample queries

#### Step 4: Run the API

Start the FastAPI server:

```bash
uv run uvicorn api.rag_chat:app --reload
```

The API will be available at:
- **Swagger UI:** http://127.0.0.1:8000/docs
- **Base URL:** http://127.0.0.1:8000

**API Endpoints:**

**1. Health Check**
```bash
GET http://127.0.0.1:8000/
```

**2. Chat with RAG System**
```bash
POST http://127.0.0.1:8000/chat
Content-Type: application/json

{
  "message": "I can't login to my account!"
}
```

**Response:**
```json
{
  "customer_message": "I can't login to my account!",
  "intent": "Account_Issues",
  "confidence": 0.95,
  "response": "Thank you for reaching out! I've analyzed your message and identified this as a **Account Issues** issue.\n\n**Your message**: \"I can't login to my account!\"\n\n**Recommended Solutions**:\n\n1. **Immediate Action**: Try resetting your password using the \"Forgot Password\" link on the login page\n\n2. **Alternative Solution**: Clear your browser cache and cookies, then try logging in again\n\n3. **Additional Help**: Our account recovery team is available 24/7 at support@company.com\n\n**Similar cases we've resolved**:\n- Customer had login issues after password reset...\n- Account locked due to multiple failed attempts...\n\n**Confidence**: 95.0% - High confidence match\n\nIs there anything specific about your situation I should know to provide better assistance?"
}
```

### 🧪 Testing the API

#### Using curl
```bash
# Health check
curl http://127.0.0.1:8000/

# Chat request
curl -X POST "http://127.0.0.1:8000/chat" \
  -H "Content-Type: application/json" \
  -d '{"message": "Why was I charged twice this month?"}'
```

#### Using Python
```python
import requests

response = requests.post(
    "http://127.0.0.1:8000/chat",
    json={"message": "Your service has been down for hours!"}
)

print(response.json())
```

#### Using Postman
1. **Method:** POST
2. **URL:** `http://127.0.0.1:8000/chat`
3. **Headers:** `Content-Type: application/json`
4. **Body (raw JSON):**
   ```json
   {
     "message": "I need help with my order"
   }
   ```

#### Using Swagger UI (Easiest)
1. Open http://127.0.0.1:8000/docs in your browser
2. Click on `/chat` endpoint
3. Click **"Try it out"**
4. Enter your message in the JSON body
5. Click **"Execute"**

## 🛠️ Development

### Running Jupyter Notebooks

```bash
# Start Jupyter
uv run jupyter lab

# Or use VS Code's built-in notebook support
```

**Available notebooks:**
- `explore_data.ipynb` - Interactive data exploration
- `train_models.ipynb` - Model training experiments
- `rag.ipynb` - RAG system development (original implementation)
- `promt_engineering.ipynb` - Prompt template design

### Project Pipeline (Full Run)

To run the complete pipeline from scratch:

```bash
# 1. Discover intents from data
python data/explore_data.py

# 2. Train classification models
cd models && python train_model.py

# 3. Build RAG system
python models/rag.py

# 4. Start API server
cd .. && uv run uvicorn api.rag_chat:app --reload
```

### Customizing the RAG System

Edit `models/rag.py` to customize:
- **Solution templates** - Modify `CustomerServiceKnowledgeBase._build_solution_templates()`
- **Vector store size** - Adjust `sample_size` parameter in `RAGSystemManager.build_and_save()`
- **Embedding model** - Change model in `FAISSVectorStore.__init__()`
- **Retrieval count** - Modify `retrieve_k` parameter in `process_message()`

## 📊 Model Performance

Based on the latest training run:

| Model | Accuracy | Precision | Recall | F1-Score | Training Time |
|-------|----------|-----------|--------|----------|---------------|
| **Logistic Regression** | 89.2% | 0.88 | 0.89 | 0.88 | ~45s |
| Random Forest | 87.5% | 0.86 | 0.87 | 0.86 | ~120s |
| Linear SVM | 88.8% | 0.87 | 0.88 | 0.87 | ~60s |
| Naive Bayes | 84.3% | 0.83 | 0.84 | 0.83 | ~15s |
| SGD Classifier | 88.1% | 0.87 | 0.88 | 0.87 | ~30s |

**Best Model:** Logistic Regression with TF-IDF (1-2 gram) features

## 🏗️ Architecture

### RAG System Pipeline

```
Customer Message
      ↓
Intent Classification (ML Model)
      ↓
Document Retrieval (FAISS)
      ↓
Context Assembly (Knowledge Base)
      ↓
Response Generation (Template + Retrieved Docs)
      ↓
Formatted Response
```

### Technology Stack

- **ML Framework:** scikit-learn
- **Vector Search:** FAISS (Facebook AI Similarity Search)
- **Embeddings:** sentence-transformers (all-MiniLM-L6-v2)
- **API Framework:** FastAPI
- **Prompt Engineering:** Jinja2
- **Data Processing:** pandas, numpy
- **Visualization:** matplotlib, seaborn

## 📈 Key Features

✅ **Unsupervised Intent Discovery** - Automatically discovers intent categories from unlabeled data  
✅ **Multiple Model Comparison** - Trains and compares 5+ ML algorithms  
✅ **RAG System** - Context-aware responses using semantic search  
✅ **REST API** - Production-ready FastAPI endpoints  
✅ **Auto-generated Docs** - Interactive API documentation with Swagger UI  
✅ **Lazy Loading** - Efficient memory management for large models  
✅ **Confidence Scoring** - Returns prediction confidence for each response  
✅ **Solution Templates** - Pre-defined responses for each intent category  
✅ **Semantic Search** - Finds similar past cases for better context  

## 🔧 Troubleshooting

### API won't start
```bash
# Make sure you've built the RAG system first
python models/rag.py

# Check if port 8000 is already in use
lsof -i :8000
# Kill the process if needed
kill -9 <PID>
```

### ModuleNotFoundError
```bash
# Reinstall dependencies
uv sync

# Or with pip
pip install -e .
```

### FAISS index not found
```bash
# Rebuild the RAG system
python models/rag.py
```

### Out of memory during training
```python
# Edit models/train_model.py
# Reduce max_features in TfidfVectorizer:
TfidfVectorizer(max_features=5000)  # Instead of 15000
```

### Large files (.pkl, .bin) in Git
These files are ignored by `.gitignore`. If you accidentally committed them:
```bash
# Remove from Git but keep locally
git rm --cached models/rag_system/*.pkl models/rag_system/*.bin
git commit -m "Remove large binary files from tracking"
```

## 📝 Configuration Files

### pyproject.toml
Project metadata and dependencies managed by `uv` or `pip`.

### .gitignore
Excludes:
- Model files (`.pkl`, `.bin`)
- Virtual environments
- Data files (`.csv`, `.json`)
- Cache directories (`__pycache__`, `.venv`)

## 🎓 Learning Resources

- **FAISS Documentation:** https://faiss.ai/
- **FastAPI Tutorial:** https://fastapi.tiangolo.com/tutorial/
- **Sentence Transformers:** https://www.sbert.net/
- **scikit-learn Guide:** https://scikit-learn.org/stable/user_guide.html

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is for educational purposes as part of the CSIC LLM Prompting course.

## 👤 Author

**Ole-Michael Ekornrud**

## 🙏 Acknowledgments

- Twitter Customer Support Dataset from Kaggle
- FAISS by Facebook AI Research
- FastAPI by Sebastián Ramírez
- Sentence Transformers by UKPLab

---

**Last Updated:** January 5, 2026  
**Version:** 0.1.0
