1. data/

load_data.py: Script to load raw CSV/JSON data into pandas DataFrame.
explore_data.py: Perform EDA (plots, missing values, intent distribution).

2. models/

train_baseline.py: Train a scikit-learn model (Logistic Regression, RandomForest).
train_deep_model.py: Train a PyTorch/TensorFlow model (LSTM or Transformer).
responsible_ai.py: Add SHAP interpretability and fairness checks.
prompt_template.py: Generate dynamic prompts using Jinja.
rag.py: Implement FAISS-based retrieval for RAG simulation.

3. utils/

Helper scripts for tokenization, vectorization, and evaluation metrics.

4. templates/

Store Jinja templates for prompt engineering:
Jinja{% if intent == "billing" %}  Please provide billing details for {{ customer_name }}.{% else %}  How can I assist you with {{ intent }} issues?{% endif %}Vis flere linjer


5. reports/

Save visualizations (training curves, confusion matrix, SHAP plots).

6. main.py

Orchestrates the pipeline:

Load data → preprocess → train → evaluate → generate prompts → simulate RAG → interpretability.