import marimo

__generated_with = "0.17.7"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _():
    import marimo as mo
    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Sprint 2: Advanced NLP (Weeks 3-4)

    Welcome to **Sprint 2** of the Customer Feedback Insight Platform! Building on Sprint 1's foundation, this sprint focuses on advanced NLP techniques to significantly improve model performance through transformer-based models, topic modeling, and text summarization.

    ## IMPORTANT: Windows Setup Required

    **If you're on Windows and get a DLL error when importing PyTorch/Transformers:**

    You need to install Microsoft Visual C++ Redistributable:
    1. Download: [vc_redist.x64.exe](https://aka.ms/vs/17/release/vc_redist.x64.exe)
    2. Run the installer
    3. Restart your terminal/marimo
    4. Reactivate your virtual environment

    This is a one-time system requirement for PyTorch on Windows.

    ---

    ## Sprint Goals

    By the end of Sprint 2, you will have:
    - Fine-tuned transformer model for sentiment analysis
    - Topic modeling pipeline with coherent clusters
    - Extractive and abstractive summarization
    - Comprehensive comparison with baseline models
    - Human-validated topic coherence

    **Timeline:** Weeks 3-4 | **Demo:** End of Week 4
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## Sprint 2 Deliverables

    ### Required Outputs

    1. **Fine-tuned Transformer** - Improved sentiment classifier (BERT/DistilBERT/RoBERTa)
    2. **Topic Modeling Pipeline** - BERTopic or LDA with interpretable topics
    3. **Extractive Summarizer** - TextRank or similar algorithm
    4. **Abstractive Prototype** - Simple transformer-based summarization
    5. **Comparison Notebook** - Metrics comparing baseline vs advanced models
    6. **Error Analysis** - Confusion cases and improvement areas

    ### Acceptance Criteria

    - Topic coherence examples validated by humans
    - Summaries judged as useful on blind samples
    - Advanced models outperform baseline significantly
    - Topics are coherent and actionable
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## Part 1: Transformer-Based Sentiment Analysis

    ### Theory: Why Transformers?

    **Limitations of Traditional Models:**
    - **Bag-of-Words (TF-IDF):** Ignores word order and context
    - **No semantic understanding:** "not good" vs "good" treated independently
    - **Limited feature engineering:** Manual feature selection required

    **Transformer Advantages:**
    - **Contextual embeddings:** Same word, different meanings based on context
    - **Attention mechanism:** Focuses on relevant parts of text
    - **Transfer learning:** Pre-trained on massive corpora
    - **State-of-the-art performance:** Consistently outperforms traditional methods

    ### Popular Transformer Models

    | Model | Parameters | Speed | Accuracy | Use Case |
    |-------|------------|-------|----------|----------|
    | **DistilBERT** | 66M | Fast | Good | Production, limited resources |
    | **BERT-base** | 110M | Medium | Better | Balanced performance |
    | **RoBERTa** | 125M | Medium | Best | Research, high accuracy needed |
    | **ALBERT** | 12M | Fast | Good | Memory-constrained environments |

    ### Key Concepts

    - **Fine-tuning:** Adapting pre-trained model to specific task
    - **Tokenization:** Breaking text into subword units (WordPiece, BPE)
    - **[CLS] token:** Special token for classification tasks
    - **Max sequence length:** Typically 512 tokens
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Practical Example: Fine-tuning DistilBERT

    Let's fine-tune a DistilBERT model for sentiment analysis:
    """)
    return


@app.cell
def _():
    import pandas as pd
    import numpy as np
    from datetime import datetime, timedelta
    import random

    # Create sample dataset (same as Sprint 1 but ensuring we have it)
    random.seed(42)

    positive_texts = [
        'Great product! Fast delivery and excellent quality.',
        'Love this! Will definitely buy again.',
        'Good value for money. Satisfied with my purchase.',
        'Fantastic! Exceeded my expectations.',
        'Amazing quality and great customer service.',
        'Best purchase I have made in a long time.',
        'Highly recommend this product to everyone.',
        'Excellent experience from start to finish.',
    ]

    negative_texts = [
        'The item arrived damaged. Very disappointed.',
        'Worst purchase ever. Do not recommend.',
        'The product description was misleading.',
        'Terrible quality. Requesting a refund.',
        'Poor customer service and defective product.',
        'Waste of money. Very unsatisfied.',
        'Product broke after one use. Disappointing.',
        'Not as described. Very frustrating experience.',
    ]

    neutral_texts = [
        'Average experience. Product is okay.',
        'It is fine. Nothing special but does the job.',
        'Decent product for the price.',
        'Okay purchase. Met basic expectations.',
        'Acceptable quality. Not great but not terrible.',
        'Standard product. Nothing to complain about.',
        'Fair value. Works as expected.',
        'Mediocre experience overall.',
    ]

    # Generate samples
    sample_data = {
        'text': positive_texts + negative_texts + neutral_texts,
        'sentiment': ['positive']*len(positive_texts) + ['negative']*len(negative_texts) + ['neutral']*len(neutral_texts),
        'label': [2]*len(positive_texts) + [0]*len(negative_texts) + [1]*len(neutral_texts)
    }

    df = pd.DataFrame(sample_data)

    # Shuffle
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    print(f"Dataset size: {len(df)}")
    print(f"Class distribution:\n{df['sentiment'].value_counts()}")
    return df, pd


@app.cell
def _(df):
    from sklearn.model_selection import train_test_split

    # Split data
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        df['text'].tolist(),
        df['label'].tolist(),
        test_size=0.3,
        random_state=42,
        stratify=df['label']
    )

    print(f"Training samples: {len(train_texts)}")
    print(f"Validation samples: {len(val_texts)}")
    return train_labels, train_texts, val_labels, val_texts


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Transformer Tokenization

    Transformers use specialized tokenizers that break text into subword units:
    """)
    return


@app.cell
def _(train_texts):
    from transformers import AutoTokenizer

    # Load pre-trained tokenizer
    model_name = "distilbert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Example tokenization
    sample_text = train_texts[0]
    tokens = tokenizer.tokenize(sample_text)
    token_ids = tokenizer.encode(sample_text)

    print(f"Original text: {sample_text}")
    print(f"\nTokens: {tokens}")
    print(f"\nToken IDs: {token_ids}")
    print(f"\nVocabulary size: {tokenizer.vocab_size}")
    return model_name, tokenizer


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Model Training Setup

    For demonstration purposes, we'll show the setup. In practice, you'd train on larger datasets with GPU support:
    """)
    return


@app.cell
def _(tokenizer, train_labels, train_texts, val_labels, val_texts):
    from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
    import torch
    from torch.utils.data import Dataset

    # Custom dataset class
    class FeedbackDataset(Dataset):
        def __init__(self, texts, labels, tokenizer, max_length=128):
            self.texts = texts
            self.labels = labels
            self.tokenizer = tokenizer
            self.max_length = max_length

        def __len__(self):
            return len(self.texts)

        def __getitem__(self, idx):
            text = self.texts[idx]
            label = self.labels[idx]

            encoding = self.tokenizer(
                text,
                add_special_tokens=True,
                max_length=self.max_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )

            return {
                'input_ids': encoding['input_ids'].flatten(),
                'attention_mask': encoding['attention_mask'].flatten(),
                'labels': torch.tensor(label, dtype=torch.long)
            }

    # Create datasets
    train_dataset = FeedbackDataset(train_texts, train_labels, tokenizer)
    val_dataset = FeedbackDataset(val_texts, val_labels, tokenizer)

    print(f"Datasets created")
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    return (AutoModelForSequenceClassification,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Training Configuration

    **Note:** This is a demonstration setup. For real training:
    - Use GPU if available (`device='cuda'`)
    - Train for more epochs (3-5)
    - Use larger datasets (1000+ samples)
    - Monitor validation loss to avoid overfitting
    """)
    return


@app.cell
def _(AutoModelForSequenceClassification, model_name):
    # Load pre-trained model
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=3  # positive, negative, neutral
    )

    print(f"Model loaded: {model_name}")
    print(f"Number of parameters: {model.num_parameters():,}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Task 1: Fine-tune Transformer Model

    **Objective:** Train an improved sentiment classifier using transformers

    **Steps:**
    1. Choose appropriate model (DistilBERT for speed, BERT for accuracy)
    2. Prepare dataset with proper tokenization
    3. Set up training arguments:
       - Learning rate: 2e-5 to 5e-5
       - Batch size: 16-32 (based on GPU memory)
       - Epochs: 3-5
       - Warmup steps: 500
    4. Implement early stopping on validation loss
    5. Track metrics: Accuracy, F1-score per class, Loss
    6. Save best model checkpoint
    7. Compare with Sprint 1 baseline

    **Acceptance:** Model achieves >10% improvement over baseline on validation set
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## Part 2: Topic Modeling

    ### Theory: Discovering Hidden Topics

    **What is Topic Modeling?**
    Topic modeling automatically discovers abstract "topics" within a collection of documents. Each topic is a distribution over words, and each document is a mixture of topics.

    **Use Cases for Customer Feedback:**
    - Identify common pain points
    - Discover emerging issues
    - Group similar feedback
    - Track topic trends over time

    ### Popular Approaches

    | Method | Approach | Pros | Cons |
    |--------|----------|------|------|
    | **LDA** | Probabilistic | Interpretable, fast | Needs preprocessing, fixed topics |
    | **NMF** | Matrix factorization | Fast, sparse | Linear only |
    | **BERTopic** | Embeddings + clustering | State-of-art, dynamic | Slower, needs more data |
    | **Top2Vec** | Embeddings only | No preprocessing | Less control |

    ### BERTopic Pipeline

    1. **Generate embeddings** - Use sentence transformers
    2. **Reduce dimensionality** - UMAP for visualization
    3. **Cluster documents** - HDBSCAN for density-based clustering
    4. **Extract topics** - c-TF-IDF for topic representation
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Practical Example: Topic Modeling with BERTopic
    """)
    return


@app.cell
def _(df):
    from bertopic import BERTopic
    from sentence_transformers import SentenceTransformer

    # Prepare documents
    documents = df['text'].tolist()

    # Initialize BERTopic with smaller embedding model for demo
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

    topic_model = BERTopic(
        embedding_model=embedding_model,
        min_topic_size=2,  # Small for demo dataset
        nr_topics="auto",
        verbose=True
    )

    # Fit the model
    topics, probs = topic_model.fit_transform(documents)

    print(f"Number of topics found: {len(set(topics)) - 1}")  # -1 excludes outliers (topic -1)
    print(f"Number of outliers: {sum(1 for t in topics if t == -1)}")
    return documents, topic_model, topics


@app.cell
def _(topic_model):
    # Get topic information
    topic_info = topic_model.get_topic_info()
    topic_info
    return


@app.cell
def _(topic_model, topics):
    # Show topics with their top words
    for topic_id in set(topics):
        if topic_id != -1:  # Skip outliers
            topic_words = topic_model.get_topic(topic_id)
            print(f"\nTopic {topic_id}:")
            print(f"Top 5 words: {', '.join([word for word, score in topic_words[:5]])}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Topic Visualization

    BERTopic provides several visualization methods:
    """)
    return


@app.cell
def _(topic_model):
    # Visualize topics (returns plotly figure)
    fig_topics = topic_model.visualize_topics()
    fig_topics
    return


@app.cell
def _(topic_model):
    # Visualize topic hierarchy
    fig_hierarchy = topic_model.visualize_hierarchy()
    fig_hierarchy
    return


@app.cell
def _(documents, topic_model):
    # Visualize documents (2D representation)
    fig_docs = topic_model.visualize_documents(documents)
    fig_docs
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Topic Coherence Evaluation

    **Human Validation Checklist:**

    For each topic, evaluate:
    1. **Semantic coherence:** Do the top words make sense together?
    2. **Distinctiveness:** Is this topic different from others?
    3. **Actionability:** Can we take business action based on this?
    4. **Representative documents:** Do sample documents match the topic?

    **Metrics:**
    - **Topic coherence (C_v):** Measures semantic similarity of top words
    - **Topic diversity:** Percentage of unique words across topics
    - **Perplexity:** For LDA models (lower is better)
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Practical Example: LDA Topic Modeling

    Let's also implement traditional LDA for comparison:
    """)
    return


@app.cell
def _(documents):
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.decomposition import LatentDirichletAllocation

    # Vectorize documents
    vectorizer = CountVectorizer(
        max_features=100,
        stop_words='english',
        min_df=2
    )

    doc_term_matrix = vectorizer.fit_transform(documents)

    # Train LDA model
    n_topics = 3
    lda_model = LatentDirichletAllocation(
        n_components=n_topics,
        random_state=42,
        max_iter=20
    )

    lda_topics = lda_model.fit_transform(doc_term_matrix)

    print(f"LDA trained with {n_topics} topics")
    return lda_model, n_topics, vectorizer


@app.cell
def _(lda_model, n_topics, vectorizer):
    # Display LDA topics
    feature_names = vectorizer.get_feature_names_out()

    for idx in range(n_topics):
        print(f"\nLDA Topic {idx}:")
        top_indices = lda_model.components_[idx].argsort()[-10:][::-1]
        top_words = [feature_names[i] for i in top_indices]
        print(f"Top words: {', '.join(top_words)}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Task 2: Build Topic Modeling Pipeline

    **Objective:** Create interpretable topic clusters from customer feedback

    **Steps:**
    1. Choose approach: BERTopic (recommended) or LDA
    2. Prepare data:
       - Clean and preprocess text
       - Remove very short documents (< 3 words)
    3. Configure parameters:
       - BERTopic: min_topic_size, nr_topics
       - LDA: n_components, learning_method
    4. Train model and extract topics
    5. Validate coherence with team members
    6. Assign meaningful labels to topics
    7. Create visualization dashboard
    8. Save topic model for inference

    **Deliverables:**
    - `src/feedback_insights/topic_model.py` with training code
    - Topic visualization plots
    - Human validation report (2-3 reviewers)
    - Saved model in `models/checkpoints/topic_model/`

    **Acceptance:** 3+ topics with >0.4 coherence score, validated by humans
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## Part 3: Text Summarization

    ### Theory: Extractive vs Abstractive

    **Extractive Summarization:**
    - Selects important sentences from original text
    - Preserves original wording
    - Faster and more reliable
    - Good for: Long documents, factual content

    **Abstractive Summarization:**
    - Generates new sentences
    - More human-like
    - Requires large models (T5, BART, GPT)
    - Good for: Short creative summaries

    ### Popular Algorithms

    | Method | Type | Complexity | Quality |
    |--------|------|------------|---------|
    | **TextRank** | Extractive | Low | Good |
    | **LSA** | Extractive | Medium | Decent |
    | **LexRank** | Extractive | Medium | Good |
    | **BART** | Abstractive | High | Excellent |
    | **T5** | Abstractive | High | Excellent |
    | **Pegasus** | Abstractive | High | Best |
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Practical Example: Extractive Summarization with TextRank
    """)
    return


@app.cell
def _():
    import numpy as np
    from sklearn.metrics.pairwise import cosine_similarity
    from sentence_transformers import SentenceTransformer
    import nltk

    # Download required NLTK data
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt', quiet=True)

    def textrank_summarize(text, num_sentences=2, embedding_model=None):
        """
        Extractive summarization using TextRank algorithm.

        Args:
            text: Input text to summarize
            num_sentences: Number of sentences to extract
            embedding_model: SentenceTransformer model for embeddings

        Returns:
            Summary string
        """
        if embedding_model is None:
            embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

        # Split into sentences
        sentences = nltk.sent_tokenize(text)

        if len(sentences) <= num_sentences:
            return text

        # Generate embeddings
        embeddings = embedding_model.encode(sentences)

        # Calculate similarity matrix
        similarity_matrix = cosine_similarity(embeddings)

        # Apply TextRank (PageRank on similarity graph)
        scores = np.ones(len(sentences))
        damping = 0.85

        for _ in range(10):  # Iterations
            new_scores = (1 - damping) + damping * similarity_matrix.T.dot(scores)
            scores = new_scores

        # Select top sentences
        ranked_indices = scores.argsort()[-num_sentences:][::-1]
        ranked_indices.sort()  # Keep original order

        summary = ' '.join([sentences[i] for i in ranked_indices])
        return summary

    # Test on sample text
    long_text = """
    I recently purchased this product and initially was very excited about it. 
    The packaging was excellent and arrived on time. 
    However, after using it for a week, I noticed some quality issues. 
    The material seems cheaper than advertised. 
    Customer service was helpful when I reached out. 
    Overall, it's an okay product but not worth the premium price.
    """

    summary = textrank_summarize(long_text, num_sentences=2)
    print("Original text:")
    print(long_text.strip())
    print("\nSummary:")
    print(summary)
    return long_text, summary


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Practical Example: Abstractive Summarization

    Using a pre-trained transformer for abstractive summarization:
    """)
    return


@app.cell
def _(long_text):
    from transformers import pipeline

    # Load summarization pipeline (using smaller model for demo)
    summarizer = pipeline(
        "summarization",
        model="facebook/bart-large-cnn",
        device=-1  # Use CPU (-1), or 0 for GPU
    )

    # Generate abstractive summary
    abstractive_summary = summarizer(
        long_text.strip(),
        max_length=50,
        min_length=20,
        do_sample=False
    )

    print("Original text:")
    print(long_text.strip())
    print("\nAbstractive summary:")
    print(abstractive_summary[0]['summary_text'])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Summarization Quality Metrics

    **ROUGE Scores (Recall-Oriented Understudy for Gisting Evaluation):**

    - **ROUGE-1:** Overlap of unigrams
    - **ROUGE-2:** Overlap of bigrams
    - **ROUGE-L:** Longest common subsequence

    **Formula:**

    $$
    ROUGE\text{-}N = \frac{\sum_{S \in \text{Reference}} \sum_{\text{gram}_n \in S} \text{Count}_{\text{match}}(\text{gram}_n)}{\sum_{S \in \text{Reference}} \sum_{\text{gram}_n \in S} \text{Count}(\text{gram}_n)}
    $$

    **Human Evaluation Criteria:**
    1. **Relevance:** Does summary capture key points?
    2. **Coherence:** Is it grammatically correct and fluent?
    3. **Consistency:** Does it contradict the source?
    4. **Fluency:** Is it natural and readable?
    """)
    return


@app.cell
def _(long_text, summary):
    from rouge import Rouge

    def evaluate_summary(reference, summary):
        """
        Calculate ROUGE scores for summary evaluation.

        Args:
            reference: Original text or reference summary
            summary: Generated summary

        Returns:
            Dictionary of ROUGE scores
        """
        rouge = Rouge()
        scores = rouge.get_scores(summary, reference)[0]

        return {
            'ROUGE-1 F1': scores['rouge-1']['f'],
            'ROUGE-2 F1': scores['rouge-2']['f'],
            'ROUGE-L F1': scores['rouge-l']['f']
        }

    # Example evaluation
    reference = long_text.strip()
    generated = summary

    scores = evaluate_summary(reference, generated)
    print("ROUGE Scores:")
    for metric, score in scores.items():
        print(f"{metric}: {score:.3f}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Task 3: Implement Summarization Pipeline

    **Objective:** Create extractive and abstractive summarizers for feedback

    **Steps:**

    **Extractive Summarizer:**
    1. Implement TextRank or LexRank algorithm
    2. Use sentence embeddings for similarity
    3. Extract top N most important sentences
    4. Test on feedback with 5+ sentences
    5. Optimize N based on feedback length

    **Abstractive Prototype:**
    1. Load pre-trained model (BART, T5, or Pegasus)
    2. Configure generation parameters:
       - max_length: 50-100 tokens
       - min_length: 20-30 tokens
       - num_beams: 4
    3. Test on sample feedback
    4. Compare with extractive approach

    **Evaluation:**
    1. Calculate ROUGE scores on validation set
    2. Conduct blind human evaluation (5-10 samples)
    3. Measure inference time per document
    4. Create comparison table

    **Deliverables:**
    - `src/feedback_insights/summarizer.py`
    - Evaluation notebook with ROUGE scores
    - Human evaluation results (3+ reviewers)
    - Saved model artifacts

    **Acceptance:** ROUGE-L > 0.3 and humans rate 70%+ as useful
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## Part 4: Model Comparison & Evaluation

    ### Theory: Rigorous Evaluation

    **Why Compare Models?**
    - Justify computational cost
    - Understand trade-offs
    - Make informed decisions
    - Communicate with stakeholders

    **Comparison Dimensions:**

    | Aspect | Metrics | Considerations |
    |--------|---------|----------------|
    | **Accuracy** | F1, Precision, Recall | Per-class and macro |
    | **Speed** | Inference time, throughput | Real-time requirements |
    | **Resources** | Memory, GPU usage | Deployment constraints |
    | **Robustness** | Performance on edge cases | Out-of-distribution data |
    | **Interpretability** | Feature importance, attention | Trust and debugging |
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Practical Example: Model Comparison

    Let's create a comparison framework:
    """)
    return


@app.cell
def _():
    import pandas as pd
    import time

    def compare_models(models_dict, test_texts, test_labels):
        """
        Compare multiple models on same test set.

        Args:
            models_dict: Dictionary of {model_name: (model, preprocessor)}
            test_texts: List of test texts
            test_labels: List of true labels

        Returns:
            DataFrame with comparison results
        """
        from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

        results = []

        for model_name, (model, preprocessor) in models_dict.items():
            # Time inference
            start_time = time.time()

            # Preprocess and predict
            X_test = preprocessor(test_texts)
            predictions = model.predict(X_test)

            end_time = time.time()

            # Calculate metrics
            accuracy = accuracy_score(test_labels, predictions)
            precision = precision_score(test_labels, predictions, average='macro', zero_division=0)
            recall = recall_score(test_labels, predictions, average='macro', zero_division=0)
            f1 = f1_score(test_labels, predictions, average='macro', zero_division=0)

            inference_time = (end_time - start_time) / len(test_texts) * 1000  # ms per sample

            results.append({
                'Model': model_name,
                'Accuracy': f'{accuracy:.3f}',
                'Precision': f'{precision:.3f}',
                'Recall': f'{recall:.3f}',
                'F1-Score': f'{f1:.3f}',
                'Inference Time (ms)': f'{inference_time:.2f}'
            })

        return pd.DataFrame(results)

    # Example usage note
    print("Model comparison framework ready")
    print("Usage: compare_models(models_dict, test_texts, test_labels)")
    return (pd,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Visualization: Performance Comparison
    """)
    return


@app.cell
def _(pd):
    import plotly.graph_objects as go

    # Example comparison data
    comparison_data = {
        'Model': ['Baseline (TF-IDF + LR)', 'DistilBERT', 'RoBERTa'],
        'Accuracy': [0.72, 0.85, 0.88],
        'F1-Score': [0.70, 0.84, 0.87],
        'Inference Time (ms)': [2.5, 45.0, 78.0]
    }

    comparison_df = pd.DataFrame(comparison_data)

    # Create comparison plot
    fig = go.Figure()

    fig.add_trace(go.Bar(
        name='Accuracy',
        x=comparison_df['Model'],
        y=comparison_df['Accuracy'],
        marker_color='lightblue'
    ))

    fig.add_trace(go.Bar(
        name='F1-Score',
        x=comparison_df['Model'],
        y=comparison_df['F1-Score'],
        marker_color='lightgreen'
    ))

    fig.update_layout(
        title='Model Performance Comparison',
        xaxis_title='Model',
        yaxis_title='Score',
        barmode='group',
        yaxis_range=[0, 1]
    )

    fig
    return comparison_df, go


@app.cell
def _(comparison_df, go):
    # Inference time comparison
    fig_time = go.Figure()

    fig_time.add_trace(go.Bar(
        x=comparison_df['Model'],
        y=comparison_df['Inference Time (ms)'],
        marker_color='coral',
        text=comparison_df['Inference Time (ms)'],
        textposition='auto'
    ))

    fig_time.update_layout(
        title='Inference Time Comparison',
        xaxis_title='Model',
        yaxis_title='Time per Sample (ms)',
        yaxis_type='log'  # Log scale for better visualization
    )

    fig_time
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Error Analysis: Confusion Cases

    Understanding where models fail helps improve them:
    """)
    return


@app.cell
def _():
    def analyze_confusion_cases(y_true, y_pred, texts, class_names):
        """
        Identify and analyze misclassified examples.

        Args:
            y_true: True labels
            y_pred: Predicted labels
            texts: Original texts
            class_names: List of class names

        Returns:
            DataFrame with confusion cases
        """
        import pandas as pd

        misclassified = []

        for i, (true, pred, text) in enumerate(zip(y_true, y_pred, texts)):
            if true != pred:
                misclassified.append({
                    'Text': text,
                    'True Label': class_names[true],
                    'Predicted Label': class_names[pred],
                    'Error Type': f"{class_names[true]} → {class_names[pred]}"
                })

        df = pd.DataFrame(misclassified)

        if len(df) > 0:
            print(f"Total misclassifications: {len(df)}")
            print("\nError distribution:")
            print(df['Error Type'].value_counts())

        return df

    # Example usage
    print("Error analysis function ready")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Task 4: Comprehensive Model Comparison

    **Objective:** Create detailed comparison of baseline vs advanced models

    **Requirements:**

    1. **Performance Metrics:**
       - Accuracy, Precision, Recall, F1 (per class and macro)
       - Confusion matrices side-by-side
       - ROC curves (if applicable)

    2. **Efficiency Metrics:**
       - Inference time per sample
       - Memory usage
       - Model size on disk
       - Training time

    3. **Error Analysis:**
       - Categorize misclassifications
       - Identify common failure patterns
       - Document edge cases
       - Suggest improvements

    4. **Statistical Testing:**
       - McNemar's test for significance
       - Bootstrap confidence intervals
       - Document p-values

    **Deliverables:**
    - Comparison notebook in `artifacts/reports/`
    - Summary table with all metrics
    - Visualization plots
    - Written analysis (2-3 pages)

    **Acceptance:** Advanced model shows statistically significant improvement (p < 0.05)
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## Part 5: Experiment Tracking

    ### Theory: MLOps Best Practices

    **Why Track Experiments?**
    - Reproduce results
    - Compare approaches systematically
    - Share findings with team
    - Audit model development

    **What to Track:**
    - Hyperparameters
    - Metrics (train/val/test)
    - Model artifacts
    - Data versions
    - Code versions (git commit)
    - Environment (Python version, dependencies)

    ### Popular Tools

    | Tool | Pros | Cons |
    |------|------|------|
    | **MLflow** | Open-source, comprehensive | Setup required |
    | **Weights & Biases** | Beautiful UI, cloud | Requires account |
    | **TensorBoard** | Free, PyTorch integrated | Basic features |
    | **Neptune** | Advanced features | Paid plans |
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Practical Example: MLflow Integration
    """)
    return


@app.cell
def _():
    import mlflow
    import mlflow.sklearn

    def log_experiment(model_name, params, metrics, model=None):
        """
        Log experiment to MLflow.

        Args:
            model_name: Name of the experiment
            params: Dictionary of hyperparameters
            metrics: Dictionary of metrics
            model: Model object to save (optional)
        """
        with mlflow.start_run(run_name=model_name):
            # Log parameters
            for param, value in params.items():
                mlflow.log_param(param, value)

            # Log metrics
            for metric, value in metrics.items():
                mlflow.log_metric(metric, value)

            # Log model
            if model is not None:
                mlflow.sklearn.log_model(model, "model")

            print(f"Logged experiment: {model_name}")

    # Example usage
    example_params = {
        'model_type': 'DistilBERT',
        'learning_rate': 2e-5,
        'batch_size': 16,
        'epochs': 3
    }

    example_metrics = {
        'accuracy': 0.85,
        'f1_score': 0.84,
        'train_loss': 0.32
    }

    # Uncomment to log:
    # log_experiment('distilbert_v1', example_params, example_metrics)

    print("MLflow logging function ready")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## Sprint 2 Checklist

    ### Week 3: Advanced Models
    - [ ] Set up transformer training environment
    - [ ] Fine-tune sentiment classifier (DistilBERT/BERT)
    - [ ] Evaluate on validation set
    - [ ] Compare with Sprint 1 baseline
    - [ ] Document improvements and issues
    - [ ] Begin topic modeling exploration
    - [ ] Test BERTopic on sample data

    ### Week 4: Topic Modeling & Summarization
    - [ ] Finalize topic modeling pipeline
    - [ ] Validate topic coherence (3+ reviewers)
    - [ ] Assign meaningful topic labels
    - [ ] Implement extractive summarizer
    - [ ] Build abstractive prototype
    - [ ] Calculate ROUGE scores
    - [ ] Human evaluation of summaries
    - [ ] Create comprehensive comparison notebook
    - [ ] Save all model artifacts
    - [ ] Prepare sprint demo

    ### Sprint 2 Demo Preparation
    - [ ] Show side-by-side baseline vs transformer results
    - [ ] Present topic clusters with representative examples
    - [ ] Demonstrate summarization on 3-5 feedback samples
    - [ ] Display metrics comparison table
    - [ ] Discuss interesting failure cases
    - [ ] Outline Sprint 3 production plan
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## Advanced Tips & Tricks

    ### Transformer Fine-tuning

    **Common Issues:**
    - **Overfitting:** Use dropout, early stopping, smaller learning rate
    - **Catastrophic forgetting:** Freeze early layers, use gradual unfreezing
    - **Class imbalance:** Use class weights, focal loss
    - **Out of memory:** Reduce batch size, use gradient accumulation

    **Optimization Tips:**
    - Use mixed precision training (fp16)
    - Gradient checkpointing for memory
    - Learning rate warmup
    - Cosine annealing schedule

    ### Topic Modeling

    **Improving Results:**
    - Remove very common/rare words
    - Experiment with min_topic_size
    - Try different embedding models
    - Manual topic merging
    - Guided topic modeling with seed words

    ### Summarization

    **Best Practices:**
    - Combine extractive + abstractive
    - Adjust length based on source
    - Post-process for grammar
    - Add source attribution
    - Test with diverse samples
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## Additional Resources

    ### Transformers
    - [Hugging Face Course](https://huggingface.co/course) - Free comprehensive course
    - [BERT Paper](https://arxiv.org/abs/1810.04805) - Original BERT publication
    - [DistilBERT Paper](https://arxiv.org/abs/1910.01108) - Smaller, faster BERT

    ### Topic Modeling
    - [BERTopic Documentation](https://maartengr.github.io/BERTopic/) - Official docs
    - [Topic Modeling Guide](https://www.machinelearningplus.com/nlp/topic-modeling-gensim-python/) - Comprehensive tutorial

    ### Summarization
    - [The Annotated Transformer](https://nlp.seas.harvard.edu/2018/04/03/attention.html) - Understanding attention
    - [BART Paper](https://arxiv.org/abs/1910.13461) - Denoising sequence-to-sequence
    - [Pegasus Paper](https://arxiv.org/abs/1912.08777) - State-of-art summarization

    ### MLOps
    - [MLflow Documentation](https://mlflow.org/docs/latest/index.html)
    - [Weights & Biases Tutorials](https://wandb.ai/site/tutorials)
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## Next Steps

    After completing Sprint 2:

    1. **Review Achievements:**
       - Sentiment model improvement percentage
       - Number of coherent topics discovered
       - Summary quality metrics (ROUGE scores)

    2. **Gather Feedback:**
       - Stakeholder demo
       - Team retrospective
       - User testing (if available)

    3. **Document Learnings:**
       - Update technical documentation
       - Record hyperparameter choices
       - Document failure cases

    4. **Prepare for Sprint 3:**
       - Plan API endpoints
       - Design dashboard mockups
       - Set up Docker environment
       - Review production requirements

    **Ready for Sprint 3?** Check out `03_sprint_03_prod_ui.py` for production deployment and UI development!

    ---

    Excellent work on Sprint 2! You're building a robust NLP pipeline for customer feedback analysis. Keep pushing forward!
    """)
    return


if __name__ == "__main__":
    app.run()
