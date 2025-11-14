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

    ## Sprint Goals

    By the end of Sprint 2, you will have:
    - Advanced sentiment analysis with ensemble models (VADER + TF-IDF)
    - Topic modeling pipeline with coherent clusters (Gensim LDA)
    - Extractive summarization using multiple algorithms (TextRank, LexRank, LSA)
    - Comprehensive comparison with baseline models
    - Human-validated topic coherence

    **Timeline:** Weeks 3-4 | **Demo:** End of Week 4

    **Note:** This notebook uses lightweight libraries (VADER, Gensim, Sumy) for production-ready NLP without heavy dependencies.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## Sprint 2 Deliverables

    ### Required Outputs

    1. **Advanced Sentiment Pipeline** - Ensemble model combining VADER + TF-IDF
    2. **Topic Modeling Pipeline** - Gensim LDA with interpretable topics
    3. **Extractive Summarizers** - Multiple algorithms (TextRank, LexRank, LSA)
    4. **Algorithm Comparison** - Performance analysis across methods
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
    - **TF-IDF:** Ignores word order and context
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
    ### Practical Example: VADER Sentiment Analysis
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
    return df, np, pd


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
    return (train_texts,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### VADER in Action

    Let's analyze some feedback with VADER:
    """)
    return


@app.cell
def _(train_texts):
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

    # Initialize VADER
    vader = SentimentIntensityAnalyzer()

    # Analyze sample texts
    print("VADER Sentiment Analysis Examples:\n")

    for i, text in enumerate(train_texts[:5]):
        scores = vader.polarity_scores(text)

        # Determine sentiment from compound score
        if scores['compound'] >= 0.05:
            sentiment = 'POSITIVE'
        elif scores['compound'] <= -0.05:
            sentiment = 'NEGATIVE'
        else:
            sentiment = 'NEUTRAL'

        print(f"{i+1}. Text: {text}")
        print(f"   Compound: {scores['compound']:.3f} → {sentiment}")
        print(f"   Details: pos={scores['pos']:.2f}, neu={scores['neu']:.2f}, neg={scores['neg']:.2f}\n")
    return (vader,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Ensemble Approach: Combining Multiple Methods

    Instead of relying on a single method, we can combine VADER with our baseline TF-IDF model:
    """)
    return


@app.cell
def _(df, vader):
    def _():
        import pandas as pd
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import classification_report, accuracy_score

        # Get VADER scores for all texts
        vader_scores = []
        for text in df['text']:
            scores = vader.polarity_scores(text)
            vader_scores.append([
                scores['compound'],
                scores['pos'],
                scores['neu'],
                scores['neg']
            ])

        # Combine with TF-IDF features
        vectorizer = TfidfVectorizer(max_features=50, ngram_range=(1, 2))
        tfidf_features = vectorizer.fit_transform(df['text']).toarray()

        # Stack features
        import numpy as np
        X_combined = np.hstack([tfidf_features, np.array(vader_scores)])
        y = df['label'].values

        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X_combined, y, test_size=0.3, random_state=42, stratify=y
        )

        # Train ensemble model
        ensemble_model = LogisticRegression(max_iter=1000, random_state=42)
        ensemble_model.fit(X_train, y_train)

        # Predictions
        y_pred = ensemble_model.predict(X_test)

        # Evaluation
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Ensemble Model Accuracy: {accuracy:.2%}")
        print(f"\nClassification Report:")
        return print(classification_report(y_test, y_pred, target_names=['negative', 'neutral', 'positive']))


    _()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Task 1: Build Advanced Sentiment Classifier

    **Objective:** Create an improved sentiment classifier using ensemble methods

    **Steps:**
    1. Implement VADER sentiment analysis
    2. Extract TF-IDF features from text
    3. Combine VADER scores with TF-IDF features
    4. Train ensemble classifier (Logistic Regression or Random Forest)
    5. Compare with Sprint 1 baseline
    6. Analyze feature importance

    **Alternative Approaches:**
    - TextBlob for subjectivity analysis
    - Domain-specific sentiment lexicons
    - Weighted ensemble voting
    - Stacking different classifiers

    **Acceptance:** Model achieves >5% improvement over baseline
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## Part 2: Topic Modeling with Gensim LDA

    ### Theory: Latent Dirichlet Allocation

    **LDA discovers hidden topics by:**
    1. Assuming each document is a mixture of topics
    2. Each topic is a distribution over words
    3. Using probabilistic inference to find topics

    **Key Parameters:**
    - `num_topics`: Number of topics to discover
    - `alpha`: Document-topic density (lower = fewer topics per doc)
    - `beta`: Topic-word density (lower = fewer words per topic)
    - `passes`: Number of training iterations

    **Preprocessing for LDA:**
    - Remove stop words
    - Lemmatization
    - Create dictionary and corpus
    - Filter extremes (very rare/common words)
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Task 1: Build Advanced Sentiment Classifier

    **Objective:** Create an improved sentiment classifier using ensemble methods and advanced features

    **Approach 1: VADER + Machine Learning Ensemble**
    1. Extract VADER sentiment scores (compound, pos, neg, neu)
    2. Extract TF-IDF features (unigrams + bigrams)
    3. Extract additional features (text length, punctuation count, capitalization)
    4. Combine all features
    5. Train Random Forest or Gradient Boosting classifier
    6. Compare with Sprint 1 baseline

    **Approach 2: Advanced Feature Engineering**
    1. Character n-grams for misspellings
    2. Word embeddings (using pre-trained Word2Vec or GloVe)
    3. POS tagging features
    4. Named entity features

    **Acceptance:** Model achieves >10% improvement over baseline
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

    ### Popular Approaches (Production-Ready)

    | Method | Approach | Pros | Cons |
    |--------|----------|------|------|
    | **LDA (Gensim)** | Probabilistic | Interpretable, fast, lightweight | Needs preprocessing, fixed topics |
    | **NMF** | Matrix factorization | Fast, sparse, scikit-learn | Linear only |
    | **LSA** | SVD-based | Very fast, simple | Less interpretable |

    **Note:** Advanced methods like BERTopic require PyTorch/transformers. For production deployment on Windows or resource-constrained environments, Gensim LDA provides excellent results with minimal dependencies.

    ### Gensim LDA Pipeline

    1. **Tokenize documents** - Split into words, remove stopwords
    2. **Create dictionary** - Map words to IDs
    3. **Build corpus** - Bag-of-words representation
    4. **Train LDA model** - Extract latent topics with Dirichlet priors
    5. **Evaluate coherence** - Validate semantic quality
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Practical Example: Topic Modeling with Gensim LDA
    """)
    return


@app.cell
def _(df):
    # Use Gensim for topic modeling (lightweight, production-ready)
    from gensim import corpora
    from gensim.models import LdaModel
    from gensim.parsing.preprocessing import STOPWORDS
    import nltk

    # Download required NLTK data (updated for NLTK 3.9+)
    try:
        nltk.data.find('tokenizers/punkt_tab')
    except LookupError:
        nltk.download('punkt_tab', quiet=True)

    # Prepare documents
    documents = df['text'].tolist()

    # Tokenize and remove stopwords
    def preprocess(text):
        tokens = nltk.word_tokenize(text.lower())
        return [token for token in tokens if token.isalnum() and token not in STOPWORDS and len(token) > 2]

    processed_docs = [preprocess(doc) for doc in documents]

    # Create dictionary and corpus
    dictionary = corpora.Dictionary(processed_docs)
    dictionary.filter_extremes(no_below=1, no_above=0.8)
    corpus = [dictionary.doc2bow(doc) for doc in processed_docs]

    # Train LDA model
    num_topics = 3
    lda_model = LdaModel(
        corpus=corpus,
        id2word=dictionary,
        num_topics=num_topics,
        random_state=42,
        passes=10,
        alpha='auto'
    )

    print(f"LDA model trained with {num_topics} topics")
    return (
        corpus,
        dictionary,
        documents,
        lda_model,
        nltk,
        num_topics,
        processed_docs,
    )


@app.cell
def _(lda_model, num_topics):
    # Display topics
    print("Discovered Topics:\n")
    for idx in range(num_topics):
        print(f"Topic {idx}:")
        words = lda_model.show_topic(idx, topn=10)
        print(f"  Top words: {', '.join([word for word, prob in words])}\n")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Topic Visualization

    Let's create visualizations for our topics:
    """)
    return


@app.cell
def _(corpus, documents, lda_model, pd):
    # Get dominant topic for each document
    topic_assignments = []
    for j, doc_bow in enumerate(corpus):
        topic_dist = lda_model.get_document_topics(doc_bow)
        if topic_dist:
            dominant_topic = max(topic_dist, key=lambda x: x[1])[0]
            topic_assignments.append(dominant_topic)
        else:
            topic_assignments.append(-1)

        # Create summary dataframe
    topic_summary = pd.DataFrame({
        'Document': documents,
        'Topic': topic_assignments
    })

    topic_counts = topic_summary['Topic'].value_counts().sort_index()
    print("Documents per topic:")
    print(topic_counts)

    return (topic_counts,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Topic Distribution Visualization

    Visualize how documents are distributed across topics:
    """)
    return


@app.cell
def _(topic_counts):
    import plotly.graph_objects as go

    # Create bar chart for topic distribution
    fig_topic_dist = go.Figure(data=[
        go.Bar(
            x=[f"Topic {i}" for i in topic_counts.index],
            y=topic_counts.values,
            marker_color='lightblue'
        )
    ])

    fig_topic_dist.update_layout(
        title="Document Distribution Across Topics",
        xaxis_title="Topic",
        yaxis_title="Number of Documents",
        height=400
    )

    fig_topic_dist
    return (go,)


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
    - **Topic coherence (C_v):** Measures semantic similarity of top words (can use gensim.models.CoherenceModel)
    - **Topic diversity:** Percentage of unique words across topics
    - **Perplexity:** For LDA models (lower is better)
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Computing Topic Coherence

    Let's calculate coherence score for our LDA model:
    """)
    return


@app.cell
def _(dictionary, lda_model, processed_docs):
    from gensim.models import CoherenceModel

    # Calculate coherence score
    coherence_model = CoherenceModel(
        model=lda_model,
        texts=processed_docs,
        dictionary=dictionary,
        coherence='c_v'
    )

    coherence_score = coherence_model.get_coherence()
    print(f"Topic Coherence Score (C_v): {coherence_score:.4f}")
    print(f"Interpretation: {'Good' if coherence_score > 0.4 else 'Needs improvement'}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Task 2: Build Topic Modeling Pipeline

    **Objective:** Create interpretable topic clusters from customer feedback

    **Steps:**
    1. Use Gensim LDA for lightweight, production-ready topic modeling
    2. Prepare data:
       - Clean and preprocess text
       - Remove very short documents (< 3 words)
       - Tokenize and remove stopwords
    3. Configure parameters:
       - `num_topics`: Number of topics to extract
       - `passes`: Number of training iterations
       - `alpha`: Document-topic density ('auto' for learned parameter)
    4. Train model and extract topics
    5. Validate coherence with team members and coherence metrics
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
    - Faster and more reliable (production-ready)
    - Lower computational requirements
    - Good for: Long documents, factual content, compliance/legal

    **Abstractive Summarization:**
    - Generates new sentences (paraphrasing)
    - More human-like and concise
    - Requires large models or API services
    - Higher risk of hallucinations
    - Good for: Creative summaries, API-based workflows

    ### Popular Extractive Algorithms (Lightweight & Production-Ready)

    | Method | Approach | Complexity | Speed | Quality |
    |--------|----------|------------|-------|---------|
    | **TextRank** | Graph-based (PageRank) | Low | Fast | Good |
    | **LexRank** | Graph + Lexical Similarity | Medium | Fast | Good |
    | **LSA** | Latent Semantic Analysis | Medium | Fast | Decent |
    | **Luhn** | Frequency-based | Very Low | Very Fast | Fair |

    **For this course:** We focus on extractive methods using Sumy library (no PyTorch dependencies).
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Practical Example: Extractive Summarization with TextRank (TF-IDF based)
    """)
    return


@app.cell
def _(nltk, np):
    from sklearn.metrics.pairwise import cosine_similarity
    from sklearn.feature_extraction.text import TfidfVectorizer

    # Download required NLTK data (updated for NLTK 3.9+)
    try:
        nltk.data.find('tokenizers/punkt_tab')
    except LookupError:
        nltk.download('punkt_tab', quiet=True)

    def textrank_summarize(text, num_sentences=2, embedding_model=None):
        """
        Extractive summarization using TextRank algorithm with TF-IDF.

        Args:
            text: Input text to summarize
            num_sentences: Number of sentences to extract

        Returns:
            Summary string
        """
        # Split into sentences
        sentences = nltk.sent_tokenize(text)

        if len(sentences) <= num_sentences:
            return text

        # Use TF-IDF for sentence embeddings
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(sentences)

        # Calculate similarity matrix
        similarity_matrix = cosine_similarity(tfidf_matrix)

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
    ### Practical Example: Extractive Summarization with Sumy

    Using lightweight Sumy library for extractive summarization:
    """)
    return


@app.cell
def _(long_text):
    from sumy.parsers.plaintext import PlaintextParser
    from sumy.nlp.tokenizers import Tokenizer
    from sumy.summarizers.lsa import LsaSummarizer
    from sumy.summarizers.lex_rank import LexRankSummarizer
    from sumy.summarizers.text_rank import TextRankSummarizer

    # Parse the text
    parser = PlaintextParser.from_string(long_text.strip(), Tokenizer("english"))

    # Try multiple extractive algorithms
    sentence_count = 2  # Number of sentences in summary

    # LSA (Latent Semantic Analysis) summarization
    lsa_summarizer = LsaSummarizer()
    lsa_summary = lsa_summarizer(parser.document, sentence_count)

    # LexRank summarization
    lexrank_summarizer = LexRankSummarizer()
    lexrank_summary = lexrank_summarizer(parser.document, sentence_count)

    # TextRank summarization
    textrank_summarizer = TextRankSummarizer()
    textrank_summary = textrank_summarizer(parser.document, sentence_count)

    print("Original text:")
    print(long_text.strip())

    print("\n--- LSA Summary ---")
    for sentence in lsa_summary:
        print(sentence)

    print("\n--- LexRank Summary ---")
    for sentence in lexrank_summary:
        print(sentence)

    print("\n--- TextRank Summary ---")
    for sentence in textrank_summary:
        print(sentence)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Summarization Quality Metrics

    **ROUGE Scores (Recall-Oriented Understudy for Gisting Evaluation):**

    - **ROUGE-1:** Overlap of unigrams (individual words)
    - **ROUGE-2:** Overlap of bigrams (word pairs)
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

    scores_ev = evaluate_summary(reference, generated)
    print("ROUGE Scores:")
    for metric, score in scores_ev.items():
        print(f"{metric}: {score:.3f}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Task 3: Implement Summarization Pipeline

    **Objective:** Create extractive summarizers for feedback using lightweight methods

    **Steps:**

    **Extractive Summarizer (Primary Approach):**
    1. Implement multiple algorithms (TextRank, LexRank, LSA) using Sumy
    2. Use sentence similarity for ranking important sentences
    3. Extract top N most important sentences
    4. Test on feedback with 5+ sentences
    5. Optimize N based on feedback length
    6. Compare algorithm performance on your dataset

    **Algorithm Selection:**
    - **TextRank:** Graph-based, good for general text
    - **LexRank:** Graph-based with lexical similarity, robust for varied domains
    - **LSA:** Uses SVD to find latent topics, good for technical content

    **Evaluation:**
    1. Calculate ROUGE scores on validation set
    2. Conduct blind human evaluation (5-10 samples)
    3. Measure inference time per document (should be <0.1s per doc)
    4. Create comparison table across algorithms
    5. Choose best algorithm based on ROUGE + human preference

    **Deliverables:**
    - `src/feedback_insights/summarizer.py` with all three algorithms
    - Evaluation notebook with ROUGE scores
    - Human evaluation results (3+ reviewers)
    - Algorithm comparison visualization

    **Acceptance:** ROUGE-L > 0.3 and humans rate 70%+ as useful

    **Note:** Extractive methods are production-ready, fast, and interpretable. For abstractive summarization in production, consider API-based solutions (OpenAI, Cohere) rather than hosting large models.
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
def _(pd):
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
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Visualization: Performance Comparison
    """)
    return


@app.cell
def _(go, pd):
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
    return (comparison_df,)


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
    - [ ] Set up lightweight NLP environment (VADER, Gensim, Sumy)
    - [ ] Build ensemble sentiment classifier (VADER + TF-IDF)
    - [ ] Evaluate on validation set
    - [ ] Compare with Sprint 1 baseline
    - [ ] Document improvements and feature engineering
    - [ ] Begin topic modeling exploration
    - [ ] Test Gensim LDA on sample data

    ### Week 4: Topic Modeling & Summarization
    - [ ] Finalize Gensim LDA pipeline
    - [ ] Validate topic coherence (3+ reviewers)
    - [ ] Compute coherence metrics (C_v score)
    - [ ] Assign meaningful topic labels
    - [ ] Implement extractive summarizers (TextRank, LexRank, LSA)
    - [ ] Compare summarization algorithms
    - [ ] Calculate ROUGE scores
    - [ ] Human evaluation of summaries
    - [ ] Create comprehensive comparison notebook
    - [ ] Save all model artifacts
    - [ ] Prepare sprint demo

    ### Sprint 2 Demo Preparation
    - [ ] Show side-by-side baseline vs ensemble results
    - [ ] Present topic clusters with representative examples
    - [ ] Demonstrate all 3 summarization algorithms on sample feedback
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

    ### Sentiment Analysis
    - [VADER Sentiment](https://github.com/cjhutto/vaderSentiment) - Official documentation
    - [TextBlob Tutorial](https://textblob.readthedocs.io/) - Simple sentiment API

    ### Topic Modeling
    - [Gensim Documentation](https://radimrehurek.com/gensim/) - Official docs
    - [Topic Modeling Guide](https://www.machinelearningplus.com/nlp/topic-modeling-gensim-python/) - Comprehensive tutorial
    - [LDA Explained](https://towardsdatascience.com/light-on-math-machine-learning-intuitive-guide-to-latent-dirichlet-allocation-437c81220158) - Math intuition

    ### Summarization
    - [Sumy Documentation](https://github.com/miso-belica/sumy) - Extractive algorithms
    - [TextRank Paper](https://web.eecs.umich.edu/~mihalcea/papers/mihalcea.emnlp04.pdf) - Original algorithm
    - [ROUGE Metrics](https://github.com/google-research/google-research/tree/master/rouge) - Evaluation

    ### MLOps
    - [MLflow Documentation](https://mlflow.org/docs/latest/index.html)
    - [Weights & Biases Tutorials](https://wandb.ai/site/tutorials)

    ### Advanced (Optional)
    - [Hugging Face Course](https://huggingface.co/course) - For transformer learning
    - [BERT Paper](https://arxiv.org/abs/1810.04805) - Foundational reading
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
