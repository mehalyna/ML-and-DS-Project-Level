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
    # Sprint 1: Data & Baseline (Weeks 1-2)

    Welcome to **Sprint 1** of the Customer Feedback Insight Platform! This sprint focuses on establishing the foundation of your project through data ingestion, exploratory data analysis (EDA), and building a baseline sentiment classifier.

    ## Sprint Goals

    By the end of Sprint 1, you will have:
    - Reproducible data ingestion pipeline
    - Clean, preprocessed dataset
    - Comprehensive EDA with visualizations
    - Baseline sentiment classification model
    - Initial evaluation metrics and error analysis

    **Timeline:** Weeks 1-2 | **Demo:** End of Week 2
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## Sprint 1 Deliverables

    ### Required Outputs

    1. **Ingestion Scripts** - CSV/JSON readers with validation
    2. **Cleaned Dataset Snapshots** - Stored in `data/processed/`
    3. **EDA Notebook** - Key charts and data quality notes
    4. **Baseline Sentiment Classifier** - scikit-learn or small transformer
    5. **Evaluation Metrics** - Reproducible metrics table
    6. **Error Analysis** - Short report on model mistakes

    ### Acceptance Criteria

    - Ingestion reproducible via script
    - Baseline model trains within course hardware limits
    - Produces reproducible metrics table
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## Part 1: Data Sources & Schema Design

    ### Theory: Choosing the Right Data Sources

    For customer feedback analysis, you need diverse, representative data. Public datasets provide excellent starting points:

    **Recommended Sources:**
    - **Kaggle:** "Women's E-Commerce Clothing Reviews", "Amazon Product Reviews"
    - **Yelp Open Dataset:** Restaurant and business reviews
    - **Twitter/Reddit:** Social media sentiment (via APIs)
    - **Simulated Data:** CRM notes, support chat logs

    ### Required Schema Fields

    Each feedback record should contain:

    | Field | Type | Required | Description |
    |-------|------|----------|-------------|
    | `source` | string | Yes | Origin channel (email, chat, survey, etc.) |
    | `timestamp` | datetime | Yes | When feedback was received |
    | `customer_id` | string | Yes | Anonymized customer identifier |
    | `text` | string | Yes | The actual feedback content |
    | `product_id` | string | No | Associated product/service |
    | `rating` | int | No | Numerical rating (1-5) if available |
    | `sentiment` | string | No | Label for supervised learning (positive/negative/neutral) |
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Practical Example: Sample Dataset Structure

    Let's create a sample dataset to understand the structure:
    """)
    return


@app.cell
def _():
    import pandas as pd
    from datetime import datetime, timedelta
    import random

    # Create a larger sample dataset for better demonstration
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
        'Perfect! Just what I was looking for.',
        'Outstanding quality and fast shipping.'
    ]

    negative_texts = [
        'The item arrived damaged. Very disappointed.',
        'Worst purchase ever. Do not recommend.',
        'The product description was misleading. Expected better quality.',
        'Terrible quality. Requesting a refund.',
        'Poor customer service and defective product.',
        'Waste of money. Very unsatisfied.',
        'Product broke after one use. Disappointing.',
        'Not as described. Very frustrating experience.',
        'Horrible. Would give zero stars if possible.',
        'Do not buy this. Complete waste.'
    ]

    neutral_texts = [
        'Average experience. Product is okay but customer service could be better.',
        'It is fine. Nothing special but does the job.',
        'Decent product for the price. Could be improved.',
        'Okay purchase. Met basic expectations.',
        'Acceptable quality. Not great but not terrible.',
        'Standard product. Nothing to complain about.',
        'Fair value. Works as expected.',
        'Mediocre experience overall. Could be better.',
        'It is alright. Does what it should.',
        'Average quality. Gets the job done.'
    ]

    # Generate 30 samples (10 per sentiment)
    sample_data = {
        'source': [],
        'timestamp': [],
        'customer_id': [],
        'text': [],
        'product_id': [],
        'rating': [],
        'sentiment': []
    }

    sources = ['email', 'chat', 'survey', 'social']
    products = ['PROD_A', 'PROD_B', 'PROD_C', 'PROD_D']

    idx = 0
    for sentiment_type, texts, ratings in [
        ('positive', positive_texts, [4, 5]),
        ('negative', negative_texts, [1, 2]),
        ('neutral', neutral_texts, [3])
    ]:
        for text in texts:
            sample_data['source'].append(random.choice(sources))
            sample_data['timestamp'].append(datetime.now() - timedelta(days=idx))
            sample_data['customer_id'].append(f'CUST_{1000+idx}')
            sample_data['text'].append(text)
            sample_data['product_id'].append(random.choice(products))
            sample_data['rating'].append(random.choice(ratings))
            sample_data['sentiment'].append(sentiment_type)
            idx += 1

    sample_df = pd.DataFrame(sample_data)
    return pd, sample_df


@app.cell
def _(mo, sample_df):
    mo.ui.table(sample_df, selection=None)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## Part 2: Data Ingestion Pipeline

    ### Theory: Building Robust Ingestion

    A good ingestion pipeline should:
    1. **Validate** - Check schema and data types
    2. **Handle Errors** - Gracefully skip or log bad records
    3. **Be Reproducible** - Same input → same output
    4. **Document** - Log what was done and any issues

    ### Design Pattern: ETL (Extract, Transform, Load)

    ```
    Raw Data → Validation → Cleaning → Standardization → Storage
    ```

    ### Key Considerations

    - **Encoding:** UTF-8 for international characters
    - **Missing Values:** Define strategy (drop, impute, flag)
    - **Duplicates:** Identify and handle
    - **Timestamps:** Parse and standardize timezone
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Practical Example: Ingestion Function

    Here's a basic ingestion function that validates and cleans customer feedback:
    """)
    return


@app.cell
def _(pd):
    def ingest_feedback_csv(filepath, required_columns=None):
        """
        Ingest customer feedback from CSV with validation.

        Args:
            filepath: Path to CSV file
            required_columns: List of required column names

        Returns:
            DataFrame with validated and cleaned data
        """
        if required_columns is None:
            required_columns = ['source', 'timestamp', 'customer_id', 'text']

        # Read CSV
        df = pd.read_csv(filepath)

        # Validate required columns
        missing_cols = set(required_columns) - set(df.columns)
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

        # Convert timestamp to datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')

        # Remove rows with missing critical fields
        df = df.dropna(subset=required_columns)

        # Remove duplicate entries based on customer_id and text
        df = df.drop_duplicates(subset=['customer_id', 'text'])

        # Basic text cleaning
        df['text'] = df['text'].str.strip()
        df = df[df['text'].str.len() > 0]  # Remove empty text

        # Sort by timestamp
        df = df.sort_values('timestamp').reset_index(drop=True)

        return df

    # Example usage with sample data
    # Save sample data to demonstrate
    # sample_df.to_csv('data/raw/sample_feedback.csv', index=False)
    # cleaned_df = ingest_feedback_csv('data/raw/sample_feedback.csv')
    return (ingest_feedback_csv,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Task 1: Create Your Ingestion Script

    **Objective:** Build a script that reads CSV/JSON and writes cleaned CSV

    **Steps:**
    1. Create `src/feedback_insights/ingestion.py`
    2. Implement validation logic
    3. Add error handling and logging
    4. Test with sample data
    5. Save to `data/processed/clean.csv`

    **Acceptance:** Script runs and produces `data/processed/clean.csv` with validated records
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## Part 3: Exploratory Data Analysis (EDA)

    ### Theory: Why EDA Matters

    EDA helps you understand:
    - **Data Distribution** - How values are spread
    - **Relationships** - Correlations between variables
    - **Anomalies** - Outliers and data quality issues
    - **Patterns** - Trends and seasonality

    ### Essential EDA Steps

    1. **Descriptive Statistics** - Mean, median, std, quartiles
    2. **Missing Data Analysis** - Where and how much
    3. **Distribution Plots** - Histograms, box plots
    4. **Text Statistics** - Length, word count, vocabulary
    5. **Temporal Analysis** - Trends over time
    6. **Correlation Analysis** - Feature relationships
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Practical Example: Basic EDA

    Let's perform some essential EDA on our sample dataset:
    """)
    return


@app.cell
def _(sample_df):
    # Basic statistics
    basic_stats = {
        'Total Records': len(sample_df),
        'Unique Customers': sample_df['customer_id'].nunique(),
        'Unique Products': sample_df['product_id'].nunique(),
        'Date Range': f"{sample_df['timestamp'].min().date()} to {sample_df['timestamp'].max().date()}",
        'Avg Text Length': sample_df['text'].str.len().mean(),
        'Missing Values': sample_df.isnull().sum().sum()
    }

    basic_stats
    return


@app.cell
def _(pd, sample_df):
    # Sentiment distribution
    sentiment_counts = sample_df['sentiment'].value_counts()
    sentiment_df = pd.DataFrame({
        'Sentiment': sentiment_counts.index,
        'Count': sentiment_counts.values,
        'Percentage': (sentiment_counts.values / len(sample_df) * 100).round(2)
    })
    sentiment_df
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Visualization Examples

    Here are some key visualizations for feedback data:
    """)
    return


@app.cell
def _(sample_df):
    import plotly.express as px

    # Sentiment distribution pie chart
    fig_sentiment = px.pie(
        sample_df, 
        names='sentiment', 
        title='Sentiment Distribution',
        color='sentiment',
        color_discrete_map={'positive': 'green', 'negative': 'red', 'neutral': 'gray'}
    )
    fig_sentiment
    return (px,)


@app.cell
def _(px, sample_df):
    # Rating distribution by product
    fig_rating = px.bar(
        sample_df,
        x='product_id',
        y='rating',
        color='sentiment',
        title='Average Rating by Product and Sentiment',
        barmode='group'
    )
    fig_rating
    return


@app.cell
def _(px, sample_df):
    # Text length distribution
    text_lengths = sample_df['text'].str.len()
    fig_length = px.histogram(
        x=text_lengths,
        nbins=20,
        title='Distribution of Feedback Text Length',
        labels={'x': 'Text Length (characters)', 'y': 'Count'}
    )
    fig_length
    return


@app.cell
def _(px, sample_df):
    # Feedback over time
    fig_timeline = px.scatter(
        sample_df,
        x='timestamp',
        y='rating',
        color='sentiment',
        size='rating',
        title='Feedback Timeline',
        labels={'timestamp': 'Date', 'rating': 'Rating'}
    )
    fig_timeline
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Task 2: Comprehensive EDA

    **Objective:** Explore dataset and produce 10-15 EDA charts

    **Required Visualizations:**
    1. Sentiment distribution (pie/bar chart)
    2. Rating distribution (histogram)
    3. Text length distribution
    4. Feedback volume over time (line chart)
    5. Top products by feedback count
    6. Average rating by product
    7. Source channel distribution
    8. Word cloud for positive feedback
    9. Word cloud for negative feedback
    10. Correlation heatmap (numeric features)
    11. Missing data visualization
    12. Outlier detection (box plots)

    **Acceptance:** Notebook committed with key findings and data quality notes
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## Part 4: Text Preprocessing

    ### Theory: NLP Preprocessing Pipeline

    Before modeling, text needs cleaning and standardization:

    **Common Preprocessing Steps:**
    1. **Lowercasing** - Standardize case
    2. **Remove Special Characters** - Keep only letters/numbers
    3. **Tokenization** - Split into words
    4. **Remove Stop Words** - Filter common words (the, is, at)
    5. **Lemmatization/Stemming** - Reduce to root form
    6. **Remove Noise** - URLs, emails, extra whitespace

    ### When to Apply Each Step

    - **Always:** Lowercasing, remove extra whitespace
    - **Usually:** Remove special characters, tokenization
    - **Sometimes:** Stop word removal (not for sentiment!)
    - **Depends:** Lemmatization (slower but more accurate)
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Practical Example: Text Preprocessing
    """)
    return


@app.cell
def _():
    import re
    from typing import List

    def preprocess_text(text: str, remove_stopwords: bool = False) -> str:
        """
        Clean and preprocess text for NLP tasks.

        Args:
            text: Raw text string
            remove_stopwords: Whether to remove stop words

        Returns:
            Cleaned text string
        """
        # Lowercase
        text = text.lower()

        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text)

        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)

        # Remove special characters but keep spaces
        text = re.sub(r'[^a-zA-Z0-9\s]', '', text)

        # Remove extra whitespace
        text = ' '.join(text.split())

        # Optional: Remove stop words (basic example)
        if remove_stopwords:
            stop_words = {'the', 'is', 'at', 'which', 'on', 'a', 'an', 'and', 'or', 'but'}
            words = text.split()
            text = ' '.join([w for w in words if w not in stop_words])

        return text

    # Test the function
    test_text = "Great product! Visit our website at https://example.com for more info."
    cleaned_text = preprocess_text(test_text)
    print(f"Original: {test_text}")
    print(f"Cleaned: {cleaned_text}")
    return (preprocess_text,)


@app.cell
def _(preprocess_text, sample_df):
    # Apply preprocessing to sample data
    sample_df['text_cleaned'] = sample_df['text'].apply(preprocess_text)
    sample_df[['text', 'text_cleaned']].head()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## Part 5: Baseline Sentiment Classification

    ### Theory: Starting Simple

    **Why Baseline Models?**
    - Quick to implement and train
    - Establish performance floor
    - Help understand data complexity
    - Guide feature engineering

    ### Baseline Model Options

    | Model | Pros | Cons | Use When |
    |-------|------|------|----------|
    | **Logistic Regression** | Fast, interpretable | Linear only | Small datasets |
    | **Naive Bayes** | Very fast, works well with text | Independence assumption | Quick prototyping |
    | **Random Forest** | Robust, handles non-linearity | Slower, less interpretable | Medium datasets |
    | **Small BERT** | Better accuracy | Slower, needs GPU | More compute available |

    ### Feature Extraction

    - **TF-IDF (Term Frequency-Inverse Document Frequency)** - Classic approach
    - **Count Vectorizer** - Simple word counts
    - **Word Embeddings** - Pre-trained (Word2Vec, GloVe)
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Practical Example: TF-IDF + Logistic Regression

    Let's build a simple baseline classifier:
    """)
    return


@app.cell
def _(sample_df):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

    # Prepare data
    X = sample_df['text_cleaned']
    y = sample_df['sentiment']

    # Split data (for demo purposes, using small sample)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    # Feature extraction with TF-IDF
    vectorizer = TfidfVectorizer(
        max_features=100,  # Limit features for small dataset
        ngram_range=(1, 2),  # Unigrams and bigrams
        min_df=1
    )

    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    # Train baseline model
    baseline_model = LogisticRegression(max_iter=1000, random_state=42)
    baseline_model.fit(X_train_tfidf, y_train)

    # Predictions
    y_pred = baseline_model.predict(X_test_tfidf)

    # Evaluation
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Baseline Accuracy: {accuracy:.2%}")
    return (
        X_test,
        accuracy,
        baseline_model,
        classification_report,
        confusion_matrix,
        vectorizer,
        y_pred,
        y_test,
    )


@app.cell
def _(classification_report, y_pred, y_test):
    # Detailed classification report
    report = classification_report(y_test, y_pred, output_dict=True)
    print(classification_report(y_test, y_pred))
    return


@app.cell
def _(confusion_matrix, px, y_pred, y_test):
    # Confusion matrix visualization
    cm = confusion_matrix(y_test, y_pred)
    labels = sorted(y_test.unique())

    fig_cm = px.imshow(
        cm,
        x=labels,
        y=labels,
        labels=dict(x="Predicted", y="Actual", color="Count"),
        title="Confusion Matrix - Baseline Model",
        text_auto=True,
        color_continuous_scale='Blues'
    )
    fig_cm
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Model Interpretation

    Let's examine which words are most important for each sentiment:
    """)
    return


@app.cell
def _(baseline_model, pd, vectorizer):
    def _():
        # Get feature importance (coefficients)
        feature_names = vectorizer.get_feature_names_out()

        # For multi-class, show top features per class
        importance_data = []
        for idx, sentiment in enumerate(baseline_model.classes_):
            coef = baseline_model.coef_[idx]
            top_indices = coef.argsort()[-10:][::-1]  # Top 10

            for i in top_indices:
                importance_data.append({
                    'Sentiment': sentiment,
                    'Feature': feature_names[i],
                    'Coefficient': coef[i]
                })

        importance_df = pd.DataFrame(importance_data)
        return importance_df


    _()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Task 3: Build & Evaluate Baseline Model

    **Objective:** Implement baseline sentiment model and evaluation

    **Requirements:**
    1. Create `src/feedback_insights/models.py`
    2. Implement at least one baseline model (Logistic Regression or Naive Bayes)
    3. Use TF-IDF or Count Vectorizer for features
    4. Perform train/validation/test split (60/20/20)
    5. Calculate metrics: Accuracy, Precision, Recall, F1-score
    6. Create confusion matrix visualization
    7. Perform error analysis on misclassified examples
    8. Save model artifacts to `models/checkpoints/`

    **Acceptance:** Notebook and saved model artifact with reproducible metrics
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## Part 6: Model Evaluation & Error Analysis

    ### Theory: Beyond Accuracy

    **Key Metrics for Classification:**

    - **Accuracy** = $\frac{TP + TN}{TP + TN + FP + FN}$ - Overall correctness
    - **Precision** = $\frac{TP}{TP + FP}$ - How many predicted positives are correct?
    - **Recall** = $\frac{TP}{TP + FN}$ - How many actual positives did we find?
    - **F1-Score** = $2 \times \frac{Precision \times Recall}{Precision + Recall}$ - Harmonic mean

    ### When to Use Each Metric

    - **Accuracy:** Balanced classes, all errors equally bad
    - **Precision:** Cost of false positives is high (spam detection)
    - **Recall:** Cost of false negatives is high (disease detection)
    - **F1-Score:** Balance between precision and recall

    ### Class Imbalance

    If you have 90% positive reviews:
    - A model predicting "positive" for everything gets 90% accuracy!
    - Use stratified splitting
    - Consider weighted metrics or resampling
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Practical Example: Error Analysis

    Let's analyze where our model makes mistakes:
    """)
    return


@app.cell
def _(X_test, pd, y_pred, y_test):
    # Find misclassified examples
    misclassified_mask = y_test != y_pred
    misclassified_df = pd.DataFrame({
        'Text': X_test[misclassified_mask].values,
        'True_Label': y_test[misclassified_mask].values,
        'Predicted_Label': y_pred[misclassified_mask]
    })

    if len(misclassified_df) > 0:
        print(f"Misclassified examples: {len(misclassified_df)}")
        misclassified_df
    else:
        print("No misclassifications in test set (perfect model or small sample)")
    return


@app.cell
def _(baseline_model, pd, sample_df, vectorizer):
    # Prediction confidence analysis
    X_sample = vectorizer.transform(sample_df['text_cleaned'])
    probas = baseline_model.predict_proba(X_sample)
    max_probas = probas.max(axis=1)

    confidence_df = pd.DataFrame({
        'Text': sample_df['text'].values,
        'True_Sentiment': sample_df['sentiment'].values,
        'Predicted_Sentiment': baseline_model.predict(X_sample),
        'Confidence': max_probas
    })
    confidence_df = confidence_df.sort_values('Confidence')

    print("Low confidence predictions (potential issues):")
    confidence_df.head()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## Part 7: Unit Testing & Reproducibility

    ### Theory: Why Test Your Code?

    **Benefits of Testing:**
    - Catch bugs early
    - Document expected behavior
    - Enable safe refactoring
    - Build confidence in your pipeline

    ### What to Test in ML Pipelines

    1. **Data Validation** - Schema, types, ranges
    2. **Preprocessing** - Text cleaning works correctly
    3. **Feature Engineering** - Output shapes and types
    4. **Model Loading** - Can load saved models
    5. **Predictions** - Output format is correct
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Practical Example: Basic Unit Tests
    """)
    return


@app.cell
def _(preprocess_text):
    def test_preprocess_text():
        """Test text preprocessing function"""
        # Test basic cleaning
        assert preprocess_text("Hello World!") == "hello world"

        # Test URL removal
        assert "http" not in preprocess_text("Check https://example.com now")

        # Test special character removal (avoid @ as it's caught by email regex)
        assert preprocess_text("Test!!! #Great") == "test great"

        # Test whitespace normalization
        assert preprocess_text("Too   many    spaces") == "too many spaces"
        
        # Test email removal
        assert "email" not in preprocess_text("Contact us at test@email.com")

        print("All preprocessing tests passed!")

    # Run the test
    test_preprocess_text()
    return test_preprocess_text,


@app.cell
def _(ingest_feedback_csv, sample_df):
    def test_ingestion():
        """Test data ingestion function"""
        # Save test data
        test_file = 'test_data.csv'
        sample_df.to_csv(test_file, index=False)

        # Load and validate
        df = ingest_feedback_csv(test_file)

        # Assertions
        assert len(df) > 0, "DataFrame should not be empty"
        assert 'text' in df.columns, "Must have 'text' column"
        assert df['text'].isnull().sum() == 0, "No null text values"
        assert df['text'].str.len().min() > 0, "No empty text values"

        print("All ingestion tests passed!")

    # Run the test
    test_ingestion()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Task 4: Add Unit Tests

    **Objective:** Add unit tests for basic preprocessing behaviors

    **Required Tests:**
    1. Test data ingestion with valid CSV
    2. Test data ingestion with invalid CSV (error handling)
    3. Test text preprocessing functions
    4. Test feature extraction output shapes
    5. Test model prediction output format

    **File:** `tests/unit/test_preprocessing.py`

    **Acceptance:** CI passes tests
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## Part 8: Saving Model Artifacts

    ### Theory: Model Persistence

    **What to Save:**
    - Trained model weights
    - Feature extractors (vectorizers)
    - Preprocessing configurations
    - Training metadata (hyperparameters, metrics)

    **Common Formats:**
    - **Pickle (.pkl)** - Python objects, not language-agnostic
    - **Joblib** - Better for large numpy arrays
    - **ONNX** - Cross-platform model format
    - **HuggingFace** - For transformer models
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Practical Example: Save & Load Models
    """)
    return


@app.cell
def _(accuracy, baseline_model, vectorizer):
    import joblib
    import json
    from pathlib import Path

    # Create directories
    model_dir = Path('models/checkpoints/baseline_v1')
    model_dir.mkdir(parents=True, exist_ok=True)

    # Save model
    joblib.dump(baseline_model, model_dir / 'model.pkl')
    joblib.dump(vectorizer, model_dir / 'vectorizer.pkl')

    # Save metadata
    metadata = {
        'model_type': 'LogisticRegression',
        'features': 'TF-IDF',
        'accuracy': float(accuracy),
        'date_trained': '2025-11-12',
        'hyperparameters': {
            'max_iter': 1000,
            'random_state': 42
        }
    }

    with open(model_dir / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"Model saved to {model_dir}")
    return joblib, model_dir


@app.cell
def _(joblib, model_dir):
    # Load model back
    loaded_model = joblib.load(model_dir / 'model.pkl')
    loaded_vectorizer = joblib.load(model_dir / 'vectorizer.pkl')

    # Test prediction
    test_texts = ["This product is amazing!", "Terrible quality, very disappointed."]
    test_features = loaded_vectorizer.transform([t.lower() for t in test_texts])
    predictions = loaded_model.predict(test_features)

    print("Test predictions:", predictions)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## Sprint 1 Checklist

    Use this checklist to track your progress:

    ### Week 1: Data Foundation
    - [ ] Set up project structure
    - [ ] Identify and download datasets
    - [ ] Create ingestion script
    - [ ] Validate data schema
    - [ ] Generate cleaned dataset
    - [ ] Perform initial EDA (5-7 visualizations)
    - [ ] Document data quality issues

    ### Week 2: Baseline Model
    - [ ] Complete comprehensive EDA (10-15 charts)
    - [ ] Implement text preprocessing
    - [ ] Create train/val/test splits
    - [ ] Build TF-IDF feature extractor
    - [ ] Train baseline classifier
    - [ ] Calculate evaluation metrics
    - [ ] Perform error analysis
    - [ ] Add unit tests
    - [ ] Save model artifacts
    - [ ] Prepare demo presentation

    ### Sprint 1 Demo Preparation
    - [ ] Show ingestion running on sample file
    - [ ] Display 3-5 key EDA insights
    - [ ] Demonstrate baseline model predictions
    - [ ] Present metrics table
    - [ ] Discuss 2-3 interesting errors
    - [ ] Outline Sprint 2 improvements
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## Additional Resources

    ### Recommended Reading
    - **Scikit-learn Documentation:** [Text Feature Extraction](https://scikit-learn.org/stable/modules/feature_extraction.html#text-feature-extraction)
    - **Pandas Tutorial:** [10 Minutes to Pandas](https://pandas.pydata.org/docs/user_guide/10min.html)
    - **Plotly Express:** [Getting Started](https://plotly.com/python/plotly-express/)

    ### Useful Libraries
    - `pandas` - Data manipulation
    - `numpy` - Numerical operations
    - `scikit-learn` - ML models and preprocessing
    - `plotly` - Interactive visualizations
    - `nltk` or `spacy` - Advanced NLP preprocessing

    ### Sample Datasets
    - [Kaggle - Women's E-Commerce Reviews](https://www.kaggle.com/datasets/nicapotato/womens-ecommerce-clothing-reviews)
    - [Yelp Open Dataset](https://www.yelp.com/dataset)
    - [Amazon Product Reviews](https://nijianmo.github.io/amazon/index.html)
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## Next Steps

    After completing Sprint 1:

    1. **Demo Your Work** - Present to team and stakeholders
    2. **Gather Feedback** - Note improvement areas
    3. **Document Learnings** - Update project wiki/README
    4. **Plan Sprint 2** - Start thinking about:
       - Model improvements (fine-tuning transformers)
       - Topic modeling approaches
       - Summarization techniques

    **Ready for Sprint 2?** Check out `02_sprint_02_advanced_nlp.py` for advanced NLP techniques!

    ---

    Good luck with Sprint 1! Happy coding!
    """)
    return


if __name__ == "__main__":
    app.run()
