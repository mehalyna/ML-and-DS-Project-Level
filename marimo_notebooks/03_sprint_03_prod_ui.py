import marimo

__generated_with = "0.17.7"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Sprint 3: Production & UI

    **Customer Feedback Insight Platform - Weeks 5-6**

    Welcome to the final sprint! This notebook guides you through building production-ready infrastructure: REST APIs, interactive dashboards, containerization, and deployment.

    ## Sprint Goals

    By the end of Sprint 3, you will have:
    - FastAPI endpoints for sentiment, topic modeling, and summarization
    - Streamlit dashboard for browsing feedback and insights
    - Containerized application with Docker
    - Unit tests for key functions
    - Deployment documentation
    - Final demo materials

    **Timeline:** Weeks 5-6 | **Demo:** End of Week 6

    **Note:** This notebook uses FastAPI, Streamlit, and Docker for production deployment.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## Sprint 3 Deliverables

    ### Required Outputs

    1. **FastAPI Backend** - REST endpoints for ML predictions
    2. **Streamlit Dashboard** - Interactive UI for feedback analysis
    3. **Docker Containers** - Deployable application images
    4. **Unit Tests** - Test coverage for critical functions
    5. **API Documentation** - OpenAPI/Swagger specs
    6. **Deployment Guide** - Setup and run instructions
    7. **Final Demo** - End-to-end presentation

    ### Code Artifacts

    - `api/app.py` - FastAPI application
    - `api/routes/predict.py` - Prediction endpoints
    - `dashboard/streamlit_app.py` - Dashboard application
    - `ops/Dockerfile` - Container definitions
    - `ops/docker-compose.yml` - Multi-container orchestration
    - `tests/unit/` - Unit test suite
    - `docs/runbook.md` - Deployment documentation
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## Part 1: FastAPI REST Endpoints

    ### Theory: RESTful API Design

    **What is REST?**
    - **RE**presentational **S**tate **T**ransfer
    - Architectural style for web services
    - Uses HTTP methods: GET, POST, PUT, DELETE
    - Stateless: each request contains all needed information

    **Key Principles:**
    1. **Resources:** Everything is a resource (feedback, predictions)
    2. **URIs:** Unique identifiers for resources (`/api/predict/sentiment`)
    3. **HTTP Methods:** Standard operations (GET retrieve, POST create)
    4. **Stateless:** No session stored on server
    5. **JSON:** Standard data format

    **Why FastAPI?**
    - Automatic API documentation (Swagger UI)
    - Type validation with Pydantic
    - Async support for high performance
    - Built-in data validation
    - Easy testing

    ### API Design for ML Services

    **Best Practices:**
    - Versioned endpoints: `/api/v1/predict`
    - Clear response schemas
    - Proper error handling
    - Request validation
    - Health checks
    - Rate limiting for production
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Practical Example: FastAPI Application Structure

    First, let's create a simple FastAPI app with health check:
    """)
    return


@app.cell
def _():
    # Example FastAPI app structure (not executable in marimo)
    # This would go in api/app.py

    example_app_code = '''
    from fastapi import FastAPI
    from fastapi.middleware.cors import CORSMiddleware
    import uvicorn

    app = FastAPI(
    title="Customer Feedback Insights API",
    description="ML-powered feedback analysis",
    version="1.0.0"
    )

    # CORS middleware for dashboard access
    app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    )

    @app.get("/")
    async def root():
    return {"message": "Customer Feedback Insights API"}

    @app.get("/health")
    async def health_check():
    return {
        "status": "healthy",
        "service": "feedback-insights-api",
        "version": "1.0.0"
    }

    if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
    '''

    print("FastAPI App Structure:")
    print(example_app_code)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Pydantic Schemas for Request/Response

    Pydantic models ensure type safety and automatic validation:
    """)
    return


@app.cell
def _():
    from pydantic import BaseModel, Field
    from typing import List, Optional
    from datetime import datetime

    # Request schema
    class FeedbackRequest(BaseModel):
        text: str = Field(..., min_length=5, description="Feedback text")
        source: Optional[str] = Field("unknown", description="Feedback source")
        timestamp: Optional[datetime] = None

        class Config:
            json_schema_extra = {
                "example": {
                    "text": "Great product but expensive",
                    "source": "survey",
                    "timestamp": "2025-11-14T10:00:00"
                }
            }

    # Response schemas
    class SentimentResponse(BaseModel):
        text: str
        sentiment: str = Field(..., description="positive, negative, or neutral")
        confidence: float = Field(..., ge=0, le=1)
        scores: dict

    class TopicResponse(BaseModel):
        text: str
        topics: List[str]
        topic_scores: List[float]

    class SummaryResponse(BaseModel):
        text: str
        summary: str
        method: str = Field(..., description="Algorithm used")

    # Test the schema
    sample_request = FeedbackRequest(
        text="The product quality is excellent but delivery was slow",
        source="email"
    )

    print("Request Schema:")
    print(sample_request.model_dump_json(indent=2))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Prediction Endpoints Implementation

    Example endpoint for sentiment prediction:
    """)
    return


@app.cell
def _():
    # Example prediction endpoint (api/routes/predict.py)

    prediction_endpoint_code = '''
    from fastapi import APIRouter, HTTPException
    from api.schemas import FeedbackRequest, SentimentResponse
    import joblib
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

    router = APIRouter(prefix="/api/v1", tags=["predictions"])

    # Load models on startup
    vader = SentimentIntensityAnalyzer()
    tfidf_model = joblib.load("models/checkpoints/tfidf_vectorizer.pkl")
    classifier = joblib.load("models/checkpoints/sentiment_classifier.pkl")

    @router.post("/predict/sentiment", response_model=SentimentResponse)
    async def predict_sentiment(request: FeedbackRequest):
    """
    Predict sentiment for customer feedback.

    Returns:
        - sentiment: positive, negative, or neutral
        - confidence: prediction confidence score
        - scores: detailed sentiment scores
    """
    try:
        # VADER scores
        vader_scores = vader.polarity_scores(request.text)

        # TF-IDF + classifier prediction
        tfidf_vec = tfidf_model.transform([request.text])
        ml_pred = classifier.predict(tfidf_vec)[0]
        ml_proba = classifier.predict_proba(tfidf_vec)[0].max()

        # Ensemble: combine VADER and ML
        compound = vader_scores['compound']
        if abs(compound) > 0.5:
            # VADER confident
            sentiment = "positive" if compound > 0 else "negative"
            confidence = abs(compound)
        else:
            # Use ML model
            sentiment = ml_pred
            confidence = ml_proba

        return SentimentResponse(
            text=request.text,
            sentiment=sentiment,
            confidence=float(confidence),
            scores=vader_scores
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    @router.post("/predict/topics")
    async def predict_topics(request: FeedbackRequest):
    """Predict topics for feedback text."""
    # Implementation using LDA model
    pass

    @router.post("/predict/summary")
    async def generate_summary(request: FeedbackRequest):
    """Generate extractive summary of feedback."""
    # Implementation using Sumy
    pass
    '''

    print("Prediction Endpoint Example:")
    print(prediction_endpoint_code)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Error Handling and Validation

    **HTTP Status Codes:**
    - **200 OK:** Successful request
    - **400 Bad Request:** Invalid input
    - **404 Not Found:** Resource doesn't exist
    - **422 Unprocessable Entity:** Validation error
    - **500 Internal Server Error:** Server-side error

    **Best Practices:**
    1. Use HTTPException for errors
    2. Provide clear error messages
    3. Log errors for debugging
    4. Return consistent error format
    5. Don't expose internal details
    """)
    return


@app.cell
def _():
    # Example error handling
    error_handling_code = '''
    from fastapi import HTTPException, status
    import logging

    logger = logging.getLogger(__name__)

    @router.post("/predict/sentiment")
    async def predict_sentiment(request: FeedbackRequest):
    # Input validation
    if len(request.text.strip()) < 5:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Text must be at least 5 characters"
        )

    try:
        # Prediction logic
        result = model.predict(request.text)
        return result

    except ValueError as e:
        # Expected error (bad input)
        logger.warning(f"Validation error: {e}")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Invalid input: {str(e)}"
        )

    except Exception as e:
        # Unexpected error
        logger.error(f"Prediction failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error"
        )
    '''

    print("Error Handling Pattern:")
    print(error_handling_code)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Task 1: Build FastAPI Backend

    **Objective:** Create production-ready REST API for ML models

    **Steps:**
    1. **Project Structure:**
       - Create `api/app.py` with FastAPI application
       - Create `api/routes/predict.py` for prediction endpoints
       - Create `api/routes/health.py` for monitoring
       - Create `api/schemas.py` for Pydantic models

    2. **Implement Endpoints:**
       - POST `/api/v1/predict/sentiment` - Sentiment analysis
       - POST `/api/v1/predict/topics` - Topic extraction
       - POST `/api/v1/predict/summary` - Text summarization
       - POST `/api/v1/predict/batch` - Batch processing
       - GET `/health` - Health check
       - GET `/metrics` - Basic metrics

    3. **Model Loading:**
       - Load saved models on application startup
       - Use dependency injection for model access
       - Implement model caching

    4. **Documentation:**
       - Add docstrings to all endpoints
       - Provide request/response examples
       - Test Swagger UI at `/docs`

    5. **Testing:**
       - Write unit tests for endpoints
       - Test error cases
       - Validate response schemas

    **Deliverables:**
    - `api/app.py` - FastAPI application
    - `api/routes/predict.py` - Prediction endpoints
    - `api/schemas.py` - Pydantic models
    - `tests/integration/test_api_endpoints.py` - API tests
    - OpenAPI documentation accessible at `/docs`

    **Acceptance:** All endpoints return valid responses, tests pass, Swagger UI loads

    **Commands to run:**
    ```bash
    # Start API server
    uvicorn api.app:app --reload --port 8000

    # Test endpoint
    curl -X POST "http://localhost:8000/api/v1/predict/sentiment" \
         -H "Content-Type: application/json" \
         -d '{"text": "Great product!"}'

    # View docs
    # Navigate to http://localhost:8000/docs
    ```
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## Part 2: Streamlit Dashboard

    ### Theory: Interactive Dashboards

    **What is Streamlit?**
    - Python framework for data apps
    - No frontend experience needed
    - Reactive programming model
    - Built-in widgets and charts
    - Fast prototyping

    **Dashboard Components:**
    1. **Navigation:** Sidebar for filtering and controls
    2. **KPIs:** Key metrics (total feedback, avg sentiment)
    3. **Visualizations:** Charts and graphs
    4. **Data Tables:** Browsable feedback list
    5. **Filters:** Date range, sentiment, topics
    6. **Export:** Download filtered data

    **Design Principles:**
    - **Clarity:** Clear labels and instructions
    - **Responsiveness:** Fast updates
    - **Hierarchy:** Most important info at top
    - **Consistency:** Uniform styling
    - **Accessibility:** Readable fonts, good contrast
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Practical Example: Streamlit App Structure
    """)
    return


@app.cell
def _():
    # Example Streamlit app (dashboard/streamlit_app.py)

    streamlit_app_code = '''
    import streamlit as st
    import pandas as pd
    import plotly.express as px
    import requests
    from datetime import datetime, timedelta

    # Page config
    st.set_page_config(
    page_title="Feedback Insights Dashboard",
    page_icon="📊",
    layout="wide"
    )

    # Title
    st.title("Customer Feedback Insights Dashboard")
    st.markdown("Analyze customer feedback with ML-powered sentiment, topics, and summaries")

    # Sidebar filters
    st.sidebar.header("Filters")

    # Date range
    date_range = st.sidebar.date_input(
    "Date Range",
    value=(datetime.now() - timedelta(days=30), datetime.now())
    )

    # Sentiment filter
    sentiment_filter = st.sidebar.multiselect(
    "Sentiment",
    options=["positive", "negative", "neutral"],
    default=["positive", "negative", "neutral"]
    )

    # Load data (placeholder)
    @st.cache_data
    def load_feedback_data():
    # In production, load from database or API
    return pd.DataFrame({
        'timestamp': pd.date_range(start='2025-10-01', periods=100, freq='D'),
        'text': ['Sample feedback'] * 100,
        'sentiment': ['positive'] * 50 + ['negative'] * 30 + ['neutral'] * 20,
        'source': ['email', 'survey', 'chat'] * 33 + ['email']
    })

    df = load_feedback_data()

    # Filter data
    mask = (
    (df['timestamp'] >= pd.Timestamp(date_range[0])) &
    (df['timestamp'] <= pd.Timestamp(date_range[1])) &
    (df['sentiment'].isin(sentiment_filter))
    )
    filtered_df = df[mask]

    # KPIs
    col1, col2, col3, col4 = st.columns(4)

    with col1:
    st.metric("Total Feedback", len(filtered_df))

    with col2:
    pos_pct = (filtered_df['sentiment'] == 'positive').sum() / len(filtered_df) * 100
    st.metric("Positive %", f"{pos_pct:.1f}%")

    with col3:
    neg_pct = (filtered_df['sentiment'] == 'negative').sum() / len(filtered_df) * 100
    st.metric("Negative %", f"{neg_pct:.1f}%")

    with col4:
    st.metric("Unique Sources", filtered_df['source'].nunique())

    # Visualizations
    col1, col2 = st.columns(2)

    with col1:
    st.subheader("Sentiment Distribution")
    sentiment_counts = filtered_df['sentiment'].value_counts()
    fig = px.pie(values=sentiment_counts.values, names=sentiment_counts.index)
    st.plotly_chart(fig, use_container_width=True)

    with col2:
    st.subheader("Feedback Over Time")
    timeline = filtered_df.groupby('timestamp').size().reset_index(name='count')
    fig = px.line(timeline, x='timestamp', y='count')
    st.plotly_chart(fig, use_container_width=True)

    # Feedback table
    st.subheader("Recent Feedback")
    st.dataframe(filtered_df.head(10), use_container_width=True)
    '''

    print("Streamlit Dashboard Example:")
    print(streamlit_app_code)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Interactive Widgets in Streamlit

    **Input Widgets:**
    - `st.text_input()` - Single line text
    - `st.text_area()` - Multi-line text
    - `st.selectbox()` - Dropdown selection
    - `st.multiselect()` - Multiple selections
    - `st.slider()` - Numeric range
    - `st.date_input()` - Date picker
    - `st.file_uploader()` - File upload

    **Display Widgets:**
    - `st.metric()` - KPI cards
    - `st.dataframe()` - Interactive tables
    - `st.table()` - Static tables
    - `st.plotly_chart()` - Plotly visualizations
    - `st.markdown()` - Formatted text
    """)
    return


@app.cell
def _():
    # Example: Live prediction in Streamlit

    live_prediction_code = '''
    import streamlit as st
    import requests

    st.subheader("Live Sentiment Prediction")

    # Text input
    user_input = st.text_area(
    "Enter feedback text:",
    placeholder="Type customer feedback here..."
    )

    # Predict button
    if st.button("Analyze Sentiment"):
    if user_input.strip():
        # Call API
        try:
            response = requests.post(
                "http://localhost:8000/api/v1/predict/sentiment",
                json={"text": user_input}
            )

            if response.status_code == 200:
                result = response.json()

                # Display results
                col1, col2 = st.columns(2)

                with col1:
                    st.metric("Sentiment", result['sentiment'].upper())

                with col2:
                    st.metric("Confidence", f"{result['confidence']:.2%}")

                # Detailed scores
                st.json(result['scores'])
            else:
                st.error(f"API Error: {response.status_code}")

        except Exception as e:
            st.error(f"Connection error: {e}")
    else:
        st.warning("Please enter some text")
    '''

    print("Live Prediction Widget:")
    print(live_prediction_code)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Caching and Performance

    **Streamlit Caching:**
    - `@st.cache_data` - Cache data loading
    - `@st.cache_resource` - Cache models/connections
    - Improves performance for repeated operations
    - Automatic cache invalidation

    **Best Practices:**
    1. Cache expensive operations (data loading, model inference)
    2. Use session state for user interactions
    3. Minimize API calls
    4. Lazy load large datasets
    5. Use pagination for tables
    """)
    return


@app.cell
def _():
    # Caching example
    caching_example = '''
    import streamlit as st
    import pandas as pd
    import joblib

    @st.cache_data
    def load_data(filepath):
    """Cache data loading - runs once until file changes"""
    return pd.read_csv(filepath)

    @st.cache_resource
    def load_model():
    """Cache model loading - runs once per session"""
    return joblib.load("models/checkpoints/sentiment_model.pkl")

    # Use session state for user inputs
    if 'prediction_history' not in st.session_state:
    st.session_state.prediction_history = []

    # Load cached resources
    df = load_data("data/processed/feedback.csv")
    model = load_model()

    # Make prediction
    if st.button("Predict"):
    result = model.predict([user_input])
    st.session_state.prediction_history.append(result)
    '''

    print("Caching Pattern:")
    print(caching_example)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Task 2: Build Streamlit Dashboard

    **Objective:** Create interactive dashboard for feedback analysis

    **Steps:**
    1. **Main Dashboard (Home Page):**
       - KPI cards: total feedback, sentiment distribution, avg confidence
       - Time series: feedback volume over time
       - Pie chart: sentiment breakdown
       - Bar chart: feedback by source
       - Data table: recent feedback with filters

    2. **Analytics Page:**
       - Topic distribution visualization
       - Topic-sentiment correlation
       - Word clouds for each topic
       - Trend analysis over time

    3. **Live Prediction Page:**
       - Text area for input
       - Predict button
       - Display sentiment, topics, summary
       - Show confidence scores
       - Prediction history

    4. **Data Upload Page:**
       - File uploader (CSV/JSON)
       - Data preview
       - Batch prediction
       - Export results

    5. **Sidebar Navigation:**
       - Page selector
       - Date range filter
       - Sentiment filter
       - Source filter
       - Export button

    **Deliverables:**
    - `dashboard/streamlit_app.py` - Main application
    - `dashboard/components/` - Reusable components
    - `dashboard/utils.py` - Helper functions
    - Screenshots of each page

    **Acceptance:** Dashboard loads, filters work, live prediction functional, visualizations render

    **Commands to run:**
    ```bash
    # Start dashboard
    streamlit run dashboard/streamlit_app.py

    # Navigate to http://localhost:8501
    ```
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## Part 3: Containerization with Docker

    ### Theory: Docker Fundamentals

    **What is Docker?**
    - Platform for containerization
    - Packages application with dependencies
    - Ensures consistent environment
    - Portable across systems

    **Key Concepts:**
    1. **Image:** Blueprint for container (like a class)
    2. **Container:** Running instance (like an object)
    3. **Dockerfile:** Instructions to build image
    4. **Registry:** Storage for images (Docker Hub)
    5. **Volumes:** Persistent data storage
    6. **Networks:** Container communication

    **Benefits:**
    - **Consistency:** Same environment everywhere
    - **Isolation:** Dependencies don't conflict
    - **Portability:** Run anywhere Docker runs
    - **Scalability:** Easy to replicate
    - **Version Control:** Tag images
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Dockerfile Anatomy

    **Common Instructions:**
    - `FROM` - Base image
    - `WORKDIR` - Set working directory
    - `COPY` - Copy files into image
    - `RUN` - Execute commands during build
    - `ENV` - Set environment variables
    - `EXPOSE` - Document ports
    - `CMD` - Default command to run
    """)
    return


@app.cell
def _():
    # Example Dockerfile for API

    dockerfile_api = '''
    # ops/Dockerfile.api

    # Use official Python runtime
    FROM python:3.11-slim

    # Set working directory
    WORKDIR /app

    # Install system dependencies
    RUN apt-get update && apt-get install -y \\
    build-essential \\
    && rm -rf /var/lib/apt/lists/*

    # Copy requirements and install Python dependencies
    COPY requirements.txt .
    RUN pip install --no-cache-dir -r requirements.txt

    # Copy application code
    COPY api/ ./api/
    COPY src/ ./src/
    COPY models/ ./models/

    # Download NLTK data
    RUN python -c "import nltk; nltk.download('punkt_tab', quiet=True)"

    # Expose port
    EXPOSE 8000

    # Health check
    HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \\
    CMD curl -f http://localhost:8000/health || exit 1

    # Run application
    CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]
    '''

    print("API Dockerfile:")
    print(dockerfile_api)
    return


@app.cell
def _():
    # Example Dockerfile for Dashboard

    dockerfile_dashboard = '''
    # ops/Dockerfile.dashboard

    FROM python:3.11-slim

    WORKDIR /app

    # Copy requirements
    COPY requirements.txt .
    RUN pip install --no-cache-dir -r requirements.txt

    # Copy dashboard code
    COPY dashboard/ ./dashboard/
    COPY src/ ./src/

    # Expose Streamlit port
    EXPOSE 8501

    # Health check
    HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \\
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

    # Run Streamlit
    CMD ["streamlit", "run", "dashboard/streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
    '''

    print("\nDashboard Dockerfile:")
    print(dockerfile_dashboard)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Docker Compose for Multi-Container Apps

    **Docker Compose:**
    - Orchestrate multiple containers
    - Define services, networks, volumes
    - Single command to start all services
    - Environment configuration
    """)
    return


@app.cell
def _():
    # Docker Compose configuration

    docker_compose = '''
    # ops/docker-compose.yml

    version: '3.8'

    services:
      api:
    build:
      context: ..
      dockerfile: ops/Dockerfile.api
    ports:
      - "8000:8000"
    volumes:
      - ../models:/app/models:ro
      - ../data:/app/data:ro
    environment:
      - ENVIRONMENT=production
      - LOG_LEVEL=info
    networks:
      - feedback-network
    restart: unless-stopped

      dashboard:
    build:
      context: ..
      dockerfile: ops/Dockerfile.dashboard
    ports:
      - "8501:8501"
    environment:
      - API_URL=http://api:8000
    depends_on:
      - api
    networks:
      - feedback-network
    restart: unless-stopped

    networks:
      feedback-network:
    driver: bridge

    volumes:
      model-data:
      feedback-data:
    '''

    print("Docker Compose Configuration:")
    print(docker_compose)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Docker Commands Cheat Sheet

    **Build and Run:**
    ```bash
    # Build image
    docker build -t feedback-api:latest -f ops/Dockerfile.api .

    # Run container
    docker run -p 8000:8000 feedback-api:latest

    # Run in background
    docker run -d -p 8000:8000 --name feedback-api feedback-api:latest
    ```

    **Docker Compose:**
    ```bash
    # Start all services
    docker-compose -f ops/docker-compose.yml up

    # Build and start
    docker-compose -f ops/docker-compose.yml up --build

    # Run in background
    docker-compose -f ops/docker-compose.yml up -d

    # Stop all services
    docker-compose -f ops/docker-compose.yml down

    # View logs
    docker-compose -f ops/docker-compose.yml logs -f
    ```

    **Management:**
    ```bash
    # List running containers
    docker ps

    # View logs
    docker logs feedback-api

    # Execute command in container
    docker exec -it feedback-api bash

    # Stop container
    docker stop feedback-api

    # Remove container
    docker rm feedback-api
    ```
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Task 3: Containerize Application

    **Objective:** Create deployable Docker containers for API and dashboard

    **Steps:**
    1. **Create Dockerfiles:**
       - `ops/Dockerfile.api` - API container
       - `ops/Dockerfile.dashboard` - Dashboard container
       - Use multi-stage builds if needed
       - Minimize image size

    2. **Create Docker Compose:**
       - `ops/docker-compose.yml` - Multi-container orchestration
       - Define services, networks, volumes
       - Configure environment variables
       - Set up dependencies

    3. **Optimize Images:**
       - Use slim base images
       - Minimize layers
       - Clean up after installs
       - Use .dockerignore

    4. **Test Containers:**
       - Build images locally
       - Run containers
       - Test API endpoints
       - Test dashboard access
       - Verify inter-service communication

    5. **Documentation:**
       - Create `docs/runbook.md`
       - Document build process
       - Document deployment steps
       - Include troubleshooting

    **Deliverables:**
    - `ops/Dockerfile.api` - API Dockerfile
    - `ops/Dockerfile.dashboard` - Dashboard Dockerfile
    - `ops/docker-compose.yml` - Compose configuration
    - `ops/.dockerignore` - Ignore file
    - `docs/runbook.md` - Deployment guide

    **Acceptance:** Containers build successfully, services start and communicate, documentation complete

    **Commands to validate:**
    ```bash
    # Build images
    docker-compose -f ops/docker-compose.yml build

    # Start services
    docker-compose -f ops/docker-compose.yml up

    # Test API
    curl http://localhost:8000/health

    # Access dashboard
    # Open http://localhost:8501 in browser
    ```
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## Part 4: Testing and Quality Assurance

    ### Theory: Testing Pyramid

    **Test Levels:**
    1. **Unit Tests:** Test individual functions (70% of tests)
    2. **Integration Tests:** Test component interactions (20%)
    3. **End-to-End Tests:** Test full workflows (10%)

    **Why Test?**
    - Catch bugs early
    - Enable refactoring
    - Document behavior
    - Prevent regressions
    - Build confidence

    **Testing Best Practices:**
    - Write tests first (TDD)
    - One assertion per test
    - Test edge cases
    - Use fixtures for setup
    - Mock external dependencies
    - Aim for 80%+ coverage
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Unit Testing with pytest
    """)
    return


@app.cell
def _():
    # Example unit tests

    unit_tests = '''
    # tests/unit/test_preprocessing.py

    import pytest
    from src.feedback_insights.preprocessing import preprocess_text, remove_stopwords

    class TestPreprocessing:
    """Test text preprocessing functions"""

    def test_preprocess_text_basic(self):
        """Test basic text cleaning"""
        text = "Hello World!  "
        result = preprocess_text(text)
        assert result == "hello world"

    def test_preprocess_text_urls(self):
        """Test URL removal"""
        text = "Check out https://example.com for more"
        result = preprocess_text(text)
        assert "https" not in result
        assert "example.com" not in result

    def test_preprocess_text_special_chars(self):
        """Test special character handling"""
        text = "Amazing!!! Product###"
        result = preprocess_text(text)
        assert "!" not in result
        assert "#" not in result

    def test_preprocess_text_empty(self):
        """Test empty input handling"""
        assert preprocess_text("") == ""
        assert preprocess_text("   ") == ""

    def test_remove_stopwords(self):
        """Test stopword removal"""
        words = ["the", "product", "is", "great"]
        result = remove_stopwords(words)
        assert "product" in result
        assert "great" in result
        assert "the" not in result
        assert "is" not in result

    # Run tests with:
    # pytest tests/unit/test_preprocessing.py -v
    '''

    print("Unit Test Examples:")
    print(unit_tests)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### API Integration Testing
    """)
    return


@app.cell
def _():
    # Example API integration tests

    integration_tests = '''
    # tests/integration/test_api_endpoints.py

    import pytest
    from fastapi.testclient import TestClient
    from api.app import app

    client = TestClient(app)

    class TestHealthEndpoints:
    """Test health check endpoints"""

    def test_root(self):
        """Test root endpoint"""
        response = client.get("/")
        assert response.status_code == 200
        assert "message" in response.json()

    def test_health(self):
        """Test health check"""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"

    class TestPredictionEndpoints:
    """Test ML prediction endpoints"""

    def test_sentiment_prediction_success(self):
        """Test successful sentiment prediction"""
        payload = {
            "text": "This product is absolutely amazing!",
            "source": "review"
        }
        response = client.post("/api/v1/predict/sentiment", json=payload)

        assert response.status_code == 200
        data = response.json()
        assert "sentiment" in data
        assert "confidence" in data
        assert data["sentiment"] in ["positive", "negative", "neutral"]
        assert 0 <= data["confidence"] <= 1

    def test_sentiment_prediction_invalid_input(self):
        """Test validation error handling"""
        payload = {"text": ""}  # Too short
        response = client.post("/api/v1/predict/sentiment", json=payload)

        assert response.status_code == 422  # Validation error

    def test_sentiment_prediction_missing_field(self):
        """Test missing required field"""
        payload = {}  # Missing text
        response = client.post("/api/v1/predict/sentiment", json=payload)

        assert response.status_code == 422

    # Run with:
    # pytest tests/integration/test_api_endpoints.py -v
    '''

    print("Integration Test Examples:")
    print(integration_tests)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Test Coverage and CI

    **Measuring Coverage:**
    ```bash
    # Install coverage
    pip install pytest-cov

    # Run tests with coverage
    pytest --cov=src --cov=api --cov-report=html

    # View report
    # open htmlcov/index.html
    ```

    **Continuous Integration (CI):**
    - Run tests on every commit
    - Automated quality checks
    - Fast feedback loop
    - Prevent broken code merging
    """)
    return


@app.cell
def _():
    # Example GitHub Actions CI

    github_actions_ci = '''
    # ops/ci/github-actions.yml

    name: CI Pipeline

    on:
      push:
    branches: [ main, develop ]
      pull_request:
    branches: [ main ]

    jobs:
      test:
    runs-on: ubuntu-latest

    steps:
    - uses: actions/checkout@v3

    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.11'

    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
        pip install -r requirements-dev.txt

    - name: Lint with flake8
      run: |
        flake8 src/ api/ --count --max-line-length=120 --statistics

    - name: Run unit tests
      run: |
        pytest tests/unit/ -v --cov=src --cov=api --cov-report=xml

    - name: Run integration tests
      run: |
        pytest tests/integration/ -v

    - name: Upload coverage
      uses: codecov/codecov-action@v3
      with:
        file: ./coverage.xml

      build:
    runs-on: ubuntu-latest
    needs: test

    steps:
    - uses: actions/checkout@v3

    - name: Build Docker images
      run: |
        docker build -t feedback-api:${{ github.sha }} -f ops/Dockerfile.api .
        docker build -t feedback-dashboard:${{ github.sha }} -f ops/Dockerfile.dashboard .

    - name: Test containers
      run: |
        docker-compose -f ops/docker-compose.yml up -d
        sleep 10
        curl -f http://localhost:8000/health || exit 1
        docker-compose -f ops/docker-compose.yml down
    '''

    print("CI Pipeline Configuration:")
    print(github_actions_ci)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Task 4: Implement Testing Suite

    **Objective:** Build comprehensive test coverage

    **Steps:**
    1. **Unit Tests:**
       - `tests/unit/test_preprocessing.py` - Text preprocessing
       - `tests/unit/test_models.py` - Model functions
       - `tests/unit/test_utils.py` - Utility functions
       - Test edge cases and errors

    2. **Integration Tests:**
       - `tests/integration/test_api_endpoints.py` - API tests
       - Test all endpoints
       - Test error handling
       - Test request/response schemas

    3. **Test Configuration:**
       - Create `pytest.ini` - Pytest config
       - Create `conftest.py` - Shared fixtures
       - Add test requirements to `requirements-dev.txt`

    4. **CI Setup:**
       - Create `.github/workflows/ci.yml` - GitHub Actions
       - Configure linting (flake8, black)
       - Configure test execution
       - Configure coverage reporting

    5. **Documentation:**
       - Add testing section to README
       - Document test execution
       - Document coverage targets

    **Deliverables:**
    - `tests/unit/` - Unit test suite
    - `tests/integration/` - Integration tests
    - `pytest.ini` - Test configuration
    - `.github/workflows/ci.yml` - CI pipeline
    - Test coverage report

    **Acceptance:** 80%+ test coverage, all tests pass, CI pipeline runs successfully

    **Commands:**
    ```bash
    # Run all tests
    pytest tests/ -v

    # Run with coverage
    pytest --cov=src --cov=api --cov-report=html

    # Run specific test file
    pytest tests/unit/test_preprocessing.py -v
    ```
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## Part 5: Deployment and Monitoring

    ### Theory: Production Deployment

    **Deployment Strategies:**
    1. **Single Server:** Simple, good for demos
    2. **Cloud Platforms:** AWS, GCP, Azure
    3. **Container Orchestration:** Kubernetes
    4. **Serverless:** AWS Lambda, Cloud Functions

    **Production Considerations:**
    - **Scalability:** Handle increased load
    - **Reliability:** Uptime and fault tolerance
    - **Security:** Authentication, encryption
    - **Monitoring:** Logs, metrics, alerts
    - **Backup:** Data and model versioning
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Logging and Monitoring

    **Logging Levels:**
    - DEBUG: Detailed diagnostic info
    - INFO: General informational messages
    - WARNING: Warning messages
    - ERROR: Error messages
    - CRITICAL: Critical failures

    **What to Log:**
    - Request/response details
    - Model predictions
    - Errors and exceptions
    - Performance metrics
    - User actions
    """)
    return


@app.cell
def _():
    # Example logging configuration

    logging_config = '''
    # api/logging_config.py

    import logging
    import sys
    from datetime import datetime

    def setup_logging(level=logging.INFO):
    """Configure application logging"""

    # Create formatter
    formatter = logging.Formatter(
        fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)

    # File handler
    file_handler = logging.FileHandler(
        f'logs/app_{datetime.now().strftime("%Y%m%d")}.log'
    )
    file_handler.setFormatter(formatter)

    # Configure root logger
    logger = logging.getLogger()
    logger.setLevel(level)
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger

    # Usage in API
    from api.logging_config import setup_logging

    logger = setup_logging()

    @app.post("/predict/sentiment")
    async def predict_sentiment(request: FeedbackRequest):
    logger.info(f"Prediction request received: source={request.source}")

    try:
        result = model.predict(request.text)
        logger.info(f"Prediction successful: sentiment={result['sentiment']}")
        return result
    except Exception as e:
        logger.error(f"Prediction failed: {e}", exc_info=True)
        raise
    '''

    print("Logging Configuration:")
    print(logging_config)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Environment Configuration

    **Best Practices:**
    - Use environment variables for config
    - Never commit secrets
    - Use .env files locally
    - Use secret managers in production
    """)
    return


@app.cell
def _():
    # Example environment configuration

    env_config = '''
    # .env (DO NOT COMMIT)

    ENVIRONMENT=development
    LOG_LEVEL=DEBUG
    API_HOST=0.0.0.0
    API_PORT=8000
    MODEL_PATH=models/checkpoints
    DATA_PATH=data/processed
    DATABASE_URL=sqlite:///feedback.db

    # api/config.py

    from pydantic_settings import BaseSettings

    class Settings(BaseSettings):
    environment: str = "development"
    log_level: str = "INFO"
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    model_path: str = "models/checkpoints"
    data_path: str = "data/processed"

    class Config:
        env_file = ".env"

    settings = Settings()

    # Usage
    from api.config import settings

    @app.on_event("startup")
    async def startup_event():
    logger.info(f"Starting in {settings.environment} mode")
    load_models(settings.model_path)
    '''

    print("Environment Configuration:")
    print(env_config)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Basic Monitoring Endpoint
    """)
    return


@app.cell
def _():
    # Example metrics endpoint

    metrics_endpoint = '''
    from fastapi import APIRouter
    from datetime import datetime
    import psutil

    router = APIRouter(prefix="/metrics", tags=["monitoring"])

    # Simple in-memory metrics
    metrics = {
    "requests_total": 0,
    "predictions_total": 0,
    "errors_total": 0,
    "start_time": datetime.now()
    }

    @router.get("/")
    async def get_metrics():
    """Get application metrics"""
    uptime = (datetime.now() - metrics["start_time"]).total_seconds()

    return {
        "uptime_seconds": uptime,
        "requests_total": metrics["requests_total"],
        "predictions_total": metrics["predictions_total"],
        "errors_total": metrics["errors_total"],
        "error_rate": metrics["errors_total"] / max(metrics["predictions_total"], 1),
        "cpu_percent": psutil.cpu_percent(),
        "memory_percent": psutil.virtual_memory().percent
    }

    # Middleware to track requests
    @app.middleware("http")
    async def track_requests(request: Request, call_next):
    metrics["requests_total"] += 1
    response = await call_next(request)
    return response
    '''

    print("Metrics Endpoint:")
    print(metrics_endpoint)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Task 5: Deployment Documentation

    **Objective:** Create comprehensive deployment guide

    **Steps:**
    1. **Runbook Creation:**
       - `docs/runbook.md` - Deployment procedures
       - Prerequisites and dependencies
       - Step-by-step deployment
       - Configuration guide
       - Troubleshooting section

    2. **Architecture Documentation:**
       - `docs/architecture.md` - System design
       - Component diagram
       - Data flow
       - API specifications
       - Database schema (if applicable)

    3. **Local Deployment:**
       - Document Docker Compose setup
       - Environment variables
       - Data initialization
       - Testing procedures

    4. **Cloud Deployment (Optional):**
       - AWS/GCP/Azure setup
       - Container registry usage
       - Load balancer configuration
       - SSL/HTTPS setup

    5. **Monitoring Setup:**
       - Logging configuration
       - Metrics collection
       - Alert configuration
       - Dashboard access

    **Deliverables:**
    - `docs/runbook.md` - Complete deployment guide
    - `docs/architecture.md` - Architecture documentation
    - `docs/troubleshooting.md` - Common issues and solutions
    - `.env.example` - Example environment config
    - `Makefile` - Automation commands

    **Acceptance:** Team member can deploy from documentation, all procedures tested
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## Sprint 3 Progress Checklist

    ### Week 5: API and Dashboard

    - [ ] Design API endpoints and schemas
    - [ ] Implement FastAPI application
    - [ ] Create Pydantic models for validation
    - [ ] Implement sentiment prediction endpoint
    - [ ] Implement topic prediction endpoint
    - [ ] Implement summarization endpoint
    - [ ] Add error handling and logging
    - [ ] Test API with Swagger UI
    - [ ] Create Streamlit dashboard structure
    - [ ] Implement main dashboard page with KPIs
    - [ ] Add filtering and date range selection
    - [ ] Create visualizations (charts, graphs)
    - [ ] Implement live prediction page
    - [ ] Add data table with pagination

    ### Week 6: Testing and Deployment

    - [ ] Write unit tests for preprocessing
    - [ ] Write unit tests for model functions
    - [ ] Write integration tests for API endpoints
    - [ ] Achieve 80%+ test coverage
    - [ ] Set up pytest configuration
    - [ ] Create Dockerfile for API
    - [ ] Create Dockerfile for dashboard
    - [ ] Create docker-compose.yml
    - [ ] Test containers locally
    - [ ] Configure CI pipeline (GitHub Actions)
    - [ ] Add linting and formatting checks
    - [ ] Write deployment documentation
    - [ ] Create architecture diagram
    - [ ] Write troubleshooting guide
    - [ ] Prepare final demo presentation

    ### Final Demo Preparation

    - [ ] Test full end-to-end workflow
    - [ ] Prepare demo dataset
    - [ ] Create demo script
    - [ ] Test API endpoints for demo
    - [ ] Test dashboard for demo
    - [ ] Prepare slides with metrics
    - [ ] Document lessons learned
    - [ ] Create project summary report
    - [ ] Record limitations and next steps
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## Best Practices Summary

    ### API Development

    1. **Validation:** Use Pydantic for automatic validation
    2. **Documentation:** Leverage FastAPI's auto-docs
    3. **Error Handling:** Return consistent error responses
    4. **Versioning:** Use API versioning from the start
    5. **Security:** Implement authentication in production
    6. **Performance:** Use async endpoints when possible
    7. **Monitoring:** Log all requests and errors

    ### Dashboard Development

    1. **UX:** Keep interface simple and intuitive
    2. **Performance:** Cache expensive operations
    3. **Responsiveness:** Use appropriate chart types
    4. **Accessibility:** Ensure readable fonts and colors
    5. **Mobile:** Test on different screen sizes
    6. **Feedback:** Show loading states and errors clearly

    ### Containerization

    1. **Size:** Use slim base images
    2. **Layers:** Minimize number of layers
    3. **Secrets:** Never include secrets in images
    4. **Security:** Scan images for vulnerabilities
    5. **Versioning:** Tag images properly
    6. **Documentation:** Document build and run process

    ### Testing

    1. **Coverage:** Aim for 80%+ code coverage
    2. **Automation:** Run tests in CI/CD
    3. **Isolation:** Use fixtures for test setup
    4. **Naming:** Use descriptive test names
    5. **Assertions:** One assertion per test when possible
    6. **Edge Cases:** Test boundary conditions

    ### Deployment

    1. **Environment:** Separate dev/staging/production
    2. **Configuration:** Use environment variables
    3. **Logging:** Comprehensive logging strategy
    4. **Monitoring:** Track key metrics
    5. **Backup:** Regular backups of data and models
    6. **Documentation:** Keep runbook updated
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## Additional Resources

    ### FastAPI
    - [FastAPI Documentation](https://fastapi.tiangolo.com/) - Official docs
    - [FastAPI Tutorial](https://fastapi.tiangolo.com/tutorial/) - Step-by-step guide
    - [Pydantic Documentation](https://docs.pydantic.dev/) - Data validation

    ### Streamlit
    - [Streamlit Documentation](https://docs.streamlit.io/) - Official docs
    - [Streamlit Gallery](https://streamlit.io/gallery) - Example apps
    - [Streamlit Cheat Sheet](https://docs.streamlit.io/library/cheatsheet) - Quick reference

    ### Docker
    - [Docker Documentation](https://docs.docker.com/) - Official docs
    - [Docker Best Practices](https://docs.docker.com/develop/dev-best-practices/) - Production tips
    - [Docker Compose Documentation](https://docs.docker.com/compose/) - Compose reference

    ### Testing
    - [pytest Documentation](https://docs.pytest.org/) - Testing framework
    - [pytest-cov](https://pytest-cov.readthedocs.io/) - Coverage plugin
    - [TestClient (FastAPI)](https://fastapi.tiangolo.com/tutorial/testing/) - API testing

    ### DevOps
    - [GitHub Actions](https://docs.github.com/en/actions) - CI/CD platform
    - [Python Logging](https://docs.python.org/3/library/logging.html) - Logging module
    - [12 Factor App](https://12factor.net/) - Deployment methodology
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## Next Steps

    After completing Sprint 3, you have a production-ready ML application! Here are potential next steps:

    ### Immediate Improvements

    1. **Authentication:** Add user authentication to API
    2. **Database:** Replace CSV with PostgreSQL
    3. **Caching:** Implement Redis for prediction caching
    4. **Rate Limiting:** Add request rate limiting
    5. **HTTPS:** Configure SSL certificates

    ### Advanced Features

    1. **Model Monitoring:** Track prediction drift
    2. **A/B Testing:** Compare model versions
    3. **Active Learning:** Surface uncertain predictions
    4. **Batch Processing:** Async batch prediction endpoints
    5. **Model Registry:** Version and track models

    ### Scalability

    1. **Kubernetes:** Deploy on K8s cluster
    2. **Load Balancer:** Distribute traffic
    3. **Autoscaling:** Scale based on load
    4. **CDN:** Cache static assets
    5. **Message Queue:** Process predictions async

    ### Observability

    1. **Prometheus:** Metrics collection
    2. **Grafana:** Metrics visualization
    3. **ELK Stack:** Centralized logging
    4. **Sentry:** Error tracking
    5. **OpenTelemetry:** Distributed tracing

    Congratulations on completing the Customer Feedback Insight Platform course!
    """)
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()
