# Customer Feedback Insight Platform

A comprehensive end-to-end machine learning platform for analyzing customer feedback from multiple channels. This project demonstrates production-ready ML engineering practices including data ingestion, NLP model development, REST API deployment, and interactive dashboards.

## Project Overview

### Goal

Build an intelligent system that transforms raw customer feedback into actionable insights through automated sentiment analysis, topic modeling, and text summarization. The platform empowers product teams to quickly identify customer pain points, track sentiment trends, and prioritize improvements based on real feedback data.

### Core Values

- **Production-Ready**: Focus on deployable, maintainable code rather than research experiments
- **Lightweight & Reliable**: Use proven libraries that work across platforms without heavy dependencies
- **Educational**: Comprehensive documentation and examples for learning ML engineering practices
- **Practical**: Real-world workflows including testing, containerization, and CI/CD
- **Scalable**: Architecture designed for growth from prototype to production

### Problem Statement

Product teams receive feedback from multiple channels (emails, surveys, support tickets, reviews) but struggle to:
- Manually process large volumes of feedback
- Identify common themes and pain points
- Track sentiment trends over time
- Prioritize issues based on customer impact
- Extract actionable insights quickly

This platform automates feedback analysis, surfacing insights that would otherwise require hours of manual review.

## Key Features

### 1. Multi-Channel Data Ingestion
- CSV and JSON file support
- Data validation and cleaning
- Configurable source tracking
- Timestamp normalization
- Duplicate detection

### 2. Sentiment Analysis
- Ensemble approach combining VADER lexicon-based analysis with TF-IDF + Logistic Regression
- Confidence scores for predictions
- Support for positive, negative, and neutral classifications
- Fast inference (sub-100ms per document)
- No GPU requirements

### 3. Topic Modeling
- Gensim LDA (Latent Dirichlet Allocation) for lightweight, interpretable topics
- Coherence metrics for topic quality validation
- Customizable number of topics
- Human-readable topic labels
- Topic distribution visualization

### 4. Text Summarization
- Multiple extractive algorithms: TextRank, LexRank, LSA
- Configurable summary length
- Preserves original wording for accuracy
- ROUGE score evaluation
- No transformer dependencies (production-ready)

### 5. REST API (FastAPI)
- `/api/v1/predict/sentiment` - Sentiment prediction endpoint
- `/api/v1/predict/topics` - Topic extraction endpoint
- `/api/v1/predict/summary` - Text summarization endpoint
- `/health` - Service health check
- `/metrics` - Performance metrics
- Automatic OpenAPI documentation
- Request validation with Pydantic schemas

### 6. Interactive Dashboard (Streamlit)
- Real-time sentiment distribution visualization
- Feedback timeline analysis
- Topic clustering explorer
- Live prediction interface
- Filterable data tables
- Exportable results
- KPI cards for quick insights

### 7. Production Infrastructure
- Docker containerization for API and dashboard
- Docker Compose for local deployment
- Comprehensive unit and integration tests
- GitHub Actions CI/CD pipeline
- Structured logging
- Environment-based configuration
- Health monitoring endpoints

## Technology Stack

### Core ML & Data Science
- **Python 3.11**: Primary language
- **pandas & NumPy**: Data manipulation
- **scikit-learn**: Classical ML models (TF-IDF, Logistic Regression)
- **NLTK**: Text preprocessing and tokenization
- **Gensim**: Topic modeling (LDA, coherence)
- **VADER Sentiment**: Lexicon-based sentiment analysis
- **Sumy**: Extractive summarization algorithms

### API & Serving
- **FastAPI**: High-performance REST API framework
- **Pydantic**: Data validation and serialization
- **uvicorn**: ASGI server

### Dashboard & Visualization
- **Streamlit**: Interactive web applications
- **Plotly**: Interactive visualizations
- **Matplotlib & Seaborn**: Static plots

### Development & Deployment
- **Docker**: Containerization
- **Docker Compose**: Multi-container orchestration
- **pytest**: Testing framework
- **pytest-cov**: Code coverage
- **flake8 & black**: Code quality and formatting
- **GitHub Actions**: CI/CD automation

### Notebooks & Experimentation
- **marimo**: Reactive Python notebooks for course materials

## Project Structure

```
ML-and-DS-Project-Level/
├── data/
│   ├── raw/                    # Original data files
│   ├── interim/                # Intermediate processing
│   └── processed/              # Clean, ready-to-use data
│
├── marimo_notebooks/           # Educational notebooks
│   ├── 00_course_guide.py      # Sprint 0: Course overview
│   ├── 01_sprint_01_data_baseline.py  # Sprint 1: Data & baseline models
│   ├── 02_sprint_02_advanced_nlp.py   # Sprint 2: Advanced NLP
│   └── 03_sprint_03_prod_ui.py        # Sprint 3: Production & UI
│
├── src/
│   └── feedback_insights/      # Main Python package
│       ├── ingestion.py        # Data ingestion
│       ├── preprocessing.py    # Text preprocessing
│       ├── models.py           # Model training
│       ├── topic_model.py      # Topic modeling
│       ├── summarizer.py       # Text summarization
│       └── utils.py            # Utilities
│
├── api/
│   ├── app.py                  # FastAPI application
│   ├── routes/                 # API endpoints
│   └── schemas.py              # Pydantic models
│
├── dashboard/
│   ├── streamlit_app.py        # Streamlit dashboard
│   └── components/             # Reusable UI components
│
├── models/
│   └── checkpoints/            # Saved model artifacts
│
├── tests/
│   ├── unit/                   # Unit tests
│   └── integration/            # Integration tests
│
├── ops/
│   ├── Dockerfile.api          # API container
│   ├── Dockerfile.dashboard    # Dashboard container
│   └── docker-compose.yml      # Multi-container setup
│
├── docs/
│   ├── architecture.md         # System architecture
│   ├── runbook.md             # Deployment guide
│   └── troubleshooting.md     # Common issues
│
├── requirements.txt            # Python dependencies
├── PLAN.md                     # Detailed project plan
└── README.md                   # This file
```

## Quick Start

### Prerequisites

- Python 3.11+
- Docker & Docker Compose (for containerized deployment)
- Git

### Local Setup

1. **Clone the repository**
```bash
git clone https://github.com/mehalyna/ML-and-DS-Project-Level.git
cd ML-and-DS-Project-Level
```

2. **Create virtual environment**
```bash
python -m venv venv

# Windows
.\venv\Scripts\Activate.ps1

# Linux/Mac
source venv/bin/activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Download NLTK data**
```bash
python -c "import nltk; nltk.download('punkt_tab'); nltk.download('stopwords')"
```

### Running the Notebooks

Explore the educational notebooks with marimo:

```bash
# Course guide
marimo edit marimo_notebooks/00_course_guide.py

# Sprint 1: Data & baseline
marimo edit marimo_notebooks/01_sprint_01_data_baseline.py

# Sprint 2: Advanced NLP
marimo edit marimo_notebooks/02_sprint_02_advanced_nlp.py

# Sprint 3: Production & UI
marimo edit marimo_notebooks/03_sprint_03_prod_ui.py
```

### Running with Docker

1. **Build and start services**
```bash
docker-compose -f ops/docker-compose.yml up --build
```

2. **Access services**
- API: http://localhost:8000
- API Docs: http://localhost:8000/docs
- Dashboard: http://localhost:8501

3. **Stop services**
```bash
docker-compose -f ops/docker-compose.yml down
```

### Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest --cov=src --cov=api --cov-report=html

# View coverage report
open htmlcov/index.html
```

## Usage Examples

### API Example

```python
import requests

# Predict sentiment
response = requests.post(
    "http://localhost:8000/api/v1/predict/sentiment",
    json={"text": "Great product but shipping was slow"}
)

result = response.json()
print(f"Sentiment: {result['sentiment']}")
print(f"Confidence: {result['confidence']:.2%}")
```

### Python Package Example

```python
from src.feedback_insights.preprocessing import preprocess_text
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Preprocess text
text = "I love this product! Best purchase ever."
clean_text = preprocess_text(text)

# Analyze sentiment
vader = SentimentIntensityAnalyzer()
scores = vader.polarity_scores(clean_text)
print(f"Sentiment scores: {scores}")
```

## Development Workflow

### 6-Week Sprint Timeline

**Sprint 1 (Weeks 1-2): Data & Baseline**
- Data ingestion and cleaning
- Exploratory Data Analysis
- Baseline sentiment classifier
- Evaluation metrics

**Sprint 2 (Weeks 3-4): Advanced NLP**
- Ensemble sentiment model (VADER + TF-IDF)
- Gensim LDA topic modeling
- Extractive summarization (TextRank, LexRank, LSA)
- Model comparison

**Sprint 3 (Weeks 5-6): Production & UI**
- FastAPI REST endpoints
- Streamlit dashboard
- Docker containerization
- Unit and integration tests
- CI/CD pipeline
- Documentation

### Team Roles (8+ members)

1. **Project Manager**: Sprint planning, demos, stakeholder communication
2. **Data Engineer**: Ingestion scripts, data pipelines, quality checks
3. **NLP Engineer**: Text preprocessing, topic modeling, summarization
4. **ML Engineer**: Model training, evaluation, experiment tracking
5. **Backend Engineer**: FastAPI endpoints, model serving
6. **Frontend Engineer**: Streamlit dashboard, visualizations
7. **QA Engineer**: Testing, validation, quality assurance
8. **DevOps Engineer**: Docker, CI/CD, deployment automation

## Key Design Decisions

### Why Lightweight Libraries?

**Challenge**: Initial implementation used transformers (BERT, BART) but encountered PyTorch DLL errors on Windows with marimo.

**Solution**: Migrated to lightweight, production-ready alternatives:
- **VADER + TF-IDF ensemble** instead of fine-tuned BERT
- **Gensim LDA** instead of BERTopic
- **Sumy extractive algorithms** instead of BART/T5

**Benefits**:
- 10-20x faster inference
- No GPU requirements
- Cross-platform compatibility
- Smaller Docker images
- Easier deployment
- More interpretable results

### Architecture Principles

1. **Separation of Concerns**: Clear boundaries between data, models, API, and UI
2. **Modularity**: Reusable components and functions
3. **Testability**: High test coverage with isolated unit tests
4. **Configurability**: Environment-based configuration
5. **Observability**: Comprehensive logging and metrics
6. **Scalability**: Stateless API design for horizontal scaling

## Performance Metrics

### Model Performance (on sample dataset)
- **Sentiment Accuracy**: ~85-90% (ensemble approach)
- **Topic Coherence (C_v)**: >0.4 (validated by humans)
- **ROUGE-L Score**: >0.3 for summarization

### API Performance
- **Sentiment Prediction**: <100ms per request
- **Topic Extraction**: <200ms per request
- **Summarization**: <150ms per request
- **Throughput**: 50+ requests/second (single container)

### Resource Usage
- **API Container**: ~500MB RAM, <10% CPU
- **Dashboard Container**: ~300MB RAM
- **Model Files**: <100MB total

## Contributing

This is an educational project for a 6-week ML engineering course. Contributions, suggestions, and improvements are welcome!

### Development Setup

1. Install development dependencies:
```bash
pip install -r requirements-dev.txt
```

2. Run linting:
```bash
flake8 src/ api/ --max-line-length=120
black src/ api/ --check
```

3. Run tests before committing:
```bash
pytest tests/ -v
```

## Testing Strategy

### Test Coverage Targets
- Unit Tests: 80%+ coverage
- Integration Tests: All API endpoints
- E2E Tests: Critical user workflows

### CI/CD Pipeline
- Automated testing on every commit
- Linting and formatting checks
- Docker image builds
- Coverage reporting

## Troubleshooting

### Common Issues

**NLTK Data Missing**
```bash
python -c "import nltk; nltk.download('punkt_tab'); nltk.download('stopwords')"
```

**Port Already in Use**
```bash
# Change ports in docker-compose.yml or
docker-compose down
```

**Model Files Not Found**
- Ensure models are trained and saved in `models/checkpoints/`
- Run training notebooks first

See `docs/troubleshooting.md` for more details.

## Documentation

- **PLAN.md**: Detailed 6-week project plan
- **SPRINT2_MIGRATION.md**: Migration from transformers to lightweight libraries
- **docs/architecture.md**: System architecture and design
- **docs/runbook.md**: Deployment procedures
- **docs/troubleshooting.md**: Common issues and solutions

## License

This project is created for educational purposes as part of an ML engineering course.

## Acknowledgments

- Course participants and instructors
- Open-source library maintainers (FastAPI, Streamlit, Gensim, VADER)
- Public datasets used for training and validation

## Contact

**Repository**: https://github.com/mehalyna/ML-and-DS-Project-Level  
**Owner**: mehalyna

For questions or issues, please open a GitHub issue.

---

**Last Updated**: November 2025  
**Status**: Active Development (Sprint 3 - Production & UI)