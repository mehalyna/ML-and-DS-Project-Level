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
    # Customer Feedback Insight Platform - Course Guide

    Welcome to the **Customer Feedback Insight Platform** project! This interactive guide provides an overview of the 6-week course structure, deliverables, and key milestones.

    ## Project Overview

    Build an end-to-end platform that:
    - Ingests multi-channel customer feedback
    - Extracts sentiment and topics
    - Produces concise summaries
    - Surfaces actionable insights for product teams

    **Team Size:** ≥ 8 people
    **Timeline:** 6 weeks (3 sprints)
    **Daily Standups:** Every weekday (15 minutes)
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## Minimum Viable Product (MVP)

    A reproducible pipeline that:
    1. Ingests sample feedback
    2. Runs preprocessing
    3. Produces sentiment labels and topic tags
    4. Returns extractive summaries for each item
    5. Exposes predictions via FastAPI endpoint
    6. Provides interactive Streamlit dashboard

    ### Success Criteria

    - Automated ingestion working
    - Baseline sentiment model with evaluation report
    - Topic model output with coherent clusters
    - Summarizer producing readable summaries on held-out examples
    - API + dashboard demonstrating end-to-end workflow
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## Technology Stack

    | Category | Technologies |
    |----------|-------------|
    | **Data & Modeling** | Python, pandas, NumPy, scikit-learn, Hugging Face Transformers, sentence-transformers, BERTopic, NLTK/spaCy |
    | **Serving & UI** | FastAPI, Streamlit |
    | **Embeddings & Retrieval** | FAISS or in-memory index |
    | **DevOps** | Docker, docker-compose, GitHub Actions |
    | **Experiment Tracking** | MLflow or Weights & Biases |
    | **Storage** | Local files/CSV, optional SQLite |
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## Sprint Timeline

    The course is organized into **3 sprints** over **6 weeks**, with demos at the end of weeks 2, 4, and 6.
    """)
    return


@app.cell
def _():
    import pandas as pd

    sprint_data = {
        "Sprint": ["Sprint 1", "Sprint 2", "Sprint 3"],
        "Weeks": ["Week 1-2", "Week 3-4", "Week 5-6"],
        "Focus": ["Data & Baseline", "Advanced NLP", "Production & UX"],
        "Key Deliverables": [
            "Ingestion scripts, EDA notebook, baseline sentiment classifier",
            "Fine-tuned models, topic modeling, summarization",
            "FastAPI endpoints, Streamlit dashboard, Docker containers"
        ]
    }

    sprint_df = pd.DataFrame(sprint_data)
    return pd, sprint_df


@app.cell
def _(mo, sprint_df):
    mo.ui.table(sprint_df, selection=None)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## Sprint 1: Data & Baseline (Weeks 1-2)

    ### Deliverables

    - **Ingestion scripts** for CSV/JSON
    - **Cleaned dataset snapshots**
    - **EDA notebook** with key charts and data quality notes
    - **Baseline sentiment classifier** (scikit-learn or small transformer)
    - **Evaluation metrics** and error analysis

    ### Acceptance Criteria

    - Ingestion reproducible via script
    - Baseline model trains within hardware limits
    - Produces reproducible metrics table

    ### Example Issues

    1. Create ingestion script that reads CSV/JSON and writes cleaned CSV
    2. Explore dataset and produce 10-15 EDA charts
    3. Implement baseline sentiment model and evaluation
    4. Add unit tests for basic preprocessing behaviors
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## Sprint 2: Advanced NLP (Weeks 3-4)

    ### Deliverables

    - **Fine-tuned transformer** or improved classifier
    - **Topic modeling pipeline** using BERTopic or LDA
    - **Extractive summarizer** (e.g., TextRank)
    - Simple **abstractive prototype** (if compute allows)
    - **Comparison notebook** with metrics and confusion cases

    ### Acceptance Criteria

    - Topic coherence examples validated by humans
    - Summaries judged as useful on blind samples

    ### Key Components

    - **Sentiment Analysis:** Advanced transformer-based models
    - **Topic Modeling:** Coherent and interpretable topic clusters
    - **Summarization:** Both extractive and abstractive approaches
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## Sprint 3: Production & UX (Weeks 5-6)

    ### Deliverables

    - **FastAPI** exposing sentiment/topic/summary endpoints
    - **Streamlit dashboard** for browsing feedback and aggregated insights
    - **Containerized app** with Dockerfile(s)
    - **Unit tests** for key functions
    - **Final demo** and project report

    ### Acceptance Criteria

    - Deployable container that runs locally
    - API endpoints documented
    - Dashboard shows aggregated KPIs and allows filtering

    ### Production Components

    - RESTful API with proper error handling
    - Interactive dashboard with filtering capabilities
    - Docker containerization for easy deployment
    - Comprehensive testing suite
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## Team Roles and Responsibilities

    The project requires a team of **≥ 8 people** with the following roles:
    """)
    return


@app.cell
def _(pd):
    roles_data = {
        "Role": [
            "Project Manager",
            "Data Engineer",
            "NLP Engineer",
            "ML Engineer",
            "Backend Engineer",
            "Frontend/Dashboard Engineer",
            "QA/Data Validator",
            "DevOps Engineer"
        ],
        "Key Responsibilities": [
            "Roadmap, sprint planning, demos, stakeholder communication",
            "Ingestion scripts, cleaning routines, pipeline reproducibility",
            "Tokenization, preprocessing, NER, topic modeling",
            "Model training, evaluation, experiment tracking, packaging",
            "FastAPI endpoints, model loading, inference code",
            "Streamlit pages, filtering, visualization",
            "Data quality checks, unit tests, test cases for API and UI",
            "Dockerfiles, docker-compose, CI setup, deployment notes"
        ]
    }

    roles_df = pd.DataFrame(roles_data)
    return (roles_df,)


@app.cell
def _(mo, roles_df):
    mo.ui.table(roles_df, selection=None)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## Project Repository Structure

    ```
    /repo-root
    ├─ data
    │  ├─ raw                         # original downloaded files
    │  ├─ interim                     # intermediate files
    │  └─ processed                   # cleaned, ready-to-use datasets
    ├─ marimo_notebooks               # Marimo notebooks for demos
    │  ├─ 00_course_guide.py
    │  ├─ 01_sprint_01_data_baseline.py
    │  ├─ 02_sprint_02_advanced_nlp.py
    │  └─ 03_sprint_03_prod_ui.py
    ├─ src
    │  ├─ feedback_insights           # main Python package
    │  └─ scripts                      # CLI helpers
    ├─ api                             # FastAPI application
    ├─ dashboard                       # Streamlit dashboard
    ├─ models                          # saved model artifacts
    ├─ artifacts                       # experiments and reports
    ├─ tests                           # unit and integration tests
    ├─ ops                             # Docker and CI/CD
    └─ docs                            # documentation
    ```
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## Daily Standup Template

    Every weekday, each team member answers these **3 quick items** (15 minutes total):

    1. **What I did yesterday**
    2. **What I will do today**
    3. **Blockers or help needed**

    Keep it concise and focused on progress and impediments!
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## Evaluation Rubric

    Your project will be graded based on the following criteria:
    """)
    return


@app.cell
def _(pd):
    evaluation_data = {
        "Category": [
            "Data & EDA Quality",
            "Model Performance & Robustness",
            "Usability of Dashboard & API",
            "Code Quality, Tests, Reproducibility",
            "Teamwork & Demo"
        ],
        "Weight": ["20%", "30%", "20%", "15%", "15%"],
        "Focus Areas": [
            "Depth of analysis, handling missing data, realistic preprocessing",
            "Metrics, baseline→improved model, calibration, error analysis",
            "Clarity, filtering, latency for demo",
            "Structure, tests, Docker",
            "Daily standups, sprint demos, clear final presentation"
        ]
    }

    evaluation_df = pd.DataFrame(evaluation_data)
    return (evaluation_df,)


@app.cell
def _(evaluation_df, mo):
    mo.ui.table(evaluation_df, selection=None)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## Grading Acceptance Examples

    ### Pass Level

    - Ingestion script reproduces dataset
    - Baseline model achieves reasonable metrics on validation set and is documented
    - Dashboard loads and shows aggregated sentiment

    ### Distinction Level

    - Advanced models outperform baseline significantly
    - Topics coherent and actionable
    - API + dashboard are polished and containerized
    - Tests and CI present
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## Final Demo Checklist

    Your final presentation should cover:

    1. Show ingestion running on sample file and produced cleaned snapshot
    2. Display EDA highlights and one surprising insight
    3. Run baseline model and improved model predictions on same example, show metrics
    4. Show topic clusters and representative examples
    5. Display summarizer output for 2-3 items
    6. Hit FastAPI endpoint live and show dashboard filtering by product and sentiment
    7. Describe limitations, next steps, and lessons learned
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## Optional Stretch Tasks

    If you have extra time and want to go above and beyond:

    - **Named Entity Recognition (NER):** Surface common entities like product features and competitors
    - **Offline Dashboard:** Aggregate trends per product and time window
    - **Active Learning Loop:** Surface uncertain examples for human labeling
    - **Model Monitoring:** Add lightweight drift detection demonstration
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## CI/CD and Docker Notes

    ### CI Pipeline

    - Run unit tests
    - Run lint
    - Run smoke test that starts FastAPI app and queries `/health` endpoint

    ### Docker Strategy

    Build two images if needed:
    - **Image 1:** Model artifact + API
    - **Image 2:** Streamlit dashboard

    Use `docker-compose` to orchestrate both locally.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## Data Sources

    ### Public Sources to Seed Tasks

    - Kaggle "customer reviews" datasets
    - Amazon reviews samples
    - Yelp reviews
    - Small in-house CSVs of simulated feedback

    ### Required Fields in Each Example

    - `source`
    - `timestamp`
    - `customer_id` (anonymized)
    - `text`
    - Optional: `product_id`, `rating`

    ### Synthetic Augmentation

    Combine short CRM notes, support chat snippets, and short survey answers to form multi-channel inputs.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## Getting Started

    Ready to begin? Here's your action plan:

    1. **Week 0:** Set up development environment, clone repo, review PLAN.md
    2. **Week 1:** Start Sprint 1 - Focus on data ingestion and EDA
    3. **Keep the rhythm:** Daily standups, regular commits, sprint demos
    4. **Collaborate:** Use GitHub issues, pull requests, and code reviews
    5. **Document:** Keep notes, update README, write clear commit messages

    ### Next Steps

    - Review the other marimo notebooks:
      - `01_sprint_01_data_baseline.py` - Sprint 1 detailed guide
      - `02_sprint_02_advanced_nlp.py` - Sprint 2 detailed guide
      - `03_sprint_03_prod_ui.py` - Sprint 3 detailed guide

    **Good luck with your project! **
    """)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
