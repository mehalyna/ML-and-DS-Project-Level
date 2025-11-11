# Customer Feedback Insight Platform

Short summary. Build an end-to-end platform that ingests multi-channel customer feedback, extracts sentiment and topics, produces concise summaries, and surfaces actionable insights for product teams. The course has daily standups, three sprints, a team ≥ 8 people, and a 6-week timeline.

## 1. Minimum Viable Product (MVP) and success criteria

MVP. A reproducible pipeline that ingests sample feedback, runs preprocessing, produces sentiment labels and topic tags, returns an extractive summary for each item, and exposes predictions via a simple FastAPI endpoint plus an interactive Streamlit dashboard.
Success criteria. Automated ingestion working, baseline sentiment model with evaluation report, topic model output with coherent clusters, summarizer that produces readable summaries on held-out examples, API + dashboard that can demo a small dataset end-to-end.

## 2. Suggested technology stack

1. Data and modeling. Python, pandas, NumPy, scikit-learn, Hugging Face Transformers, sentence-transformers, BERTopic, NLTK or spaCy for tokenization.
2. Serving and UI. FastAPI for model endpoints, Streamlit for interactive demo.
3. Embeddings & retrieval. FAISS or simple in-memory index for similarity search.
4. DevOps. Docker, docker-compose, GitHub Actions for CI (unit tests + lint).
5. Experiment tracking. MLflow or Weights & Biases lightweight usage.
6. Storage. Local files / CSV for course datasets, optional SQLite for metadata.

## 3. Data sources and small dataset ideas

1. Public sources to seed tasks. Kaggle "customer reviews" datasets, Amazon reviews samples, Yelp reviews, or small in-house CSVs of simulated feedback.
2. Synthetic augmentation. Combine short CRM notes, support chat snippets, and short survey answers to form multi-channel inputs.
3. Required fields in each example. source, timestamp, customer_id (anonymized), text, optional metadata (product_id, rating).

## 4. 6-week timeline mapped to 3 sprints

Sprint cadence. Sprint 1 spans week 1–2, Sprint 2 week 3–4, Sprint 3 week 5–6. Daily status meetings every weekday, 15 minutes standup. Demo at the end of weeks 2, 4, and 6.

Sprint 1. Data & baseline. Week 1–2. Deliverables. Ingestion scripts for CSV/JSON, cleaned dataset snapshots, EDA notebook with key charts and data quality notes, baseline sentiment classifier (scikit-learn or small transformer), evaluation metrics and a short error analysis. Acceptance. Ingestion reproducible via a script. Baseline model trains within course hardware limits and produces a reproducible metrics table.

Sprint 2. Advanced NLP. Week 3–4. Deliverables. Fine-tuned transformer or improved classifier, topic modeling pipeline using BERTopic or LDA with interpretable topics, extractive summarizer (e.g., TextRank) and a simple abstractive prototype if compute allows, comparison notebook with metrics and confusion cases. Acceptance. Topic coherence examples validated by humans. Summaries judged as useful on blind samples.

Sprint 3. Production & UX. Week 5–6. Deliverables. FastAPI exposing sentiment/topic/summary endpoints, Streamlit dashboard for browsing feedback and aggregated insights, containerized app with Dockerfile(s), unit tests for key functions, final demo and project report. Acceptance. Deployable container that runs locally, API endpoints documented, dashboard shows aggregated KPIs and allows filtering.

## 5. Team roles and mapped tasks

1. Project manager. Roadmap, sprint planning, demos, stakeholder communication.
2. Data engineer. Ingestion scripts, cleaning routines, pipeline reproducibility.
3. NLP engineer. Tokenization, preprocessing, NER, topic modeling.
4. ML engineer. Model training, evaluation, experiment tracking, model packaging.
5. Backend engineer. FastAPI endpoints, model loading, inference code.
6. Frontend / dashboard engineer. Streamlit pages, filtering, visualization.
7. QA / data validator. Data quality checks, unit tests, test cases for API and UI.
8. DevOps. Dockerfiles, docker-compose, CI setup, simple deployment notes.

## 6. Daily standup template (3 quick items)

1. What I did yesterday.
2. What I will do today.
3. Blockers or help needed.

## 7. Example GitHub repo layout (starter)

```
/repo-root
├─ data
│  ├─ raw                         # original downloaded files (do not edit)
│  ├─ interim                     # intermediate files produced by ingestion
│  └─ processed                   # cleaned, ready-to-use datasets
├─ marimo_notebooks               # Marimo .py notebooks for course support and demos
│  ├─ 00_course_guide.py
│  ├─ 01_sprint_01_data_baseline.py
│  ├─ 02_sprint_02_advanced_nlp.py
│  └─ 03_sprint_03_prod_ui.py
├─ src
│  ├─ feedback_insights           # main Python package
│  │  ├─ __init__.py
│  │  ├─ ingestion.py
│  │  ├─ preprocessing.py
│  │  ├─ features.py
│  │  ├─ models.py                 # training and evaluation orchestration
│  │  ├─ topic_model.py
│  │  ├─ summarizer.py
│  │  └─ utils.py
│  └─ scripts                      # CLI helpers and one-off scripts
│     ├─ build_embeddings.py
│     ├─ run_evaluation.py
│     └─ export_artifacts.py
├─ api
│  ├─ app.py                       # FastAPI app entrypoint
│  ├─ routes
│  │  ├─ __init__.py
│  │  ├─ predict.py
│  │  └─ health.py
│  └─ schemas                      # pydantic models for requests/responses
├─ dashboard
│  ├─ streamlit_app.py
│  └─ components
├─ models
│  ├─ checkpoints                  # saved model weights (.pt/.bin)
│  └─ vector_index                  # FAISS / Milvus snapshots or embeddings
├─ artifacts
│  ├─ experiments                  # experiment logs, mlflow artifacts
│  └─ reports                      # EDA, model comparison, final report PDFs
├─ tests
│  ├─ unit
│  │  ├─ test_preprocessing.py
│  │  └─ test_models.py
│  └─ integration
│     └─ test_api_endpoints.py
├─ ops
│  ├─ Dockerfile
│  ├─ docker-compose.yml
│  ├─ ci
│  │  └─ github-actions.yml
│  └─ k8s                          # optional Kubernetes manifests for later
├─ docs
│  ├─ architecture.md
│  ├─ runbook.md
│  └─ onboarding.md
├─ requirements.txt
├─ requirements-dev.txt
├─ Makefile
├─ PLAN.md
└─ README.md
```

## 8. Starter issue list for Sprint 1 (examples)

1. Create ingestion script that reads CSV / JSON and writes cleaned CSV. Acceptance. Script runs and writes `data/processed/clean.csv`.
2. Explore dataset and produce 10–15 EDA charts in a notebook. Acceptance. Notebook committed with key findings.
3. Implement baseline sentiment model and evaluation. Acceptance. Notebook and saved model artifact with metrics.
4. Add unit tests for basic preprocessing behaviors. Acceptance. CI passes tests.

## 9. Evaluation rubric and weights

1. Data & EDA quality. 20% . Depth of analysis, handling of missing data, realistic preprocessing.
2. Model performance and robustness. 30% . Metrics, baseline→improved model, calibration and error analysis.
3. Usability of dashboard and API. 20% . Clarity, filtering, latency for demo.
4. Code quality, tests, reproducibility. 15% . Structure, tests, Docker.
5. Teamwork and demo. 15% . Daily standups, sprint demos, clear final presentation.

## 10. Grading acceptance examples

1. Pass. Ingestion script reproduces dataset. Baseline model achieves reasonable metrics on validation set and is documented. Dashboard loads and shows aggregated sentiment.
2. Distinction. Advanced models outperform baseline significantly. Topics coherent and actionable. API + dashboard are polished and containerized. Tests and CI present.

## 11. Quick CI / Docker notes

1. CI pipeline. Run unit tests, run lint, run a tiny smoke test that starts the FastAPI app and queries `/health` endpoint.
2. Docker. Build two images if needed. One image for model artifact + API, one for Streamlit. Use docker-compose to orchestrate both locally.

## 12. Demo checklist for final presentation

1. Show ingestion running on sample file and produced cleaned snapshot.
2. Display EDA highlights and one surprising insight.
3. Run baseline model and improved model predictions on same example, show metrics.
4. Show topic clusters and representative examples.
5. Display summarizer output for 2–3 items.
6. Hit FastAPI endpoint live and show dashboard filtering by product and sentiment.
7. Describe limitations, next steps, and lessons learned.

## 13. Optional stretch tasks and extensions

1. Add NER to surface common entities like product features and competitors.
2. Build an offline dashboard that aggregates trends per product and time window.
3. Implement simple active learning loop to surface uncertain examples for human labeling.
4. Add lightweight model monitoring and drift detection demonstration.
