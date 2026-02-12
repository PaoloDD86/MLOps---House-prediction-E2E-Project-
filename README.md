Housing Regression MLE is an end-to-end machine learning pipeline for predicting housing prices using XGBoost. The project follows ML engineering best practices with modular pipelines, experiment tracking via MLflow, containerization, AWS cloud deployment, and comprehensive testing. The system includes both a REST API and a Streamlit dashboard for interactive predictions.

Architecture
The codebase is organized into distinct pipelines following the flow: Load → Preprocess → Feature Engineering → Train → Tune → Evaluate → Inference → Batch → Serve

Core Modules
src/feature_pipeline/: Data loading, preprocessing, and feature engineering

load.py: Time-aware data splitting (train <2020, eval 2020-21, holdout ≥2022)
preprocess.py: City normalization, deduplication, outlier removal
feature_engineering.py: Date features, frequency encoding (zipcode), target encoding (city_full)
src/training_pipeline/: Model training and hyperparameter optimization

train.py: Baseline XGBoost training with configurable parameters
tune.py: Optuna-based hyperparameter tuning with MLflow integration
eval.py: Model evaluation and metrics calculation
src/inference_pipeline/: Production inference

inference.py: Applies same preprocessing/encoding transformations using saved encoders
src/batch/: Batch prediction processing

run_monthly.py: Generates monthly predictions on holdout data
src/api/: FastAPI web service

main.py: REST API with S3 integration, health checks, prediction endpoints, and batch processing
Web Applications
app.py: Streamlit dashboard for interactive housing price predictions
Real-time predictions via FastAPI integration
Interactive filtering by year, month, and region
Visualization of predictions vs actuals with metrics (MAE, RMSE, % Error)
Yearly trend analysis with highlighted selected periods
Cloud Infrastructure & Deployment
AWS S3 Integration: Data and model storage in housing-regression-data bucket
Amazon ECR: Container registry for Docker images
Amazon ECS: Container orchestration with Fargate
Application Load Balancer: Traffic distribution and routing
CI/CD Pipeline: Automated deployment via GitHub Actions
ECS Services:
housing-api-service: FastAPI backend (port 8000, 1024 CPU, 3072 MB memory)
housing-streamlit-service: Streamlit dashboard (port 8501, 512 CPU, 1024 MB memory)
Data Leakage Prevention
The project implements strict data leakage prevention:

Time-based splits (not random)
Encoders fitted only on training data
Leakage-prone columns dropped before training
Schema alignment enforced between train/eval/inference
