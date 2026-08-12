from pathlib import Path

# Project root = folder where this script is located
ROOT = Path(__file__).parent

# Folders to create
folders = [
    ".github/workflows",

    "agent",

    "api",

    "app/components",
    "app/pages",

    "data/raw",

    "docs",

    "genai",

    "models",

    "notebooks",

    "outputs/figures",
    "outputs/reports",

    "scripts",

    "src/business",
    "src/data",
    "src/explainability",
    "src/models",
    "src/monitoring",
    "src/prediction",
    "src/preprocessing",
    "src/recommendation",
    "src/risk",

    "tests",
]

# Files to create
files = [
    # Root
    ".env.example",
    ".gitignore",
    "README.md",
    "requirements.txt",
    "config.yaml",
    "Dockerfile",

    # GitHub
    ".github/workflows/ci.yml",

    # Agent
    "agent/__init__.py",
    "agent/graph.py",
    "agent/nodes.py",
    "agent/prompts.py",
    "agent/state.py",

    # API
    "api/__init__.py",
    "api/main.py",
    "api/schemas.py",

    # App
    "app/streamlit_app.py",

    "app/components/charts.py",
    "app/components/customer_profile.py",
    "app/components/kpi_cards.py",
    "app/components/recommendation_card.py",

    "app/pages/01_Executive_Dashboard.py",
    "app/pages/02_Customer_Risk.py",
    "app/pages/03_Retention_Recommendations.py",
    "app/pages/04_AI_Retention_Assistant.py",
    "app/pages/05_Model_Performance.py",

    # Data
    "data/raw/.gitkeep",

    # Docs
    "docs/business_logic.md",
    "docs/data_dictionary.md",
    "docs/project_flow.md",

    # GenAI
    "genai/__init__.py",
    "genai/llm_client.py",
    "genai/offer_generator.py",

    # Models
    "models/.gitkeep",

    # Notebooks
    "notebooks/day1_eda.ipynb",

    # Scripts
    "scripts/train.py",

    # Source
    "src/__init__.py",

    # Business
    "src/business/__init__.py",
    "src/business/revenue_at_risk.py",

    # Data processing
    "src/data/__init__.py",
    "src/data/loader.py",
    "src/data/cleaner.py",
    "src/data/feature_engineering.py",

    # Explainability
    "src/explainability/__init__.py",
    "src/explainability/shap_explainer.py",

    # Models
    "src/models/__init__.py",
    "src/models/logistic_regression.py",
    "src/models/random_forest.py",
    "src/models/xgboost_model.py",
    "src/models/lightgbm_model.py",
    "src/models/dnn_model.py",
    "src/models/evaluate.py",

    # Monitoring
    "src/monitoring/__init__.py",
    "src/monitoring/drift_monitor.py",

    # Prediction
    "src/prediction/__init__.py",
    "src/prediction/predictor.py",

    # Preprocessing
    "src/preprocessing/__init__.py",
    "src/preprocessing/pipeline.py",

    # Recommendation
    "src/recommendation/__init__.py",
    "src/recommendation/offer_engine.py",
    "src/recommendation/priority_engine.py",
    "src/recommendation/retention_rules.py",

    # Risk
    "src/risk/__init__.py",
    "src/risk/risk_engine.py",

    # Tests
    "tests/test_data.py",
    "tests/test_prediction.py",
    "tests/test_recommendation.py",
    "tests/test_risk.py",
    "tests/test_models.py",
    "tests/test_preprocessing.py",
    "tests/test_agent.py",
]


def create_structure():
    print("Creating project structure...\n")

    # Create folders
    for folder in folders:
        path = ROOT / folder
        path.mkdir(parents=True, exist_ok=True)

    # Create files only if they don't already exist
    created = 0
    skipped = 0

    for file in files:
        path = ROOT / file

        if path.exists():
            print(f"[SKIP]   {file}")
            skipped += 1
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.touch()
            print(f"[CREATE] {file}")
            created += 1

    print("\n--------------------------------")
    print("Project structure setup complete")
    print("--------------------------------")
    print(f"Created : {created} files")
    print(f"Skipped : {skipped} existing files")
    print("\nExisting files were NOT overwritten.")


if __name__ == "__main__":
    create_structure()