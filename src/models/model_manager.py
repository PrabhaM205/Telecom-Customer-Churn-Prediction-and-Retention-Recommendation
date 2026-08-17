import os
import json
import shutil
from datetime import datetime


class ModelManager:
    """
    Production model manager.

    Responsibilities:
        - Track model versions
        - Track production model
        - Backup production model
        - Register models
        - Compare old and new models
        - Promote better models
        - Reject worse models
    """

    def __init__(self):

        # ====================================================
        # PROJECT ROOT
        # ====================================================

        self.project_root = os.path.abspath(
            os.path.join(
                os.path.dirname(__file__),
                "..",
                ".."
            )
        )

        # ====================================================
        # ARTIFACTS
        # ====================================================

        self.artifacts_dir = os.path.join(
            self.project_root,
            "artifacts"
        )

        os.makedirs(
            self.artifacts_dir,
            exist_ok=True
        )

        # ====================================================
        # MODEL PATH
        # ====================================================

        self.production_model_path = os.path.join(
            self.artifacts_dir,
            "xgboost_churn_model.joblib"
        )

        self.backup_model_path = os.path.join(
            self.artifacts_dir,
            "xgboost_churn_model_old.joblib"
        )

        # ====================================================
        # REGISTRY
        # ====================================================

        self.registry_path = os.path.join(
            self.artifacts_dir,
            "model_registry.json"
        )

        # ====================================================
        # INITIALIZE
        # ====================================================

        self._initialize_registry()

    # ========================================================
    # INITIALIZE REGISTRY
    # ========================================================

    def _initialize_registry(self):

        if not os.path.exists(
            self.registry_path
        ):

            registry = {
                "production_version": None,
                "models": []
            }

            self._save_registry(
                registry
            )

    # ========================================================
    # LOAD REGISTRY
    # ========================================================

    def _load_registry(self):

        if not os.path.exists(
            self.registry_path
        ):

            return {
                "production_version": None,
                "models": []
            }

        with open(
            self.registry_path,
            "r"
        ) as file:

            return json.load(file)

    # ========================================================
    # SAVE REGISTRY
    # ========================================================

    def _save_registry(
        self,
        registry
    ):

        with open(
            self.registry_path,
            "w"
        ) as file:

            json.dump(
                registry,
                file,
                indent=4
            )

    # ========================================================
    # GET CURRENT PRODUCTION VERSION
    # ========================================================

    def get_current_version(self):

        registry = self._load_registry()

        return registry.get(
            "production_version"
        )

    # ========================================================
    # GET CURRENT PRODUCTION MODEL
    # ========================================================

    def get_current_model(self):

        registry = self._load_registry()

        current_version = registry.get(
            "production_version"
        )

        if current_version is None:
            return None

        for model in registry.get(
            "models",
            []
        ):

            if model.get(
                "version"
            ) == current_version:

                return model

        return None

    # ========================================================
    # GET NEXT VERSION
    # ========================================================

    def get_next_version(self):

        registry = self._load_registry()

        versions = []

        for model in registry.get(
            "models",
            []
        ):

            version = model.get(
                "version",
                "v0"
            )

            try:

                number = int(
                    version.replace(
                        "v",
                        ""
                    )
                )

                versions.append(
                    number
                )

            except ValueError:

                continue

        if not versions:

            return "v1"

        return (
            f"v{max(versions) + 1}"
        )

    # ========================================================
    # BACKUP PRODUCTION MODEL
    # ========================================================

    def backup_production_model(self):

        if not os.path.exists(
            self.production_model_path
        ):

            print(
                "\nNo production model found."
            )

            return False

        shutil.copy2(
            self.production_model_path,
            self.backup_model_path
        )

        print(
            "\nExisting production model backed up:"
        )

        print(
            self.backup_model_path
        )

        return True

    # ========================================================
    # REGISTER MODEL
    # ========================================================

    def register_model(
        self,
        version,
        accuracy,
        precision,
        recall,
        f1_score,
        roc_auc,
        status
    ):

        registry = self._load_registry()

        model_record = {

            "version":
                version,

            "model_name":
                "XGBoost",

            "accuracy":
                round(
                    float(accuracy),
                    4
                ),

            "precision":
                round(
                    float(precision),
                    4
                ),

            "recall":
                round(
                    float(recall),
                    4
                ),

            "f1_score":
                round(
                    float(f1_score),
                    4
                ),

            "roc_auc":
                round(
                    float(roc_auc),
                    4
                ),

            "status":
                status,

            "model_path":
                self.production_model_path,

            "created_at":
                datetime.now().isoformat()
        }

        registry.setdefault(
            "models",
            []
        ).append(
            model_record
        )

        if status == "production":

            registry[
                "production_version"
            ] = version

        self._save_registry(
            registry
        )

        print(
            f"\nModel {version} registered "
            f"with status: {status}"
        )

    # ========================================================
    # COMPARE MODELS
    # ========================================================

    def compare_models(
        self,
        old_metrics,
        new_metrics
    ):

        old_roc_auc = float(
            old_metrics["roc_auc"]
        )

        new_roc_auc = float(
            new_metrics["roc_auc"]
        )

        old_accuracy = float(
            old_metrics["accuracy"]
        )

        new_accuracy = float(
            new_metrics["accuracy"]
        )

        print(
            "\n"
            + "=" * 60
        )

        print(
            "MODEL COMPARISON"
        )

        print(
            "=" * 60
        )

        print(
            "\nOLD PRODUCTION MODEL"
        )

        print(
            f"Accuracy : "
            f"{old_accuracy:.4f}"
        )

        print(
            f"ROC-AUC  : "
            f"{old_roc_auc:.4f}"
        )

        print(
            "\nNEW RETRAINED MODEL"
        )

        print(
            f"Accuracy : "
            f"{new_accuracy:.4f}"
        )

        print(
            f"ROC-AUC  : "
            f"{new_roc_auc:.4f}"
        )

        # ----------------------------------------------------
        # Primary metric = ROC-AUC
        # ----------------------------------------------------

        if new_roc_auc > old_roc_auc:

            decision = "PROMOTE"

        else:

            decision = "REJECT"

        print(
            "\nDecision:"
        )

        print(
            decision
        )

        return decision

    # ========================================================
    # PROMOTE NEW MODEL
    # ========================================================

    def promote_new_model(
        self,
        version,
        metrics
    ):

        registry = self._load_registry()

        current_version = (
            registry.get(
                "production_version"
            )
        )

        # ----------------------------------------------------
        # Mark previous production model
        # ----------------------------------------------------

        for model in registry.get(
            "models",
            []
        ):

            if (
                model.get("version")
                == current_version
            ):

                model["status"] = (
                    "previous_production"
                )

        # ----------------------------------------------------
        # Register new production model
        # ----------------------------------------------------

        model_record = {

            "version":
                version,

            "model_name":
                "XGBoost",

            "accuracy":
                round(
                    float(metrics["accuracy"]),
                    4
                ),

            "precision":
                round(
                    float(metrics["precision"]),
                    4
                ),

            "recall":
                round(
                    float(metrics["recall"]),
                    4
                ),

            "f1_score":
                round(
                    float(metrics["f1_score"]),
                    4
                ),

            "roc_auc":
                round(
                    float(metrics["roc_auc"]),
                    4
                ),

            "status":
                "production",

            "model_path":
                self.production_model_path,

            "created_at":
                datetime.now().isoformat()
        }

        registry.setdefault(
            "models",
            []
        ).append(
            model_record
        )

        registry[
            "production_version"
        ] = version

        self._save_registry(
            registry
        )

        print(
            "\n"
            + "=" * 60
        )

        print(
            "NEW MODEL PROMOTED"
        )

        print(
            "=" * 60
        )

        print(
            f"\nProduction model: "
            f"{version}"
        )

    # ========================================================
    # REJECT NEW MODEL
    # ========================================================

    def reject_new_model(
        self,
        version,
        metrics
    ):

        self.register_model(

            version=version,

            accuracy=metrics["accuracy"],

            precision=metrics["precision"],

            recall=metrics["recall"],

            f1_score=metrics["f1_score"],

            roc_auc=metrics["roc_auc"],

            status="rejected"
        )

        print(
            "\n"
            + "=" * 60
        )

        print(
            "NEW MODEL REJECTED"
        )

        print(
            "=" * 60
        )

        print(
            "\nExisting production model "
            "will remain active."
        )

    # ========================================================
    # DISPLAY HISTORY
    # ========================================================

    def display_history(self):

        registry = self._load_registry()

        print(
            "\n"
            + "=" * 60
        )

        print(
            "MODEL HISTORY"
        )

        print(
            "=" * 60
        )

        models = registry.get(
            "models",
            []
        )

        if not models:

            print(
                "\nNo model history found."
            )

            return

        for model in models:

            print(
                "\n"
                + "-" * 60
            )

            print(
                f"Version   : "
                f"{model.get('version')}"
            )

            print(
                f"Accuracy  : "
                f"{model.get('accuracy')}"
            )

            print(
                f"Precision : "
                f"{model.get('precision')}"
            )

            print(
                f"Recall    : "
                f"{model.get('recall')}"
            )

            print(
                f"F1 Score  : "
                f"{model.get('f1_score')}"
            )

            print(
                f"ROC-AUC   : "
                f"{model.get('roc_auc')}"
            )

            print(
                f"Status    : "
                f"{model.get('status')}"
            )

            print(
                f"Created   : "
                f"{model.get('created_at')}"
            )


# ============================================================
# DIRECT EXECUTION
# ============================================================

if __name__ == "__main__":

    print(
        "=" * 60
    )

    print(
        "MODEL MANAGER"
    )

    print(
        "=" * 60
    )

    manager = ModelManager()

    print(
        "\nModel registry:"
    )

    print(
        manager.registry_path
    )

    print(
        "\nCurrent production version:"
    )

    print(
        manager.get_current_version()
    )

    manager.display_history()