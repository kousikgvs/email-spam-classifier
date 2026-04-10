import logging

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import MultinomialNB

logger = logging.getLogger(__name__)

CANDIDATE_MODELS = {
    "naive_bayes": MultinomialNB(),
    "logistic_regression": LogisticRegression(max_iter=2000, n_jobs=-1),
    "random_forest": RandomForestClassifier(n_estimators=250, random_state=42, n_jobs=-1),
}

PARAM_GRIDS = {
    "naive_bayes": {"alpha": [0.1, 0.3, 0.5, 1.0]},
    "logistic_regression": {"C": [0.1, 0.5, 1.0, 2.0], "solver": ["liblinear", "lbfgs"]},
    "random_forest": {
        "n_estimators": [200, 300],
        "max_depth": [None, 20],
        "min_samples_split": [2, 5],
    },
}


def select_best_model(X_train_vec, y_train) -> tuple[str, object]:
    results = []
    for name, model in CANDIDATE_MODELS.items():
        model.fit(X_train_vec, y_train)
        preds = model.predict(X_train_vec)
        results.append(
            {
                "model": name,
                "f1": f1_score(y_train, preds),
                "accuracy": accuracy_score(y_train, preds),
            }
        )
    results_df = pd.DataFrame(results).sort_values(by=["f1", "accuracy"], ascending=False)
    logger.info("Model comparison:\n%s", results_df.to_string(index=False))
    return results_df.iloc[0]["model"]


def tune_model(model_name: str, X_train_vec, y_train, cv: int = 3) -> object:
    estimator = {
        "naive_bayes": MultinomialNB(),
        "logistic_regression": LogisticRegression(max_iter=3000, n_jobs=-1),
        "random_forest": RandomForestClassifier(random_state=42, n_jobs=-1),
    }[model_name]

    grid = GridSearchCV(
        estimator=estimator,
        param_grid=PARAM_GRIDS[model_name],
        scoring="f1",
        cv=cv,
        n_jobs=-1,
        verbose=1,
    )
    grid.fit(X_train_vec, y_train)
    logger.info("Best params for %s: %s", model_name, grid.best_params_)
    return grid.best_estimator_
