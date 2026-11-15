"""Module 6b: final refit on 2010-2020 (declared AFTER seeing the v1 test result).

    python src/06b_refit.py

Transparency note for the write-up: model v1 (06_model.py, trained 2010-2018)
tied the volatility rule on the test set. Before looking at the test set again
we fixed ONE change: retrain with exactly the same settings and the same number
of trees on train + validation (2010-2020), so the model also learns from the
2020 COVID crash. Standard practice for a final model. Nothing else changes.
Both v1 and v2 are reported in module 7.

Output: score_model_v2 in preds.parquet, models/lgbm_v2.txt
"""
import json
import warnings

import lightgbm as lgb

from common import FEATURES, PREDS_PATH, ROOT, load_panel, load_preds

warnings.filterwarnings("ignore", message=".*eval_set.*")
MODEL_DIR = ROOT / "models"


def main():
    params = json.loads((MODEL_DIR / "lgbm_params.json").read_text())
    feats = params.pop("features")
    assert feats == FEATURES, "feature list changed since v1"

    panel, preds = load_panel(), load_preds()
    fit_rows = panel["split"].isin(["train", "val"])
    model = lgb.LGBMClassifier(**params)
    model.fit(panel.loc[fit_rows, FEATURES], panel.loc[fit_rows, "label"])

    preds["score_model_v2"] = model.predict_proba(panel[FEATURES])[:, 1]
    preds.to_parquet(PREDS_PATH, index=False)
    model.booster_.save_model(str(MODEL_DIR / "lgbm_v2.txt"))
    print(f"v2 trained on {fit_rows.sum():,} rows (2010-2020), "
          f"{params['n_estimators']} trees, leaves={params['num_leaves']}. "
          f"Scores saved as score_model_v2.")


if __name__ == "__main__":
    main()
