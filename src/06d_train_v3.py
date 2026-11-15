"""Module 6d: train model v3 (chosen on validation in 06c) and save it.

    python src/06d_train_v3.py

Trained on 2010-2018, early stopping on 2019-2020 -- exactly as selected in 06c.
Writes models/lgbm_v3.txt + lgbm_v3.json and score_model_v3 into preds.parquet.
Does not print any test results.
"""
import json
import warnings

import lightgbm as lgb
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score

from common import PREDS_PATH, load_panel, load_preds
from model_v3 import (META_PATH, MODEL_PATH, V3_FEATURES, V3_PARAMS, add_v3_features,
                      logit, predict_v3, vol_prob)

warnings.filterwarnings("ignore")


def main():
    df = add_v3_features(load_panel())
    preds = load_preds()
    tr, va = df["split"] == "train", df["split"] == "val"
    y = df["label"]

    lv = logit(vol_prob(df["vol_20"])).reshape(-1, 1)
    cal = LogisticRegression().fit(lv[tr], y[tr])
    a, b = float(cal.coef_[0][0]), float(cal.intercept_[0])
    init = a * lv[:, 0] + b

    m = lgb.LGBMClassifier(**V3_PARAMS, n_estimators=2000)
    m.fit(df.loc[tr, V3_FEATURES], y[tr], init_score=init[tr],
          eval_set=[(df.loc[va, V3_FEATURES], y[va])], eval_init_score=[init[va]],
          eval_metric="average_precision", callbacks=[lgb.early_stopping(100, verbose=False)])
    m.booster_.save_model(str(MODEL_PATH), num_iteration=m.best_iteration_)
    META_PATH.write_text(json.dumps({"cal_a": a, "cal_b": b, "trees": int(m.best_iteration_),
                                     "features": V3_FEATURES, "params": V3_PARAMS}, indent=2))

    preds["score_model_v3"] = predict_v3(df)
    preds.to_parquet(PREDS_PATH, index=False)
    v = preds[va.to_numpy()]
    print(f"v3 saved: {m.best_iteration_} trees. Validation PR-AUC "
          f"{average_precision_score(v['label'], v['score_model_v3']):.3f} "
          f"(volatility rule {average_precision_score(v['label'], v['score_vol']):.3f})")


if __name__ == "__main__":
    main()
