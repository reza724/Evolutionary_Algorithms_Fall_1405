"""ML sample: logistic regression predicts the TP/SL label class of the current bar; LabelModel trades it."""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import bench

FEATS = ["ret_1", "ret_24", "vol_24", "range", "qvol_z24", "taker_ratio", "prem", "oi_chg_1", "ob_imb1", "fund_z30d"]


class LogReg(bench.LabelModel):
    model_type, author = "ml", "sample"
    lookback = 1
    horizon, levels = 12, [1, 2]               # label setting: 12 bars, TP 1 % / 2 %, SL 1 %

    def fit(self, train):
        ok, y = self.targets(train)
        X = train.loc[ok, FEATS].fillna(0.0).to_numpy()
        self.pipe = make_pipeline(StandardScaler(), LogisticRegression(C=0.5, max_iter=1000)).fit(X, y)
        print(f"ml_logreg: {len(X):,} rows, train acc {self.pipe.score(X, y):.3f}")

    def predict(self, obs):
        x = np.nan_to_num([[obs[c] for c in FEATS]])
        return int(self.pipe.predict(x)[0])
