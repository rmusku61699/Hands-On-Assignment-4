# Unsupervised Credit Card Fraud Detection with a PyOD AutoEncoder

Hands-On Assignment 4. A deep-learning **AutoEncoder** from [PyOD](https://github.com/yzhao062/pyod)
is trained **without labels** on anonymised credit card transactions. Each transaction's
*reconstruction error* is used as its anomaly score: transactions the network cannot rebuild
well are flagged as likely fraud.

![Results](outputs/results.png)

## Results (test set: 85,443 transactions, 148 frauds)

| Metric | Value |
|---|---|
| ROC-AUC | **0.940** |
| PR-AUC (average precision) | 0.047 (random baseline 0.0017, about 27x better) |
| Frauds caught in the top 1% riskiest transactions | 59 / 148 (39.9%) |
| Frauds caught in the top 5% riskiest transactions | **121 / 148 (81.8%)** |
| Default threshold (contamination = 0.17%) | 8 frauds caught, 118 false alarms |

The model ranks fraud well (ROC-AUC 0.94), but the default PyOD threshold, set from the
expected fraud rate, is too strict: most frauds score between 6 and 20 while the threshold
is 21.9. Using the score as a *risk ranking* for review queues is far more effective. See
`outputs/run_log.txt` for the full console output.

## Project structure

```
.
├── fraud_detection_autoencoder.py   # Main script (load → train → score → evaluate)
├── requirements.txt                 # Manifest of Python dependencies
├── data/                            # Put creditcard.csv here (not committed)
└── outputs/
    ├── results.png                  # Loss curve, score histogram, ROC and PR curves
    ├── confusion_matrix.png
    ├── metrics.json                 # All metrics + configuration
    └── run_log.txt                  # Full console output of the run
```

## How to run

1. Install Python 3.10+ and clone this repository.
2. Create a virtual environment and install the dependencies:
   ```bash
   python -m venv .venv
   source .venv/bin/activate        # Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   ```
3. Download `creditcard.csv` from the
   [Kaggle fraud detection dataset](https://www.kaggle.com/datasets/whenamancodes/fraud-detection)
   and save it as `data/creditcard.csv`.
4. Run the experiment:
   ```bash
   python fraud_detection_autoencoder.py --data data/creditcard.csv
   ```
   Optional flags: `--epochs 20`, `--batch-size 512`, `--output-dir outputs`.

Training takes about 35 seconds on a 2-core CPU; no GPU is needed.

## Method

| Step | Details |
|---|---|
| Data | 284,807 transactions, 31 columns (`Time`, `V1`–`V28` PCA features, `Amount`, `Class`), 492 frauds (0.17%) |
| Preprocessing | Drop `Time`; `log1p` + standard-scale `Amount` (fitted on train only) |
| Split | Stratified 70 / 30 train / test, `random_state=42` |
| Model | `pyod.models.auto_encoder.AutoEncoder`, layers 29 → 24 → 12 → 24 → 29, ReLU, batch norm, dropout 0.2 |
| Training | 10 epochs, batch 256, Adam (lr 0.001), MSE loss. Labels are **not** used |
| Scoring | `decision_function` = reconstruction error; `predict` uses the contamination-based threshold |
| Evaluation | ROC-AUC, PR-AUC, precision@n, confusion matrix, recall at top-k% |

### Why log-transform `Amount`?

`Amount` ranges from 0 to about 25,700. On the raw scale, a few very large legitimate purchases
produce huge reconstruction errors and crowd real fraud out of the top of the ranking. With `Time`
and raw `Amount`, ROC-AUC was 0.933 and the top-5% review caught 73.7% of frauds; after the change
these rose to 0.940 and 81.8%.

## References

- PyOD documentation: https://pyod.readthedocs.io/
- Packt sample notebook: [chapter1_pyod_autoencoder.ipynb](https://github.com/PacktPublishing/Deep-Learning-and-XAI-Techniques-for-Anomaly-Detection/blob/main/Chapter1/PyOD_autoencoder/chapter1_pyod_autoencoder.ipynb)
