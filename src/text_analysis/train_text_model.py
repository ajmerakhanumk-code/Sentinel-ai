"""
Sentinel AI - Text Sentiment Classifier Training & Evaluation Pipeline
Academic reproducible training module using TF-IDF + Logistic Regression.
Evaluates on held-out test data and serializes model artifacts.
"""

import os
import json
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix
)

import config

def get_sentiment_corpus() -> pd.DataFrame:
    """
    Returns a comprehensive, balanced corpus of labelled sentences covering Positive, Negative,
    and Neutral sentiments across technical, conversational, customer, and academic domains.
    """
    positives = [
        "The system performance exceeded our expectations significantly.",
        "I am absolutely thrilled with the exceptional accuracy and responsiveness.",
        "This is by far the most reliable and efficient implementation we have tested.",
        "Outstanding results, excellent user experience and very clean code.",
        "I love how intuitive and polished this command center interface looks.",
        "The algorithm operates flawlessly and solved our core engineering bottleneck.",
        "Great job on meeting every milestone ahead of schedule with top quality.",
        "The user feedback has been overwhelmingly positive and encouraging.",
        "Incredible breakthrough in multi-modal classification speed and accuracy.",
        "We are very happy with this decision and look forward to scaling it.",
        "The team provided wonderful support and prompt solutions to all issues.",
        "Remarkable improvement in inference latency and model stability.",
        "Everything is running smoothly and all diagnostic indicators are green.",
        "Highly recommended solution for anyone needing dependable AI monitoring.",
        "I feel great about our project progress and team dedication.",
        "Superb design, elegant visual aesthetics and seamless user interaction.",
        "The live demo worked perfectly on the first attempt with no errors.",
        "Fantastic engineering craftsmanship demonstrated across all project components.",
        "We achieved high precision and robust real-time prediction performance.",
        "The documentation is clear, thorough, and exceptionally helpful.",
        "This tool has greatly streamlined our daily intelligence workflow.",
        "Delighted with the rapid turnaround and high quality of this deliverable.",
        "Very pleasant surprise to see CPU memory usage remain so remarkably low.",
        "The facial analysis module detected subtle cues with impressive fidelity.",
        "Our evaluation metrics showed solid gains across all tested benchmarks.",
        "The feedback was complimentary and highlighted the project's innovation.",
        "Super excited to deploy this multimodal framework into production.",
        "Brilliant architecture that balances performance and lightweight compute.",
        "I am confident and optimistic about the outcome of our academic review.",
        "Clean, structured, and genuinely reliable software engineering.",
        "I am genuinely impressed by how smoothly this whole application runs.",
        "What a wonderful achievement, congratulations to everyone involved.",
        "The predictions are extremely accurate and match our manual inspections.",
        "Fast, reliable, intuitive, and remarkably lightweight on system memory.",
        "The visual design is beautiful and conveys professionalism immediately.",
        "I am so glad we chose this approach, the results speak for themselves.",
        "Splendid performance under heavy continuous inference workloads.",
        "Tremendous progress made in resolving all outstanding project issues.",
        "It is a true pleasure to interact with such a well-engineered tool.",
        "The classification confidence is consistently strong and trustworthy.",
        "A truly remarkable and satisfying demonstration of multimodal AI.",
        "Very impressed with the responsiveness of the webcam and text engine.",
        "This is an exemplary implementation of multimodal sentiment analysis.",
        "The dashboard analytics provide valuable and clear actionable insights.",
        "Flawless execution during live testing on Windows 11 hardware.",
        "We experienced zero downtime and total stability during stress tests.",
        "I am ecstatic with how easy it is to operate this platform.",
        "The results are consistently positive, reassuring, and highly accurate.",
        "Superb engineering standards and top-tier code quality throughout.",
        "We are deeply thankful for the rapid resolution and great outcome."
    ]

    negatives = [
        "The application crashed abruptly with an unhandled memory exception.",
        "Terrible latency issues and unacceptable degradation under light load.",
        "I am very frustrated with the constant errors and poor documentation.",
        "This is a complete failure to meet basic project specifications.",
        "The predictions are erratic, completely unreliable, and full of false positives.",
        "Severe bug detected in the webcam stream causing system freeze.",
        "I am extremely disappointed by the substandard quality of this release.",
        "The model failed miserably on all edge cases and noisy input samples.",
        "Worst performance we have observed in any comparable evaluation test.",
        "The user interface is broken, unresponsive, and painfully confusing.",
        "Critical vulnerability identified during security audit and validation.",
        "We experienced disastrous data corruption during the inference phase.",
        "Horrible experience trying to run this script on Windows environment.",
        "The code is messy, undocumented, and impossible to debug effectively.",
        "Inaccurate results throughout the entire validation dataset.",
        "I feel stressed and dissatisfied with the lack of progress on this bug.",
        "The camera connection repeatedly dropped without any meaningful error message.",
        "Completely unacceptable delay in processing user requests.",
        "Flawed methodology resulting in biased and meaningless classifications.",
        "The system consumed excessive memory and crashed the development environment.",
        "Extremely painful to configure with constant dependency conflicts.",
        "The output was garbage and contradicted observable facts.",
        "Regretful decision to proceed with this unstable architecture.",
        "The model exhibits severe overfitting and fails on real-world test data.",
        "Disastrous demonstration where all features stalled during live presentation.",
        "Deeply disappointed with the lack of attention to engineering standards.",
        "This error is catastrophic and corrupts session telemetry files.",
        "Unbearably slow execution time that renders the tool unusable.",
        "The facial detector lost tracking immediately upon minimal head motion.",
        "Frustrating experience dealing with broken packages and missing symbols.",
        "I hate dealing with these awful, persistent runtime errors.",
        "Everything went wrong and the entire pipeline collapsed midway.",
        "Unacceptable inaccuracies that make this system completely unsafe.",
        "The customer service was rude, dismissive, and utterly useless.",
        "I am furious about the wasted time caused by this defective module.",
        "Total waste of resources on a poorly conceived and broken idea.",
        "Extremely bad experience with frequent freezing and unresponsiveness.",
        "The predictions are completely wrong and hallucinate non-existent cues.",
        "Dreadful performance that fails every single automated validation check.",
        "The application failed to launch and spat out cryptic traceback errors.",
        "I feel miserable trying to get this problematic code to execute.",
        "Useless garbage output with zero diagnostic value.",
        "Pathetic execution speed that lags several seconds behind the camera feed.",
        "The results are hopelessly inaccurate and contradict manual observations.",
        "This was an absolute disaster from start to finish.",
        "Completely broken logic with dangerous edge case vulnerabilities.",
        "I am sick and tired of these endless, unhelpful exception crashes.",
        "The worst user interface I have ever had the misfortune to navigate.",
        "Fails consistently under normal operating conditions with high failure rate.",
        "Deeply regrettable and substandard engineering work."
    ]

    neutrals = [
        "The system initialized the camera device at index zero.",
        "The text input contains fourteen tokens and three punctuation marks.",
        "The scheduled update will run at 04:00 UTC tomorrow morning.",
        "Please enter your observation into the designated text area above.",
        "The server returned status code 200 indicating successful transmission.",
        "The dataset is partitioned into training, validation, and testing sets.",
        "MediaPipe mesh extraction identified four hundred sixty-eight landmarks.",
        "The document was saved to the project assets directory.",
        "The session history stores recorded inferences in JSON format.",
        "The CPU clock frequency remained constant throughout the execution cycle.",
        "The configuration file specifies a fifteen frame per second limit.",
        "The user navigated from the dashboard tab to the combined intelligence panel.",
        "The model was compiled using Python version three point thirteen.",
        "This report summarizes the operational statistics collected during the run.",
        "The input sequence consists of alphanumeric characters and spaces.",
        "Memory usage was logged at sixty megabytes upon application startup.",
        "The current system timestamp is recorded in Greenwich Mean Time.",
        "The webcam resolution is configured to six hundred forty by four eighty pixels.",
        "The pipeline accepts strings of text as input and returns class probabilities.",
        "There are three distinct classes defined in the classification ontology.",
        "The data table displays timestamps, input strings, and predicted categories.",
        "The process was executed via standard powershell command terminal.",
        "The script reads environment variables from the local configuration file.",
        "The camera permission dialog was presented to the user by the browser.",
        "The algorithm calculates Euclidean distances between normalized coordinate points.",
        "All diagnostic logs are stored in the temporary system directory.",
        "The evaluation matrix presents true labels against model predictions.",
        "The user clicked the clear history action button at twelve minutes past noon.",
        "The batch size for inference was set to one single item per iteration.",
        "This statement contains factual information regarding system operation.",
        "The input text contains seventy-five characters including spaces.",
        "The neural network layers were initialized with random normal weights.",
        "Standard operating procedure requires verification of all sensor feeds.",
        "The telemetry panel displays CPU load, memory usage, and inference delay.",
        "Data packets were transmitted across local loopback socket 8501.",
        "The experiment was conducted under normal ambient room temperature.",
        "A total of three modalities were evaluated during the research trial.",
        "The function returns a floating point value between zero and one.",
        "The classification threshold was configured at point five zero.",
        "The camera frame was resized to three hundred by three hundred pixels.",
        "The operating system platform was identified as Windows 64-bit.",
        "The pipeline executes sequentially from tokenization to classification.",
        "System documentation is located in the root repository directory.",
        "The process ID was recorded in the operating system task manager.",
        "The model parameters occupy twenty-four megabytes of memory storage.",
        "The user selected the second option from the navigation menu.",
        "Inference completed in forty-two milliseconds on standard CPU thread.",
        "The table columns are labeled Timestamp, Prediction, and Confidence.",
        "All measurements were recorded in SI metric standard units.",
        "The application is listening on the default local host address."
    ]

    rows = []
    for t in positives:
        rows.append((t, "Positive"))
    for t in negatives:
        rows.append((t, "Negative"))
    for t in neutrals:
        rows.append((t, "Neutral"))

    return pd.DataFrame(rows, columns=["text", "sentiment"])

def train_and_evaluate_model():
    """
    Executes reproducible training, evaluates on held-out test data,
    saves evaluation metrics, and persists the serialized scikit-learn pipeline.
    """
    print("=" * 60)
    print("SENTINEL AI: Text Sentiment Classifier Training")
    print("=" * 60)

    # 1. Prepare Data
    df = get_sentiment_corpus()
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(config.TRAINING_DATA_PATH, index=False)
    print(f"Dataset generated with {len(df)} samples across classes: {df['sentiment'].value_counts().to_dict()}")

    X = np.array(df["text"].tolist(), dtype=object)
    y = np.array(df["sentiment"].tolist(), dtype=object)

    # 2. Stratified Train-Test Split (75% train, 25% test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    print(f"Training set: {len(X_train)} samples | Held-out test set: {len(X_test)} samples")

    # 3. Build Reproducible Pipeline (TF-IDF + Logistic Regression)
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(
            ngram_range=(1, 2),
            min_df=1,
            sublinear_tf=True,
            strip_accents="unicode",
            lowercase=True
        )),
        ("classifier", LogisticRegression(
            C=3.0,
            max_iter=1000,
            class_weight="balanced",
            random_state=42
        ))
    ])

    # 4. Fit Pipeline
    pipeline.fit(X_train, y_train)
    print("Pipeline training completed successfully.")

    # 5. Evaluate on Held-out Test Set
    y_pred = pipeline.predict(X_test)
    y_pred_proba = pipeline.predict_proba(X_test)

    acc = float(accuracy_score(y_test, y_pred))
    precision, recall, f1, support = precision_recall_fscore_support(
        y_test, y_pred, average="macro"
    )
    cm = confusion_matrix(y_test, y_pred, labels=config.CLASSES).tolist()
    rep = classification_report(y_test, y_pred, target_names=config.CLASSES, output_dict=True)

    print("\n--- Model Evaluation Summary (Held-out Test Data) ---")
    print(f"Accuracy:  {acc * 100:.2f}%")
    print(f"Precision: {precision * 100:.2f}% (macro)")
    print(f"Recall:    {recall * 100:.2f}% (macro)")
    print(f"F1-Score:  {f1 * 100:.2f}% (macro)")
    print("Confusion Matrix:")
    print(pd.DataFrame(cm, index=config.CLASSES, columns=config.CLASSES))

    # 6. Save Model Artifacts & Metrics
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, config.TEXT_MODEL_PATH)
    print(f"Model persisted to: {config.TEXT_MODEL_PATH}")

    metrics = {
        "accuracy": round(acc, 4),
        "macro_precision": round(float(precision), 4),
        "macro_recall": round(float(recall), 4),
        "macro_f1": round(float(f1), 4),
        "classes": config.CLASSES,
        "confusion_matrix": cm,
        "classification_report": rep,
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "algorithm": "TF-IDF (Unigrams + Bigrams) + L2 Regularized Logistic Regression"
    }

    with open(config.TEXT_METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics persisted to: {config.TEXT_METRICS_PATH}")
    print("=" * 60)
    return metrics

if __name__ == "__main__":
    train_and_evaluate_model()
