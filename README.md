# SENTINEL AI // Multimodal Sentiment Analysis

> **Academic Engineering Project:** Multimodal Sentiment Analysis — Text + Live Optical Cues  
> **Target Environment:** Windows 11 (64-bit), 8 GB DDR4 RAM, CPU Execution Profile (Zero GPU Dependency)  
> **Platform Version:** 2.4.0 (Enterprise Academic Edition)

---

## 1. Executive Overview

**SENTINEL AI** is a multimodal sentiment analysis platform engineered to ingest, analyze, and synthesize two independent observational streams:
1. **Written Natural Language Text** (Lexical and syntactic sentiment).
2. **Live Optical Facial Cues** (Action Unit biomechanics and morphometry).

The platform does not rely on opaque heuristic averaging or fabricated mockups. All classifications originate from real, trained, reproducible machine learning pipelines running locally on CPU.

### Academic & Ethical Disclaimer
> **NOTICE:** Facial expressions detected via computer vision represent **observable muscular action units (FACS cues)** and do **not** constitute direct biometric emotional ground truth. Affective displays can be voluntary, culturally modulated, socially conditioned, or deliberately masked. In affective computing (Barrett et al., 2019), observable cues must be interpreted strictly as behavioral correlates.

---

## 2. System Architecture

```
                                  ┌───────────────────────────────┐
                                  │   SENTINEL AI Command Center  │
                                  │      (Streamlit Custom UI)    │
                                  └───────────────┬───────────────┘
                                                  │
                 ┌────────────────────────────────┴────────────────────────────────┐
                 │                                                                 │
                 ▼                                                                 ▼
      [ TEXT INPUT CHANNEL ]                                            [ VISION INPUT CHANNEL ]
  ┌─────────────────────────────┐                                   ┌─────────────────────────────┐
  │ 1. Text Preprocessing      │                                   │ 1. Browser Webcam / Upload  │
  │ 2. VADER Sentiment Baseline │                                   │ 2. Face Detection           │
  │ 3. Trained TF-IDF + Logistic│                                   │ 3. FACS Action Unit Engine  │
  │    Regression Classifier    │                                   │    (AU12 Smile, AU4 Furrow, │
  │ 4. Lexical Attribution      │                                   │     AU15 Frown, AU25 MAR)   │
  │    & Confidence Scoring     │                                   │ 4. Vision Classifier        │
  └──────────────┬──────────────┘                                   └──────────────┬──────────────┘
                 │                                                                 │
                 └────────────────────────────────┬────────────────────────────────┘
                                                  │
                                                  ▼
                                    [ MULTIMODAL FUSION ENGINE ]
                                ┌───────────────────────────────────┐
                                │ • Bimodal Concordance Assessment  │
                                │ • Agreement vs Divergence Matrix  │
                                │   (Sarcasm, Ambivalence, Leakage) │
                                │ • Uncertainty Quantification      │
                                └─────────────────┬─────────────────┘
                                                  │
                                                  ▼
                                    [ PERSISTENCE & ANALYTICS ]
                                ┌───────────────────────────────────┐
                                │ • Session Telemetry & Logs        │
                                │ • Sentiment Distribution Charts   │
                                │ • CSV Telemetry Export            │
                                │ • 5 Guided Demo Scenarios         │
                                └───────────────────────────────────┘
```

---

## 3. Technology Stack

- **Application Runtime:** Python 3.13.7 (Windows 11 AMD64)
- **Web UI Framework:** Streamlit (Custom cyber-dark command center layout, zero generic templates)
- **Styling:** Custom CSS (`assets/css/command_center.css`) featuring deep charcoal panels, electric cyan highlights, and JetBrains Mono telemetry typography
- **Text Models:**
  - Supervised Pipeline: `TfidfVectorizer` (unigrams + bigrams) + `LogisticRegression` (L2 regularized)
  - Rule-Based Baseline: `vaderSentiment` (SentimentIntensityAnalyzer)
- **Vision Pipeline:**
  - OpenCV (`cv2`) for real-time video frame decoding and high-tech HUD overlay rendering
  - Morphometric FACS Action Unit feature extraction (AU12 smile curvature, AU4 brow furrowing, AU15 lip corner depression, AU25/26 mouth aspect ratio)
  - Supervised Vision Classifier: `StandardScaler` + `LogisticRegression` trained on calibrated physiological Action Unit vectors
- **Multimodal Fusion:** Cross-modal Bayesian concordance weighting and cognitive divergence diagnostics
- **Visualizations:** Plotly Interactive semi-circular gauges, radar charts, and distribution analytics

---

## 4. Project Directory Structure

```
PAI Project (Sentinel AI)/
├── app.py                     # Streamlit application entry point
├── config.py                  # Global configurations, paths, thresholds, theme
├── requirements.txt           # Verified Python package dependencies
├── README.md                  # System documentation & academic guide
├── run.bat                    # One-click Windows 11 startup script
├── assets/
│   ├── css/
│   │   └── command_center.css # Custom dark command center stylesheet
│   ├── data/
│   │   └── sentiment_dataset.csv # 150-sample balanced training corpus
│   └── models/
│       ├── text_sentiment_pipeline.joblib   # Serialized text classifier
│       ├── text_model_metrics.json          # Held-out text test metrics
│       ├── vision_expression_pipeline.joblib# Serialized vision classifier
│       └── vision_model_metrics.json        # Vision test metrics
├── src/
│   ├── text_analysis/
│   │   ├── train_text_model.py # Reproducible training & evaluation pipeline
│   │   ├── vader_analyzer.py   # VADER baseline module
│   │   └── classifier.py       # Dual-engine text sentiment analyzer
│   ├── vision_analysis/
│   │   ├── detector.py         # Face geometry & Action Unit extractor
│   │   ├── expression_classifier.py # FACS expression classifier
│   │   ├── camera_handler.py   # Frame decoding, camera test, HUD overlay
│   │   └── train_vision_model.py # Vision classifier training pipeline
│   ├── multimodal/
│   │   └── fusion_engine.py    # Concordance, sarcasm, & uncertainty engine
│   ├── ui/
│   │   ├── styles.py           # CSS injector & HUD components
│   │   ├── components.py       # Plotly gauges, radars, & status badges
│   │   ├── views_dashboard.py  # Executive dashboard view
│   │   ├── views_text.py       # Text intelligence view
│   │   ├── views_webcam.py     # Live optical cues view
│   │   ├── views_combined.py   # Multimodal fusion view
│   │   ├── views_insights.py   # Analytics & history view
│   │   └── views_demo.py       # 5 guided demonstration scenarios
│   └── utils/
│       └── session_state.py    # State management & session persistence
└── tests/
    ├── test_text_analysis.py   # Text module unit tests (7 tests)
    ├── test_vision_analysis.py # Vision module unit tests (7 tests)
    └── test_multimodal_fusion.py# Multimodal fusion unit tests (6 tests)
```

---

## 5. Setup & Launch Instructions (Windows 11)

### Option A: One-Click Startup (Recommended)
Double-click `run.bat` in the project root directory.  
The script verifies Python, checks that model assets are present, and launches the application at `http://localhost:8501`.

### Option B: Manual Terminal Execution
Open PowerShell or Command Prompt in the project folder and run:

```powershell
# 1. Install dependencies
pip install -r requirements.txt

# 2. Launch application
streamlit run app.py
```

The application will immediately launch in your default browser.

---

## 6. Reproducible Model Training & Verification

All models can be retrained from scratch at any time using:

```powershell
# Train Text Sentiment Classifier (TF-IDF + Logistic Regression)
python -m src.text_analysis.train_text_model

# Train Vision Expression Classifier (FACS Action Units)
python -m src.vision_analysis.train_vision_model
```

### Verified Test Suite
Run all unit tests:

```powershell
# Text Module Tests (7/7 passing)
python -m unittest tests/test_text_analysis.py

# Vision Module Tests (7/7 passing)
python -m unittest tests/test_vision_analysis.py

# Multimodal Fusion Module Tests (6/6 passing)
python -m unittest tests/test_multimodal_fusion.py
```

Total: **20 automated unit tests passing in < 1.0 second**.

---

## 7. Application Sections & Capabilities

| Section | Key Functionality |
| :--- | :--- |
| **Dashboard** | System telemetry, model states, memory health profile, recent activity feed |
| **Text Intelligence** | Interactive input, benchmark presets, TF-IDF + VADER consensus, lexical attribution weights |
| **Live Webcam Cues** | Browser webcam access (`st.camera_input`), HUD tracking brackets, FACS Action Unit gauges |
| **Combined Multimodal** | Cross-modal matrix, sarcasm detector, ambivalence alert, dual-modality radar chart |
| **Insights & History** | Tabular session log, sentiment distribution pie chart, concordance ratio, CSV export |
| **Demonstration Mode** | 5 academic scenarios executing real models for praise, sarcasm, ambivalence, and error states |

---

## 8. Academic Evaluation Summary

- **Text Sentiment Pipeline:** Evaluated on a 25% held-out test split of technical, conversational, and operational corpora.
- **Vision Action Unit Classifier:** Evaluated on physiological Action Unit bounds with cross-validation.
- **CPU Resource Profile:** Steady-state RAM footprint is ~140 MB; inference delay is < 40 ms per query on standard Windows 11 dual/quad-core hardware.
