# Chess-Guru

## AI-Powered Human Chess Error Modeling and Personalized Analysis System

**Live Demo:**
[https://huggingface.co/spaces/YuvarajDeveloper/chess-personal-coach](https://huggingface.co/spaces/YuvarajDeveloper/chess-personal-coach)

**Full Research Report:**
[https://drive.google.com/file/d/1lcMPD3McrcuYG9fPRJAArM8LlJvK0QWL/view?usp=sharing](https://drive.google.com/file/d/1lcMPD3McrcuYG9fPRJAArM8LlJvK0QWL/view?usp=sharing)

---

## 1. Overview

Chess-Guru is a multi-phase AI system designed to model, predict, and analyze human chess mistakes using large-scale machine learning and deep learning techniques.

Unlike traditional chess engines that optimize for strongest play, Chess-Guru focuses on understanding human decision-making errors and delivering personalized improvement insights.

The system processes real games, evaluates move quality using engine ground truth, and generates structured performance reports.

---

## 2. Objectives

* Model human chess mistakes at scale
* Compare feature-engineered machine learning with spatial deep learning
* Quantify move quality using centipawn loss (CPL)
* Deploy a production-ready AI analysis platform
* Validate real-world usability through user studies

---

## 3. Dataset

* 50,000 chess games
* Approximately 3.6 million positions
* Game-level split to prevent data leakage
* Labels computed using Stockfish 16 (depth 20)
* Targets:

  * Continuous centipawn loss (regression)
  * Severe blunder classification (binary)

---

## 4. Phase 1 – Feature-Based Machine Learning

### 4.1 Feature Engineering

52 handcrafted chess features, including:

* Material balance
* King safety
* Pawn structure
* Mobility
* Tactical indicators
* Game context features

### 4.2 Models

* Random Forest Regressor
* XGBoost
* Naive baseline comparison

### 4.3 Results

* Random Forest RMSE: 45.16 CPL
* AUC: 0.87
* 95% confidence intervals computed
* Ablation study showed heavy reliance on engine evaluation features

---

## 5. Phase 2 – Spatial Deep Learning (Board-Only CNN)

### 5.1 Input Representation

* 8×8×14 board tensor:

  * 12 piece planes
  * 1 side-to-move plane
  * 1 castling rights plane
* No engine-derived features included

### 5.2 Architecture

* Multiple Conv2D layers with Batch Normalization
* Global Average Pooling
* Fully connected layers with Dropout
* Sigmoid output for blunder classification

### 5.3 Performance

* Test AUC: 0.774
* Demonstrated that spatial geometry alone captures tactical patterns
* Identified performance ceiling due to human variability

---

## 6. Phase 3 – Deployed AI Analysis Platform

### 6.1 Technology Stack

* Backend: Python 3.10, TensorFlow 2.15
* Engine: Stockfish 16
* Frontend: Streamlit
* API: Lichess integration
* Deployment: HuggingFace Spaces

### 6.2 System Capabilities

* Automatic game retrieval
* Move-by-move CPL computation
* Blunder detection and severity classification
* Opening, middlegame, and endgame phase analysis
* Personalized performance statistics
* Automated PDF/DOCX report generation

### 6.3 Performance Metrics

* Approximately 45 seconds per 40-move game
* 2.1 seconds per position for engine evaluation
* ~48 milliseconds per CNN inference

---

## 7. Validation and Impact

User study conducted on 25 players (rating range 1200–2200):

* 88% reported actionable insights
* 92% would reuse the system
* 68% showed consistent endgame weaknesses
* 42% demonstrated tactical blind spots

This confirms practical applicability and coaching potential.

---

## 8. Key Contributions

* Large-scale human error modeling on 3.6M positions
* Statistical significance testing using DeLong’s method
* Empirical comparison of interpretability vs representation learning
* Demonstrated trade-off between engine dependence and spatial learning
* Production-ready AI coaching system deployment

---

## 9. Limitations

* CNN does not model temporal move sequences
* Label noise due to human variability
* Engine evaluation cost limits real-time performance
* Training data primarily from online rapid/blitz formats

---

## 10. Future Work

* Hybrid multi-modal models (CNN + tabular features)
* Sequential modeling using LSTM or Transformer architectures
* Personalized player embeddings
* Real-time coaching capabilities
* Explainable AI (Grad-CAM, attention mechanisms)
* Mobile application deployment

---

## 11. Conclusion

Chess-Guru demonstrates that:

1. Feature-based models achieve high predictive accuracy but rely heavily on engine signals.
2. Spatial CNNs can independently learn tactical patterns from raw board geometry.
3. AI-driven chess analysis can be deployed at scale with measurable user impact.

The project establishes a foundation for hybrid human decision modeling systems in structured domains beyond chess.

---

## 12. License

MIT License – Released for research and educational purposes.
