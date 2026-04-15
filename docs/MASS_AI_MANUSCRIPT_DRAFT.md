# MASS-AI: An Explainable and Context-Aware Theft Intelligence Platform for Smart Meter Analytics

## Abstract
Electricity theft remains one of the most costly and operationally challenging problems in power distribution systems, especially under the rapid expansion of smart metering infrastructures. Traditional anomaly detection approaches often focus only on raw consumption irregularities and may generate excessive false positives when legitimate behavioral changes, such as vacations, holidays, or seasonal occupancy shifts, are present. This paper presents **MASS-AI**, an explainable and context-aware theft intelligence platform designed for smart meter analytics. The proposed system combines synthetic theft scenario generation, context-aware feature engineering, ensemble learning, explainability analysis, and an operational investigation workflow into a unified framework. A synthetic smart meter dataset representing 2,000 customers over 180 days with 15-minute interval readings and multiple theft scenarios was used to evaluate the approach. Experimental results show that the proposed framework achieves strong detection performance, with Random Forest and XGBoost models providing high discriminative power, while the stacking-based pipeline improves operational utility through risk scoring, prioritization, and explainability. In addition to prediction accuracy, MASS-AI is designed as a decision-support platform that converts alerts into reviewable cases and supports pilot KPI monitoring for real-world deployment. The results suggest that context-aware and explainable analytics can improve both the technical and operational viability of theft detection systems in modern distribution environments.

**Keywords:** electricity theft detection, smart meters, anomaly detection, explainable AI, context-aware analytics, ensemble learning, decision support systems

## 1. Introduction
Electricity theft is a persistent challenge for utility companies and distribution system operators, creating substantial financial losses, operational inefficiencies, and reduced grid visibility. As smart metering infrastructures become more widespread, utilities gain access to increasingly granular time-series consumption data, enabling the development of advanced analytics for theft detection and non-technical loss management. However, detecting suspicious behavior from smart meter data remains difficult because abnormal consumption patterns do not always indicate fraud. Legitimate behavioral changes, including holidays, extended absences, seasonal migration, or irregular occupancy, can resemble theft-like patterns and cause high false-positive rates.

The literature contains a broad set of approaches for electricity theft detection, including statistical anomaly detection, supervised classification, deep learning, and hybrid methods. Despite these advances, many methods remain limited in one or more of the following ways. First, they rely heavily on raw consumption variation without explicitly modeling contextual signals. Second, they may provide strong classification scores but offer limited interpretability for field inspectors or analysts. Third, many works stop at the model level and do not address how suspicious cases should be prioritized, reviewed, and operationally managed in a real utility workflow.

To address these limitations, this paper proposes **MASS-AI**, an integrated theft intelligence platform for smart meter analytics. The framework is built around five main ideas:

1. **Synthetic theft scenario generation** for controlled experimentation and robustness analysis.
2. **Context-aware feature engineering** to distinguish suspicious patterns from legitimate lifestyle-driven changes.
3. **Ensemble-based theft scoring** for strong classification performance on structured smart meter features.
4. **Explainability mechanisms** to provide analyst-facing justification for suspicious cases.
5. **Operational workflow integration** that transforms model outputs into alerts, prioritized cases, and pilot performance indicators.

The main contribution of this work is not a single novel algorithm in isolation, but rather the integration of machine learning, contextual reasoning, explainability, and operational case management into one end-to-end platform. From a practical standpoint, this is important because utilities do not only need high-scoring models; they need systems that help analysts decide which cases should be reviewed first, why they are suspicious, and how field outcomes can be fed back into the evaluation loop.

## 2. Related Work
Electricity theft detection has been studied extensively under the broader topics of non-technical loss detection, consumption anomaly analysis, and smart meter fraud analytics. Existing methods can be broadly grouped into statistical methods, machine learning methods, deep learning models, and hybrid operational systems.

### 2.1 Statistical and classical anomaly detection approaches
Early works in theft detection often relied on rule-based thresholds, consumption deviation ratios, and outlier detection techniques. These methods are attractive because they are simple to implement and computationally inexpensive. However, they often struggle when customer consumption behavior is highly variable or when suspicious patterns resemble legitimate activity shifts.

### 2.2 Supervised machine learning for theft detection
As labeled theft datasets became more common, supervised learning methods such as Decision Trees, Random Forests, Gradient Boosting, and XGBoost gained popularity. These models are well suited for structured features derived from consumption history and can capture nonlinear interactions among features. Tree-based methods are particularly attractive for utility applications because they often provide strong performance while remaining easier to interpret than many deep neural architectures.

### 2.3 Deep learning and sequence modeling
Recent studies have explored convolutional and recurrent architectures, including CNNs, LSTMs, and autoencoders, for electricity theft detection. These methods can model raw consumption sequences more directly and may capture temporal signatures that handcrafted features fail to represent. Nevertheless, they often require larger datasets, more computation, and more careful tuning. They may also be less transparent for field analysts when compared with structured feature-based approaches.

### 2.4 Explainability in utility analytics
Explainable AI has become increasingly important in operational environments where predictions must be justified to domain experts. Feature importance methods and SHAP-based explanations are commonly used to provide global and local insight into model behavior. In the context of theft detection, explainability can increase analyst trust and improve the usability of model outputs during case review.

### 2.5 Gaps in current practice
While many prior studies report strong detection accuracy, fewer works focus on the full operational chain from scoring to review. In practice, utilities need systems that do more than classify. They need mechanisms to prioritize cases, explain suspicious behavior, record review outcomes, and measure pilot effectiveness. Moreover, contextual signals such as holidays, customer lifestyle changes, and regional deviations are still underused in many practical systems.

MASS-AI is designed to bridge these gaps by combining context-aware analytics, explainability, and operational workflow design into a single platform.

## 3. Proposed Framework: MASS-AI
MASS-AI is designed as a modular platform for electricity theft intelligence. Its architecture can be summarized as:

$$
\text{Raw or engineered smart meter data} \rightarrow \text{Feature engineering} \rightarrow \text{Risk scoring} \rightarrow \text{Alert generation} \rightarrow \text{Case review} \rightarrow \text{Pilot KPI feedback}
$$

### 3.1 System overview
The framework consists of four main layers:

- **Data layer:** ingestion of raw smart meter data, engineered feature datasets, or pre-scored customer files.
- **Analytics layer:** feature engineering, model inference, scoring, and prioritization.
- **Explainability layer:** local and global reasoning summaries for suspicious cases.
- **Operational layer:** alert management, case assignment, analyst notes, inspection outcomes, and pilot KPI tracking.

This design enables MASS-AI to function both as a research prototype and as a pilot-ready utility decision-support platform.

### 3.2 Synthetic theft scenario generation
Since access to large, labeled theft datasets is limited, a synthetic data generation module was used to simulate different customer behaviors and theft patterns. The scenario generator includes multiple suspicious behaviors, such as constant reduction, night-time zeroing, random zero intervals, peak clipping, and gradual consumption decrease.

### 3.3 Context-aware feature engineering
One of the key components of MASS-AI is context-aware feature engineering. Instead of relying only on raw means and variances, the system extracts structured indicators that capture suspicious patterns while accounting for legitimate changes.

Examples of features include daily and weekly variability, zero-consumption ratios, sudden change ratios, night/day usage balance, holiday consumption behavior, neighborhood deviation scores, near-zero persistence, and post-low recovery behavior.

Conceptually, the feature space can be written as:

$$
X_{\text{features}} = X_{\text{statistical}} + X_{\text{temporal}} + X_{\text{contextual}}
$$

### 3.4 Ensemble learning and risk scoring
The main prediction engine is built around ensemble learning. Baseline models include Random Forest and XGBoost. More advanced versions of the pipeline also include stacking, calibration, and threshold tuning.

Given a customer feature vector $x$, the model estimates a theft-related risk probability:

$$
p(\text{theft} \mid x)
$$

This score is then transformed into an operational priority using confidence and loss-aware logic:

$$
\text{Priority} = 0.5 \cdot \text{Risk} + 0.25 \cdot \text{Confidence} + 0.25 \cdot \text{Normalized Loss}
$$

### 3.5 Explainability layer
MASS-AI uses explanation summaries to answer the practical question: **Why is this customer suspicious?** Local explanation mechanisms summarize the most influential signals behind each risk score.

### 3.6 Case management and pilot monitoring
Rather than stopping at prediction, MASS-AI includes an operational review layer in which alerts are converted into cases. Cases can be assigned, reviewed, updated, and closed with outcomes such as confirmed fraud or rejected suspicion. This enables pilot metrics such as hit-rate, review coverage, false-positive rate, and recovered amount to be tracked over time.

## 4. Dataset and Feature Design
### 4.1 Synthetic dataset construction
To evaluate the proposed framework, a synthetic smart meter dataset was generated to resemble realistic residential customer behavior while allowing controlled insertion of theft scenarios. The dataset contains 2,000 customers, 180 days of usage history, 15-minute interval readings, approximately 12% suspicious cases, and multiple theft patterns with varied severity and temporal signatures.

### 4.2 Input formats
The platform is designed to support three input modes:
1. Raw smart meter time-series with customer ID, timestamp, and consumption.
2. Engineered feature tables for faster model execution.
3. Pre-scored customer files where probabilities are already available.

### 4.3 Feature groups
The engineered feature set can be grouped as statistical features, temporal pattern features, theft-specific features, and contextual features.

## 5. Experimental Setup and Results
### 5.1 Experimental setup
The models were trained and evaluated on the synthetic dataset using a supervised classification setup. Class imbalance was addressed through weighting and threshold-aware evaluation. Performance was analyzed using standard classification metrics, including ROC-AUC, F1-score, precision, and recall.

### 5.2 Baseline model performance
Among the baseline models, Random Forest and XGBoost provided the strongest results. In the evaluated synthetic setup, Random Forest achieved an ROC-AUC of approximately **0.9471**, while XGBoost achieved approximately **0.9373**.

### 5.3 Operational interpretation of results
In MASS-AI, model outputs are transformed into theft probability, confidence score, estimated monthly loss, priority score, and recommended action.

### 5.4 Explainability outcomes
Explainability analyses showed that the most influential features often included sudden change behavior, zero-consumption persistence, and contextual deviation indicators.

### 5.5 Pilot KPI layer
Beyond model metrics, MASS-AI supports pilot metrics such as:

$$
\text{Hit-rate} = \frac{\text{confirmed fraud cases}}{\text{reviewed cases}}
$$

$$
\text{Review coverage} = \frac{\text{reviewed cases}}{\text{total high-risk cases}}
$$

$$
\text{False positive rate} = \frac{\text{rejected suspicious cases}}{\text{reviewed cases}}
$$

## 6. Discussion
The results demonstrate that context-aware, feature-based ensemble models can perform strongly on theft detection tasks, particularly when supported by explainability and workflow integration. From a research perspective, the study confirms that synthetic scenario diversity can provide a useful experimental foundation when real-world theft labels are limited. From an operational perspective, the work suggests that model value is significantly enhanced when predictions are transformed into structured alerts, prioritized cases, and measurable pilot outcomes.

### 6.1 Limitations
The main evaluation was performed on synthetic data rather than on a large, real utility dataset. Although the system supports raw ingestion and case workflow logic, a full-scale utility deployment requires stronger validation under real-world data quality issues, organizational constraints, and regulatory requirements.

### 6.2 Future work
Future work will focus on validation on real MASS or distribution-company datasets, graph-based or neighborhood-aware anomaly modeling, calibrated probabilistic risk estimation, semi-supervised learning, federated learning, and deeper integration with pilot inspection outcomes and retraining workflows.

## 7. Conclusion
This paper introduced MASS-AI, an explainable and context-aware theft intelligence platform for smart meter analytics. The system integrates synthetic scenario generation, contextual feature engineering, ensemble learning, explainability, and case-based operational review into a single end-to-end framework. Results on a synthetic dataset indicate that the proposed approach can achieve strong detection performance while also providing practical decision-support functionality through prioritization, case tracking, and pilot KPI monitoring.
