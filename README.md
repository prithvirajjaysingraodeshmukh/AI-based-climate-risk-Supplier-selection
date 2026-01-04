# AI-Based Climate Risk-Aware Supplier Selection for Sustainable Global Supply Chains

## Project Overview

This project implements an AI-powered decision support system that predicts climate risk for supplier regions using machine learning and ranks suppliers for global sourcing decisions. The system combines traditional supply chain metrics (cost, lead time, quality) with AI-predicted climate risk scores to provide comprehensive supplier rankings.

**Commodity:** Coffee  
**Supplier Countries:** Brazil, Colombia, Vietnam, Ethiopia, Honduras

## Problem Statement

Global supply chains face increasing vulnerability to climate-related disruptions. Traditional supplier selection methods often prioritize cost and quality but fail to adequately account for climate risks that can cause supply disruptions, quality degradation, and financial losses. This project addresses this gap by integrating AI-predicted climate risk assessments into supplier evaluation frameworks.

## Objectives

1. Develop an ML model to predict climate risk levels (Low/Medium/High) based on climate indicators
2. Integrate climate risk predictions into a weighted supplier scoring model
3. Rank suppliers considering both traditional supply chain metrics and climate resilience
4. Provide actionable insights through visualizations and comprehensive reports

## Methodology

### 1. Data Collection & Preparation

The dataset includes:
- **Climate Features:**
  - Average temperature (Celsius)
  - Annual rainfall (mm)
  - Number of extreme climate events per year

- **Supply Chain Features:**
  - Cost per unit (USD)
  - Lead time (days)
  - Quality score (0-1 scale)
  - Reliability score (0-1 scale)

- **Target Variable:**
  - Climate risk level (Low/Medium/High)

The dataset contains 250 samples (50 per country) with realistic distributions based on actual coffee-producing regions.

### 2. Data Preprocessing

- **Missing Value Handling:** Median imputation for numeric features
- **Feature Normalization:** StandardScaler for ML model inputs
- **Label Encoding:** Categorical risk levels encoded for classification
- **Train-Test Split:** 80-20 split with stratification

### 3. Machine Learning Model

**Algorithm:** Random Forest Classifier

**Rationale:**
- Handles non-linear relationships between climate features and risk levels
- Provides feature importance insights
- Robust to outliers and handles class imbalance
- Good interpretability for decision support systems

**Model Configuration:**
- Number of estimators: 100
- Max depth: 10
- Class weights: Balanced (to handle class imbalance)
- Random state: 42 (for reproducibility)

**Evaluation Metrics:**
- Accuracy
- Confusion Matrix
- Classification Report (Precision, Recall, F1-score)
- Feature Importance Analysis

### 4. Supplier Ranking System

Suppliers are ranked using a weighted composite score:

| Metric | Weight | Normalization |
|--------|--------|---------------|
| Cost | 25% | Lower is better (inverted) |
| Lead Time | 25% | Lower is better (inverted) |
| Quality | 20% | Higher is better |
| AI-Predicted Climate Risk | 30% | Low=1.0, Medium=0.5, High=0.0 |

**Composite Score Formula:**
```
Score = 0.25 × Cost_Norm + 0.25 × LeadTime_Norm + 0.20 × Quality_Norm + 0.30 × ClimateRisk_Score
```

Higher composite scores indicate better suppliers. All metrics are normalized to 0-1 scale before weighting.

## Project Structure

```
.
├── data/                      # Dataset directory
│   └── supplier_data.csv     # Generated supplier dataset
├── src/                       # Source code modules
│   ├── generate_dataset.py   # Dataset generation script
│   ├── data_preprocessing.py # Data preprocessing module
│   ├── model_training.py     # ML model training and evaluation
│   ├── supplier_ranking.py   # Supplier ranking system
│   └── visualizations.py     # Visualization functions
├── models/                    # Trained models
│   └── climate_risk_model.pkl
├── outputs/                   # Generated outputs
│   ├── supplier_rankings.csv
│   ├── summary_report.txt
│   ├── feature_importance.png
│   ├── confusion_matrix.png
│   ├── supplier_ranking.png
│   ├── climate_risk_by_country.png
│   └── supplier_metrics_comparison.png
├── main.py                    # Main execution script
├── requirements.txt           # Python dependencies
└── README.md                  # This file
```

## Installation & Setup

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)

### Installation Steps

1. **Clone or download the project:**
   ```bash
   cd "path/to/Supply EL"
   ```

2. **Create a virtual environment (recommended):**
   ```bash
   python -m venv venv
   
   # On Windows:
   venv\Scripts\activate
   
   # On Linux/Mac:
   source venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Running the Complete Pipeline

Execute the main script to run the entire pipeline:

```bash
python main.py
```

This will:
1. Generate the supplier dataset
2. Preprocess the data
3. Train the climate risk prediction model
4. Evaluate the model performance
5. Predict climate risk for all suppliers
6. Rank suppliers using weighted scoring
7. Generate visualizations
8. Create summary reports

### Running the Interactive Dashboard

After running the main pipeline, you can launch an interactive web dashboard:

```bash
streamlit run app.py
```

The dashboard will open in your default web browser and provides:

- **Interactive Rankings Table** with filtering and sorting
- **Analytics & Insights** with interactive charts
- **Climate Risk Analysis** visualizations
- **Supplier Comparison** tools (radar charts, side-by-side comparison)
- **Real-time Filtering** by country, risk level, score ranges
- **Data Export** functionality

**Note:** Make sure to run `python main.py` first to generate the required data files.

### Outputs

After running the script, you'll find:

- **`data/supplier_data.csv`**: Complete supplier dataset
- **`models/climate_risk_model.pkl`**: Trained ML model
- **`outputs/supplier_rankings.csv`**: Full ranking results with scores
- **`outputs/summary_report.txt`**: Text summary of results
- **`outputs/*.png`**: Five visualization plots:
  1. Feature importance bar chart
  2. Confusion matrix heatmap
  3. Top 10 supplier ranking bar chart
  4. Climate risk distribution by country
  5. Top 5 suppliers metrics comparison

## Key Results & Insights

### Model Performance

The Random Forest classifier achieves high accuracy in predicting climate risk levels. Key features contributing to predictions:

1. **Extreme Events per Year**: Most important indicator
2. **Average Temperature**: Strong correlation with risk
3. **Annual Rainfall**: Moderate importance

### Supplier Rankings

The ranking system identifies suppliers that balance:
- Competitive pricing
- Efficient lead times
- High quality standards
- Climate resilience

Top-ranked suppliers typically demonstrate:
- Lower climate risk exposure
- Competitive cost structures
- Acceptable lead times
- Good quality scores

### Country-Level Insights

The system provides country-level summaries showing:
- Average performance metrics per country
- Climate risk distribution
- Best-performing regions for sourcing

## Technical Details

### Model Architecture

- **Type:** Random Forest Classifier (Ensemble Method)
- **Input Features:** 3 (temperature, rainfall, extreme events)
- **Output Classes:** 3 (Low, Medium, High risk)
- **Training Strategy:** Supervised learning with balanced class weights

### Data Assumptions

1. Climate data represents annual averages/aggregates
2. Supply chain metrics are consistent across suppliers from the same country
3. Climate risk impacts are uniform within country regions
4. Historical climate patterns predict future risk (reasonable for short-term planning)

### Limitations & Future Enhancements

**Current Limitations:**
- Synthetic dataset (real-world data would improve accuracy)
- Simplified climate risk categorization (3 levels)
- Static weighting scheme (could be made adaptive)
- No temporal modeling (time series analysis could enhance predictions)

**Potential Enhancements:**
- Integration with real climate APIs (NOAA, World Bank)
- Multi-objective optimization for ranking
- Dynamic risk assessment based on seasonal patterns
- Integration with supplier performance history
- Risk scenario analysis and sensitivity testing
- Real-time monitoring and alerting system

## Dependencies

- **pandas** (≥1.5.0): Data manipulation and analysis
- **numpy** (≥1.23.0): Numerical computations
- **scikit-learn** (≥1.2.0): Machine learning algorithms
- **matplotlib** (≥3.6.0): Plotting and visualization
- **seaborn** (≥0.12.0): Statistical visualizations
- **joblib** (≥1.2.0): Model serialization
- **streamlit** (≥1.28.0): Interactive web dashboard (optional)
- **plotly** (≥5.17.0): Interactive visualizations for dashboard (optional)

## Author & Credits

This project was developed as a comprehensive AI + Supply Chain engineering solution for sustainable global sourcing decisions.

## License

This project is provided for educational and research purposes.

## Contact & Support

For questions or issues, please refer to the code comments or documentation within each module.

---

**Note:** This project demonstrates the integration of AI/ML techniques with supply chain decision-making. For production use, it is recommended to:
- Validate with real-world data
- Incorporate domain expert feedback
- Perform extensive testing and validation
- Consider regulatory and ethical implications
- Implement robust error handling and monitoring

