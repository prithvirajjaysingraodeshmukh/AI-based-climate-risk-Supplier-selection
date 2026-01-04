# Quick Start Guide

## Installation

1. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Running the Project

### Option 1: Command Line Pipeline

Simply run:
```bash
python main.py
```

This will execute the complete pipeline and generate all outputs in the `outputs/` directory.

### Option 2: Interactive Web Dashboard

After running the pipeline, launch the interactive dashboard:

```bash
streamlit run app.py
```

The dashboard will open in your browser with interactive features:
- Filter and sort supplier rankings
- Interactive charts and visualizations
- Supplier comparison tools
- Real-time analytics

## Expected Outputs

After running, you'll find:

1. **Dataset**: `data/supplier_data.csv`
2. **Trained Model**: `models/climate_risk_model.pkl`
3. **Rankings**: `outputs/supplier_rankings.csv`
4. **Report**: `outputs/summary_report.txt`
5. **Visualizations**: 
   - `outputs/feature_importance.png`
   - `outputs/confusion_matrix.png`
   - `outputs/supplier_ranking.png`
   - `outputs/climate_risk_by_country.png`
   - `outputs/supplier_metrics_comparison.png`

## Project Components

- **Commodity**: Coffee
- **Countries**: Brazil, Colombia, Vietnam, Ethiopia, Honduras
- **ML Model**: Random Forest Classifier
- **Ranking Weights**: Cost (25%), Lead Time (25%), Quality (20%), Climate Risk (30%)

