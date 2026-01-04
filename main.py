"""
Main Execution Script
AI-Based Climate Risk-Aware Supplier Selection for Sustainable Global Supply Chains

Commodity: Coffee
Supplier Countries: Brazil, Colombia, Vietnam, Ethiopia, Honduras

This script runs the complete pipeline:
1. Dataset generation
2. Data preprocessing
3. ML model training and evaluation
4. Supplier ranking
5. Visualization generation
"""

import os
import sys
import pandas as pd
import numpy as np

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from generate_dataset import generate_supplier_dataset, COMMODITY_CONFIGS
from data_preprocessing import DataPreprocessor
from model_training import ClimateRiskPredictor
from supplier_ranking import SupplierRanker
from visualizations import Visualizer

def main(commodity='Coffee'):
    """Main execution function"""
    
    print("="*70)
    print("AI-Based Climate Risk-Aware Supplier Selection System")
    print(f"Commodity: {commodity}")
    print("="*70)
    print()
    
    # Validate commodity
    if commodity not in COMMODITY_CONFIGS:
        print(f"Error: Commodity '{commodity}' not supported.")
        print(f"Available commodities: {', '.join(COMMODITY_CONFIGS.keys())}")
        return
    
    # Create necessary directories
    os.makedirs('data', exist_ok=True)
    os.makedirs('outputs', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    
    # Step 1: Generate Dataset
    print("\n" + "="*70)
    print("STEP 1: Generating Supplier Dataset")
    print("="*70)
    df = generate_supplier_dataset(commodity=commodity)
    dataset_path = f'data/supplier_data_{commodity.lower().replace(" ", "_")}.csv'
    df.to_csv(dataset_path, index=False)
    print(f"Dataset saved to {dataset_path}")
    print(f"Dataset shape: {df.shape}")
    print(f"\nDataset preview:")
    print(df.head())
    print(f"\nClimate risk distribution:")
    print(df['climate_risk_level'].value_counts())
    
    # Step 2: Data Preprocessing
    print("\n" + "="*70)
    print("STEP 2: Data Preprocessing")
    print("="*70)
    preprocessor = DataPreprocessor()
    X_train, X_test, y_train, y_test, df_processed, label_mapping = preprocessor.preprocess_pipeline(
        dataset_path, normalize=True, test_size=0.2
    )
    
    print(f"\nLabel mapping: {label_mapping}")
    
    # Step 3: Model Training
    print("\n" + "="*70)
    print("STEP 3: Training Climate Risk Prediction Model")
    print("="*70)
    predictor = ClimateRiskPredictor(n_estimators=100, random_state=42, max_depth=10)
    predictor.train(X_train, y_train)
    
    # Display feature importance
    feature_importance = predictor.get_feature_importance()
    print("\nFeature Importance:")
    print(feature_importance)
    
    # Save model
    model_path = f'models/climate_risk_model_{commodity.lower().replace(" ", "_")}.pkl'
    predictor.save_model(model_path)
    
    # Step 4: Model Evaluation
    print("\n" + "="*70)
    print("STEP 4: Model Evaluation")
    print("="*70)
    evaluation_results = predictor.evaluate(X_test, y_test, label_mapping)
    
    # Step 5: Predict Climate Risk for All Suppliers
    print("\n" + "="*70)
    print("STEP 5: Predicting Climate Risk for All Suppliers")
    print("="*70)
    
    # Prepare features for all suppliers
    X_all = df_processed[preprocessor.feature_columns]
    X_all_scaled = preprocessor.normalize_features(X_all, fit=False)
    
    # Predict
    climate_risk_predictions = predictor.predict(X_all_scaled)
    climate_risk_predictions_labels = [label_mapping[pred] for pred in climate_risk_predictions]
    
    # Add predictions to dataframe
    df_processed['predicted_climate_risk'] = climate_risk_predictions_labels
    
    print(f"\nPredicted climate risk distribution:")
    print(pd.Series(climate_risk_predictions_labels).value_counts())
    
    # Step 6: Supplier Ranking
    print("\n" + "="*70)
    print("STEP 6: Ranking Suppliers with Weighted Scoring")
    print("="*70)
    print("Weighting scheme:")
    print("  - Cost: 25%")
    print("  - Lead Time: 25%")
    print("  - Quality: 20%")
    print("  - AI-Predicted Climate Risk: 30%")
    
    ranker = SupplierRanker()
    df_ranked = ranker.calculate_supplier_scores(df_processed, climate_risk_predictions_labels)
    
    # Get ranking report
    ranking_report = ranker.generate_ranking_report(df_ranked, top_n=15)
    print("\nTop 15 Ranked Suppliers:")
    print(ranking_report.to_string(index=False))
    
    # Country-level summary
    country_summary = ranker.get_aggregated_ranking(df_ranked)
    print("\n" + "="*70)
    print("Country-Level Summary (Average Scores)")
    print("="*70)
    print(country_summary.round(3).to_string())
    
    # Save ranking results
    ranking_output_path = f'outputs/supplier_rankings_{commodity.lower().replace(" ", "_")}.csv'
    df_ranked.to_csv(ranking_output_path, index=False)
    print(f"\nFull ranking results saved to {ranking_output_path}")
    
    # Step 7: Generate Visualizations
    print("\n" + "="*70)
    print("STEP 7: Generating Visualizations")
    print("="*70)
    
    visualizer = Visualizer(output_dir='outputs')
    
    # Plot 1: Feature Importance
    visualizer.plot_feature_importance(feature_importance)
    print("[OK] Feature importance plot created")
    
    # Plot 2: Confusion Matrix
    visualizer.plot_confusion_matrix(
        evaluation_results['confusion_matrix'],
        list(label_mapping.values())
    )
    print("[OK] Confusion matrix plot created")
    
    # Plot 3: Supplier Ranking
    visualizer.plot_supplier_ranking(df_ranked, top_n=10)
    print("[OK] Supplier ranking plot created")
    
    # Plot 4: Climate Risk by Country
    visualizer.plot_climate_risk_by_country(df_ranked)
    print("[OK] Climate risk by country plot created")
    
    # Plot 5: Metrics Comparison
    visualizer.plot_supplier_metrics_comparison(df_ranked, top_n=5)
    print("[OK] Supplier metrics comparison plot created")
    
    # Step 8: Generate Summary Report
    print("\n" + "="*70)
    print("STEP 8: Generating Summary Report")
    print("="*70)
    
    summary_report = f"""
{'='*70}
AI-BASED CLIMATE RISK-AWARE SUPPLIER SELECTION - SUMMARY REPORT
Commodity: Coffee
{'='*70}

MODEL PERFORMANCE:
------------------
Accuracy: {evaluation_results['accuracy']:.4f} ({evaluation_results['accuracy']*100:.2f}%)

Feature Importance:
{feature_importance.to_string(index=False)}

TOP 10 SUPPLIERS:
-----------------
{ranking_report.head(10).to_string(index=False)}

COUNTRY RANKINGS (by average composite score):
-----------------------------------------------
{country_summary[['country_rank', 'composite_score', 'cost_per_unit_usd', 
                  'lead_time_days', 'quality_score', 'predicted_climate_risk']].to_string()}

CLIMATE RISK DISTRIBUTION:
--------------------------
{pd.Series(climate_risk_predictions_labels).value_counts().to_string()}

OUTPUT FILES:
-------------
- Dataset: data/supplier_data.csv
- Model: models/climate_risk_model.pkl
- Rankings: outputs/supplier_rankings.csv
- Visualizations: outputs/*.png

{'='*70}
"""
    
    print(summary_report)
    
    # Save summary report
    with open('outputs/summary_report.txt', 'w') as f:
        f.write(summary_report)
    print("Summary report saved to outputs/summary_report.txt")
    
    print("\n" + "="*70)
    print("PIPELINE COMPLETED SUCCESSFULLY!")
    print("="*70)
    print("\nAll outputs have been saved to the 'outputs' directory.")
    print("Please refer to the README.md for detailed explanations.")

if __name__ == "__main__":
    main()

