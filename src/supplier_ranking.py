"""
Supplier Ranking Module
Ranks suppliers based on weighted scoring that includes AI-predicted climate risk
"""

import pandas as pd
import numpy as np

class SupplierRanker:
    """
    Ranks suppliers using weighted scoring:
    - Cost (25%)
    - Lead time (25%)
    - Quality (20%)
    - AI-predicted climate risk (30%)
    """
    
    def __init__(self, weights=None):
        """
        Initialize with custom weights or use defaults
        
        weights: dict with keys 'cost', 'lead_time', 'quality', 'climate_risk'
        """
        if weights is None:
            self.weights = {
                'cost': 0.25,
                'lead_time': 0.25,
                'quality': 0.20,
                'climate_risk': 0.30
            }
        else:
            self.weights = weights
        
        # Validate weights sum to 1.0
        total_weight = sum(self.weights.values())
        if abs(total_weight - 1.0) > 0.01:
            raise ValueError(f"Weights must sum to 1.0, got {total_weight}")
    
    def climate_risk_to_score(self, risk_level):
        """
        Convert climate risk level to numerical score (0-1)
        Lower risk = higher score
        """
        risk_mapping = {
            'Low': 1.0,      # Best score
            'Medium': 0.5,   # Medium score
            'High': 0.0      # Worst score
        }
        return risk_mapping.get(risk_level, 0.5)
    
    def normalize_feature(self, values, reverse=False):
        """
        Normalize feature values to 0-1 scale
        reverse=True: lower values are better (for cost, lead_time)
        reverse=False: higher values are better (for quality)
        """
        values = np.array(values)
        min_val = values.min()
        max_val = values.max()
        
        if max_val == min_val:
            return np.ones_like(values) * 0.5
        
        normalized = (values - min_val) / (max_val - min_val)
        
        if reverse:
            normalized = 1 - normalized  # Invert so lower is better
        
        return normalized
    
    def calculate_supplier_scores(self, df, climate_risk_predictions):
        """
        Calculate weighted scores for each supplier
        
        Parameters:
        -----------
        df : DataFrame
            Supplier data with columns: cost_per_unit_usd, lead_time_days, 
            quality_score, reliability_score
        climate_risk_predictions : array-like
            Predicted climate risk levels for each supplier
        
        Returns:
        --------
        DataFrame with supplier scores and ranking
        """
        df_scored = df.copy()
        
        # Add predicted climate risk
        df_scored['predicted_climate_risk'] = climate_risk_predictions
        
        # Convert climate risk to numerical score
        df_scored['climate_risk_score'] = df_scored['predicted_climate_risk'].apply(
            self.climate_risk_to_score
        )
        
        # Normalize features (all to 0-1 scale, higher is better)
        df_scored['cost_normalized'] = self.normalize_feature(
            df_scored['cost_per_unit_usd'], reverse=True
        )
        df_scored['lead_time_normalized'] = self.normalize_feature(
            df_scored['lead_time_days'], reverse=True
        )
        df_scored['quality_normalized'] = self.normalize_feature(
            df_scored['quality_score'], reverse=False
        )
        
        # Calculate weighted composite score
        df_scored['composite_score'] = (
            self.weights['cost'] * df_scored['cost_normalized'] +
            self.weights['lead_time'] * df_scored['lead_time_normalized'] +
            self.weights['quality'] * df_scored['quality_normalized'] +
            self.weights['climate_risk'] * df_scored['climate_risk_score']
        )
        
        # Rank suppliers (higher score = better rank)
        df_scored['rank'] = df_scored['composite_score'].rank(ascending=False, method='min').astype(int)
        
        # Sort by rank
        df_scored = df_scored.sort_values('rank')
        
        return df_scored
    
    def get_aggregated_ranking(self, df_scored):
        """
        Aggregate scores by country and create country-level ranking
        Useful for selecting best supplier from each country
        """
        country_summary = df_scored.groupby('country').agg({
            'composite_score': 'mean',
            'cost_per_unit_usd': 'mean',
            'lead_time_days': 'mean',
            'quality_score': 'mean',
            'predicted_climate_risk': lambda x: x.mode()[0] if len(x.mode()) > 0 else x.iloc[0],
            'supplier_id': 'count'
        }).rename(columns={'supplier_id': 'num_suppliers'})
        
        country_summary['country_rank'] = country_summary['composite_score'].rank(
            ascending=False, method='min'
        ).astype(int)
        country_summary = country_summary.sort_values('country_rank')
        
        return country_summary
    
    def generate_ranking_report(self, df_scored, top_n=10):
        """
        Generate a formatted ranking report
        
        Returns:
        --------
        DataFrame with key information for top N suppliers
        """
        report_columns = [
            'rank',
            'supplier_name',
            'country',
            'cost_per_unit_usd',
            'lead_time_days',
            'quality_score',
            'predicted_climate_risk',
            'composite_score'
        ]
        
        report = df_scored[report_columns].head(top_n).copy()
        report = report.round({
            'cost_per_unit_usd': 2,
            'lead_time_days': 0,
            'quality_score': 3,
            'composite_score': 4
        })
        
        return report

