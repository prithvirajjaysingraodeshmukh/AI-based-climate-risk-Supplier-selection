"""
Visualization Module
Creates visualizations for climate risk predictions and supplier rankings
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10

class Visualizer:
    """Creates visualizations for the supplier selection project"""
    
    def __init__(self, output_dir='outputs'):
        """Initialize visualizer with output directory"""
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def plot_feature_importance(self, feature_importance_df, save_path=None):
        """
        Plot feature importance from trained model
        
        Parameters:
        -----------
        feature_importance_df : DataFrame
            DataFrame with 'feature' and 'importance' columns
        save_path : str, optional
            Path to save the figure
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Sort by importance
        df_sorted = feature_importance_df.sort_values('importance', ascending=True)
        
        # Create horizontal bar plot
        bars = ax.barh(df_sorted['feature'], df_sorted['importance'], color='steelblue')
        
        # Customize
        ax.set_xlabel('Importance Score', fontsize=12, fontweight='bold')
        ax.set_ylabel('Feature', fontsize=12, fontweight='bold')
        ax.set_title('Feature Importance in Climate Risk Prediction Model', 
                    fontsize=14, fontweight='bold', pad=20)
        ax.grid(axis='x', alpha=0.3)
        
        # Add value labels on bars
        for i, (idx, row) in enumerate(df_sorted.iterrows()):
            ax.text(row['importance'] + 0.01, i, f"{row['importance']:.3f}",
                   va='center', fontsize=10)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Feature importance plot saved to {save_path}")
        else:
            plt.savefig(os.path.join(self.output_dir, 'feature_importance.png'), 
                       dpi=300, bbox_inches='tight')
        
        plt.close()
    
    def plot_confusion_matrix(self, confusion_matrix, labels, save_path=None):
        """
        Plot confusion matrix
        
        Parameters:
        -----------
        confusion_matrix : array-like
            Confusion matrix array
        labels : list
            List of class labels
        save_path : str, optional
            Path to save the figure
        """
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Create heatmap
        sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='Blues',
                   xticklabels=labels, yticklabels=labels, ax=ax,
                   cbar_kws={'label': 'Count'})
        
        ax.set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
        ax.set_ylabel('True Label', fontsize=12, fontweight='bold')
        ax.set_title('Confusion Matrix - Climate Risk Prediction', 
                    fontsize=14, fontweight='bold', pad=20)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Confusion matrix saved to {save_path}")
        else:
            plt.savefig(os.path.join(self.output_dir, 'confusion_matrix.png'), 
                       dpi=300, bbox_inches='tight')
        
        plt.close()
    
    def plot_supplier_ranking(self, df_ranked, top_n=10, save_path=None):
        """
        Plot top N suppliers ranked by composite score
        
        Parameters:
        -----------
        df_ranked : DataFrame
            DataFrame with supplier rankings
        top_n : int
            Number of top suppliers to display
        save_path : str, optional
            Path to save the figure
        """
        # Get top N suppliers
        top_suppliers = df_ranked.head(top_n).copy()
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Create horizontal bar plot
        y_pos = np.arange(len(top_suppliers))
        bars = ax.barh(y_pos, top_suppliers['composite_score'], color='darkgreen', alpha=0.7)
        
        # Customize
        ax.set_yticks(y_pos)
        ax.set_yticklabels([f"{row['supplier_name']} ({row['country']})" 
                           for _, row in top_suppliers.iterrows()], fontsize=9)
        ax.set_xlabel('Composite Score', fontsize=12, fontweight='bold')
        ax.set_title(f'Top {top_n} Suppliers - Ranked by Composite Score', 
                    fontsize=14, fontweight='bold', pad=20)
        ax.grid(axis='x', alpha=0.3)
        ax.invert_yaxis()  # Top supplier at top
        
        # Add value labels
        for i, (idx, row) in enumerate(top_suppliers.iterrows()):
            ax.text(row['composite_score'] + 0.01, i, f"{row['composite_score']:.3f}",
                   va='center', fontsize=9)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Supplier ranking plot saved to {save_path}")
        else:
            plt.savefig(os.path.join(self.output_dir, 'supplier_ranking.png'), 
                       dpi=300, bbox_inches='tight')
        
        plt.close()
    
    def plot_climate_risk_by_country(self, df, save_path=None):
        """
        Plot climate risk distribution by country
        
        Parameters:
        -----------
        df : DataFrame
            DataFrame with 'country' and 'predicted_climate_risk' columns
        save_path : str, optional
            Path to save the figure
        """
        # Count risk levels by country
        risk_by_country = pd.crosstab(df['country'], df['predicted_climate_risk'])
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Create stacked bar plot
        risk_by_country.plot(kind='bar', stacked=True, ax=ax, 
                            color=['green', 'orange', 'red'], alpha=0.8)
        
        ax.set_xlabel('Country', fontsize=12, fontweight='bold')
        ax.set_ylabel('Number of Suppliers', fontsize=12, fontweight='bold')
        ax.set_title('Climate Risk Distribution by Country', 
                    fontsize=14, fontweight='bold', pad=20)
        ax.legend(title='Risk Level', bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Climate risk by country plot saved to {save_path}")
        else:
            plt.savefig(os.path.join(self.output_dir, 'climate_risk_by_country.png'), 
                       dpi=300, bbox_inches='tight')
        
        plt.close()
    
    def plot_supplier_metrics_comparison(self, df_ranked, top_n=5, save_path=None):
        """
        Create radar/spider chart comparing top suppliers across metrics
        
        Parameters:
        -----------
        df_ranked : DataFrame
            DataFrame with supplier data and scores
        top_n : int
            Number of top suppliers to compare
        save_path : str, optional
            Path to save the figure
        """
        top_suppliers = df_ranked.head(top_n).copy()
        
        # Prepare data for comparison
        metrics = ['cost_normalized', 'lead_time_normalized', 
                  'quality_normalized', 'climate_risk_score']
        metric_labels = ['Cost\n(Lower Better)', 'Lead Time\n(Lower Better)', 
                        'Quality\n(Higher Better)', 'Climate Risk\n(Lower Better)']
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        x = np.arange(len(metric_labels))
        width = 0.15
        
        for i, (idx, row) in enumerate(top_suppliers.iterrows()):
            values = [row[metric] for metric in metrics]
            offset = (i - top_n/2) * width + width/2
            ax.bar(x + offset, values, width, 
                  label=f"{row['supplier_name'][:15]}... ({row['country']})",
                  alpha=0.8)
        
        ax.set_ylabel('Normalized Score (0-1)', fontsize=12, fontweight='bold')
        ax.set_title(f'Top {top_n} Suppliers - Metrics Comparison', 
                    fontsize=14, fontweight='bold', pad=20)
        ax.set_xticks(x)
        ax.set_xticklabels(metric_labels, fontsize=10)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
        ax.set_ylim([0, 1.1])
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Metrics comparison plot saved to {save_path}")
        else:
            plt.savefig(os.path.join(self.output_dir, 'supplier_metrics_comparison.png'), 
                       dpi=300, bbox_inches='tight')
        
        plt.close()

