"""
Dataset Generation Script
Generates realistic datasets for multiple commodities with climate and supply chain features
"""

import pandas as pd
import numpy as np
from datetime import datetime

# Commodity configurations
COMMODITY_CONFIGS = {
    'Coffee': {
        'countries': ['Brazil', 'Colombia', 'Vietnam', 'Ethiopia', 'Honduras'],
        'country_profiles': {
            'Brazil': {
                'temp_range': (20, 28),
                'rainfall_range': (1000, 1800),
                'events_range': (2, 8),
                'cost_range': (3.5, 5.0),
                'lead_time_range': (25, 45),
                'quality_base': 0.75,
                'reliability_base': 0.80
            },
            'Colombia': {
                'temp_range': (18, 24),
                'rainfall_range': (1500, 2500),
                'events_range': (1, 6),
                'cost_range': (4.0, 6.0),
                'lead_time_range': (20, 35),
                'quality_base': 0.85,
                'reliability_base': 0.85
            },
            'Vietnam': {
                'temp_range': (22, 28),
                'rainfall_range': (1200, 2000),
                'events_range': (3, 10),
                'cost_range': (3.0, 4.5),
                'lead_time_range': (30, 50),
                'quality_base': 0.70,
                'reliability_base': 0.75
            },
            'Ethiopia': {
                'temp_range': (15, 25),
                'rainfall_range': (800, 1500),
                'events_range': (2, 7),
                'cost_range': (4.5, 6.5),
                'lead_time_range': (35, 55),
                'quality_base': 0.90,
                'reliability_base': 0.70
            },
            'Honduras': {
                'temp_range': (20, 26),
                'rainfall_range': (1000, 1800),
                'events_range': (4, 12),
                'cost_range': (3.5, 5.5),
                'lead_time_range': (25, 40),
                'quality_base': 0.75,
                'reliability_base': 0.72
            }
        }
    },
    'Cocoa': {
        'countries': ['Ivory Coast', 'Ghana', 'Indonesia', 'Ecuador', 'Cameroon'],
        'country_profiles': {
            'Ivory Coast': {
                'temp_range': (24, 30),
                'rainfall_range': (1200, 2000),
                'events_range': (3, 9),
                'cost_range': (2.8, 4.2),
                'lead_time_range': (30, 50),
                'quality_base': 0.78,
                'reliability_base': 0.75
            },
            'Ghana': {
                'temp_range': (24, 32),
                'rainfall_range': (1000, 1800),
                'events_range': (2, 8),
                'cost_range': (3.0, 4.5),
                'lead_time_range': (35, 55),
                'quality_base': 0.82,
                'reliability_base': 0.80
            },
            'Indonesia': {
                'temp_range': (22, 28),
                'rainfall_range': (1500, 2500),
                'events_range': (4, 12),
                'cost_range': (2.5, 4.0),
                'lead_time_range': (25, 45),
                'quality_base': 0.72,
                'reliability_base': 0.70
            },
            'Ecuador': {
                'temp_range': (20, 26),
                'rainfall_range': (2000, 3000),
                'events_range': (1, 6),
                'cost_range': (4.0, 6.5),
                'lead_time_range': (40, 60),
                'quality_base': 0.88,
                'reliability_base': 0.78
            },
            'Cameroon': {
                'temp_range': (22, 28),
                'rainfall_range': (1500, 2200),
                'events_range': (3, 10),
                'cost_range': (3.2, 4.8),
                'lead_time_range': (30, 50),
                'quality_base': 0.75,
                'reliability_base': 0.73
            }
        }
    },
    'Cotton': {
        'countries': ['China', 'India', 'United States', 'Brazil', 'Pakistan'],
        'country_profiles': {
            'China': {
                'temp_range': (18, 28),
                'rainfall_range': (500, 1200),
                'events_range': (2, 8),
                'cost_range': (1.8, 3.0),
                'lead_time_range': (20, 40),
                'quality_base': 0.80,
                'reliability_base': 0.85
            },
            'India': {
                'temp_range': (20, 32),
                'rainfall_range': (600, 1500),
                'events_range': (4, 12),
                'cost_range': (1.5, 2.8),
                'lead_time_range': (25, 45),
                'quality_base': 0.75,
                'reliability_base': 0.78
            },
            'United States': {
                'temp_range': (15, 28),
                'rainfall_range': (400, 1000),
                'events_range': (1, 7),
                'cost_range': (2.5, 4.0),
                'lead_time_range': (15, 30),
                'quality_base': 0.85,
                'reliability_base': 0.90
            },
            'Brazil': {
                'temp_range': (22, 30),
                'rainfall_range': (800, 1600),
                'events_range': (3, 9),
                'cost_range': (2.0, 3.5),
                'lead_time_range': (30, 50),
                'quality_base': 0.78,
                'reliability_base': 0.80
            },
            'Pakistan': {
                'temp_range': (18, 32),
                'rainfall_range': (200, 800),
                'events_range': (2, 10),
                'cost_range': (1.6, 2.9),
                'lead_time_range': (35, 55),
                'quality_base': 0.73,
                'reliability_base': 0.75
            }
        }
    },
    'Rice': {
        'countries': ['China', 'India', 'Thailand', 'Vietnam', 'Bangladesh'],
        'country_profiles': {
            'China': {
                'temp_range': (18, 28),
                'rainfall_range': (800, 1600),
                'events_range': (2, 8),
                'cost_range': (0.35, 0.55),
                'lead_time_range': (25, 45),
                'quality_base': 0.82,
                'reliability_base': 0.83
            },
            'India': {
                'temp_range': (22, 30),
                'rainfall_range': (1000, 2000),
                'events_range': (3, 11),
                'cost_range': (0.30, 0.50),
                'lead_time_range': (30, 50),
                'quality_base': 0.78,
                'reliability_base': 0.80
            },
            'Thailand': {
                'temp_range': (24, 32),
                'rainfall_range': (1200, 2200),
                'events_range': (2, 9),
                'cost_range': (0.40, 0.60),
                'lead_time_range': (25, 40),
                'quality_base': 0.85,
                'reliability_base': 0.82
            },
            'Vietnam': {
                'temp_range': (22, 30),
                'rainfall_range': (1500, 2500),
                'events_range': (4, 12),
                'cost_range': (0.32, 0.52),
                'lead_time_range': (30, 50),
                'quality_base': 0.80,
                'reliability_base': 0.78
            },
            'Bangladesh': {
                'temp_range': (20, 30),
                'rainfall_range': (1500, 3000),
                'events_range': (5, 15),
                'cost_range': (0.28, 0.48),
                'lead_time_range': (35, 55),
                'quality_base': 0.75,
                'reliability_base': 0.72
            }
        }
    },
    'Palm Oil': {
        'countries': ['Indonesia', 'Malaysia', 'Thailand', 'Colombia', 'Nigeria'],
        'country_profiles': {
            'Indonesia': {
                'temp_range': (24, 30),
                'rainfall_range': (2000, 3500),
                'events_range': (3, 10),
                'cost_range': (0.65, 0.95),
                'lead_time_range': (30, 50),
                'quality_base': 0.78,
                'reliability_base': 0.75
            },
            'Malaysia': {
                'temp_range': (24, 32),
                'rainfall_range': (2000, 3000),
                'events_range': (2, 8),
                'cost_range': (0.70, 1.00),
                'lead_time_range': (25, 45),
                'quality_base': 0.82,
                'reliability_base': 0.80
            },
            'Thailand': {
                'temp_range': (24, 32),
                'rainfall_range': (1200, 2200),
                'events_range': (2, 9),
                'cost_range': (0.68, 0.98),
                'lead_time_range': (28, 48),
                'quality_base': 0.80,
                'reliability_base': 0.78
            },
            'Colombia': {
                'temp_range': (22, 28),
                'rainfall_range': (1500, 2500),
                'events_range': (1, 7),
                'cost_range': (0.72, 1.05),
                'lead_time_range': (35, 55),
                'quality_base': 0.85,
                'reliability_base': 0.82
            },
            'Nigeria': {
                'temp_range': (24, 32),
                'rainfall_range': (1200, 2500),
                'events_range': (3, 11),
                'cost_range': (0.60, 0.90),
                'lead_time_range': (40, 60),
                'quality_base': 0.75,
                'reliability_base': 0.70
            }
        }
    }
}

def generate_supplier_dataset(commodity='Coffee', seed=42):
    """
    Generate realistic dataset for suppliers of a given commodity
    
    Parameters:
    -----------
    commodity : str
        Commodity name (Coffee, Cocoa, Cotton, Rice, Palm Oil)
    seed : int
        Random seed for reproducibility
    
    Returns:
    --------
    DataFrame with supplier data
    """
    
    if commodity not in COMMODITY_CONFIGS:
        raise ValueError(f"Commodity '{commodity}' not supported. Available: {list(COMMODITY_CONFIGS.keys())}")
    
    np.random.seed(seed)  # For reproducibility
    
    config = COMMODITY_CONFIGS[commodity]
    countries = config['countries']
    country_profiles = config['country_profiles']
    
    # Generate data for each supplier (multiple entries per country for training)
    num_samples_per_country = 50
    total_samples = len(countries) * num_samples_per_country
    
    data = {
        'commodity': [],
        'supplier_id': [],
        'country': [],
        'supplier_name': [],
        'avg_temperature_celsius': [],
        'rainfall_mm_per_year': [],
        'extreme_events_per_year': [],
        'cost_per_unit_usd': [],
        'lead_time_days': [],
        'quality_score': [],
        'reliability_score': [],
        'climate_risk_level': []  # This will be our target variable
    }
    
    sample_id = 1
    for country in countries:
        profile = country_profiles[country]
        
        for i in range(num_samples_per_country):
            # Generate climate features
            temp = np.random.uniform(*profile['temp_range'])
            rainfall = np.random.uniform(*profile['rainfall_range'])
            events = np.random.poisson(profile['events_range'][0] + 
                                      (profile['events_range'][1] - profile['events_range'][0]) / 2)
            events = max(profile['events_range'][0], min(events, profile['events_range'][1]))
            
            # Determine climate risk based on features (rule-based for training data)
            risk_score = 0
            if temp > 28:
                risk_score += 1
            if rainfall < 800 or rainfall > 2800:
                risk_score += 1
            if events >= 8:
                risk_score += 1
            if events >= 5:
                risk_score += 0.5
            
            # Assign risk level
            if risk_score >= 2:
                risk_level = 'High'
            elif risk_score >= 1:
                risk_level = 'Medium'
            else:
                risk_level = 'Low'
            
            # Generate supply chain features
            cost = np.random.uniform(*profile['cost_range'])
            lead_time = int(np.random.uniform(*profile['lead_time_range']))
            quality = np.clip(np.random.normal(profile['quality_base'], 0.1), 0.5, 1.0)
            reliability = np.clip(np.random.normal(profile['reliability_base'], 0.1), 0.5, 1.0)
            
            # Add supplier name
            supplier_name = f"{country}_Supplier_{i+1}"
            
            data['commodity'].append(commodity)
            data['supplier_id'].append(sample_id)
            data['country'].append(country)
            data['supplier_name'].append(supplier_name)
            data['avg_temperature_celsius'].append(round(temp, 2))
            data['rainfall_mm_per_year'].append(round(rainfall, 2))
            data['extreme_events_per_year'].append(int(events))
            data['cost_per_unit_usd'].append(round(cost, 2))
            data['lead_time_days'].append(lead_time)
            data['quality_score'].append(round(quality, 3))
            data['reliability_score'].append(round(reliability, 3))
            data['climate_risk_level'].append(risk_level)
            
            sample_id += 1
    
    df = pd.DataFrame(data)
    
    # Add some noise and missing values (5% missing in random cells)
    missing_indices = np.random.choice(len(df), size=int(len(df) * 0.05), replace=False)
    for idx in missing_indices:
        col = np.random.choice(['rainfall_mm_per_year', 'extreme_events_per_year', 'reliability_score'])
        df.loc[idx, col] = np.nan
    
    return df

if __name__ == "__main__":
    import sys
    commodity = sys.argv[1] if len(sys.argv) > 1 else 'Coffee'
    print(f"Generating supplier dataset for {commodity}...")
    df = generate_supplier_dataset(commodity=commodity)
    
    # Save to CSV
    output_path = f'../data/supplier_data_{commodity.lower().replace(" ", "_")}.csv'
    df.to_csv(output_path, index=False)
    print(f"Dataset saved to {output_path}")
    print(f"Dataset shape: {df.shape}")
    print(f"\nFirst few rows:")
    print(df.head())
    print(f"\nClimate risk distribution:")
    print(df['climate_risk_level'].value_counts())
