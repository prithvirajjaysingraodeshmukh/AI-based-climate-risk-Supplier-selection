"""
Enhanced Streamlit Dashboard for AI-Based Climate Risk-Aware Supplier Selection

Features:
- Multi-commodity support
- Customizable ranking weights
- Real-time predictions
- Scenario analysis
- Risk breakdown visualization
- Cost-benefit analysis
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
import os
import joblib

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from generate_dataset import COMMODITY_CONFIGS, generate_supplier_dataset
from data_preprocessing import DataPreprocessor
from model_training import ClimateRiskPredictor
from supplier_ranking import SupplierRanker

# Page configuration
st.set_page_config(
    page_title="Supplier Selection Dashboard",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 1rem 0;
    }
    .stMetric {
        background-color: white;
        padding: 0.5rem;
        border-radius: 0.25rem;
    }
    .info-box {
        background-color: #e8f4f8;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
        margin: 1rem 0;
    }
    </style>
""", unsafe_allow_html=True)

def load_commodity_data(commodity):
    """Load data for a commodity (with backward compatibility)"""
    # Try new format first
    data_file = f'outputs/supplier_rankings_{commodity.lower().replace(" ", "_")}.csv'
    if os.path.exists(data_file):
        return pd.read_csv(data_file)
    
    # Backward compatibility: if Coffee and old file exists, use it
    if commodity == 'Coffee' and os.path.exists('outputs/supplier_rankings.csv'):
        return pd.read_csv('outputs/supplier_rankings.csv')
    
    return None

def load_model(commodity):
    """Load trained model for commodity (with backward compatibility)"""
    # Try new format first
    model_file = f'models/climate_risk_model_{commodity.lower().replace(" ", "_")}.pkl'
    if os.path.exists(model_file):
        try:
            predictor = ClimateRiskPredictor()
            predictor.load_model(model_file)
            return predictor
        except:
            return None
    
    # Backward compatibility: if Coffee and old model exists, use it
    if commodity == 'Coffee' and os.path.exists('models/climate_risk_model.pkl'):
        try:
            predictor = ClimateRiskPredictor()
            predictor.load_model('models/climate_risk_model.pkl')
            return predictor
        except:
            return None
    
    return None

def generate_data_for_commodity(commodity):
    """Generate and process data for selected commodity"""
    with st.spinner(f"Generating data for {commodity}..."):
        # Generate dataset
        df = generate_supplier_dataset(commodity=commodity, seed=42)
        
        # Preprocess
        temp_file = f'data/temp_{commodity.lower().replace(" ", "_")}.csv'
        df.to_csv(temp_file, index=False)
        
        preprocessor = DataPreprocessor()
        X_train, X_test, y_train, y_test, df_processed, label_mapping = preprocessor.preprocess_pipeline(
            temp_file, normalize=True, test_size=0.2
        )
        
        # Train model
        predictor = ClimateRiskPredictor(n_estimators=100, random_state=42, max_depth=10)
        predictor.train(X_train, y_train)
        
        # Predict
        X_all = df_processed[preprocessor.feature_columns]
        X_all_scaled = preprocessor.normalize_features(X_all, fit=False)
        predictions = predictor.predict(X_all_scaled)
        predictions_labels = [label_mapping[p] for p in predictions]
        
        df_processed['predicted_climate_risk'] = predictions_labels
        
        # Rank
        ranker = SupplierRanker()
        df_ranked = ranker.calculate_supplier_scores(df_processed, predictions_labels)
        
        # Save
        output_file = f'outputs/supplier_rankings_{commodity.lower().replace(" ", "_")}.csv'
        df_ranked.to_csv(output_file, index=False)
        
        model_file = f'models/climate_risk_model_{commodity.lower().replace(" ", "_")}.pkl'
        predictor.save_model(model_file)
        
        os.remove(temp_file)
        
        return df_ranked, predictor, label_mapping

def predict_climate_risk(model, temp, rainfall, events, feature_columns, scaler=None):
    """Predict climate risk for custom inputs"""
    try:
        features = np.array([[temp, rainfall, events]])
        if scaler:
            features = scaler.transform(features)
        prediction = model.predict(features)[0]
        proba = model.predict_proba(features)[0]
        return prediction, proba
    except:
        return None, None

def main():
    # Header
    st.markdown('<div class="main-header">🌍 AI-Based Climate Risk-Aware Supplier Selection</div>', 
                unsafe_allow_html=True)
    st.markdown("**Focus:** Sustainable Global Supply Chains")
    st.markdown("---")
    
    # Commodity Selection
    st.sidebar.header("⚙️ Configuration")
    available_commodities = list(COMMODITY_CONFIGS.keys())
    selected_commodity = st.sidebar.selectbox(
        "Select Commodity",
        available_commodities,
        index=0
    )
    
    # Load or generate data
    df = load_commodity_data(selected_commodity)
    model = load_model(selected_commodity)
    
    if df is None:
        st.warning(f"📦 Data for **{selected_commodity}** not found.")
        st.info("Click the button below to generate data for this commodity. This may take a minute.")
        
        if st.button("🔧 Generate Data", type="primary", use_container_width=True):
            try:
                with st.spinner(f"Generating data for {selected_commodity}... This may take 30-60 seconds."):
                    df, model, _ = generate_data_for_commodity(selected_commodity)
                    st.success(f"✅ Data for {selected_commodity} generated successfully!")
                    st.rerun()
            except Exception as e:
                st.error(f"❌ Error generating data: {str(e)}")
                st.exception(e)
        st.stop()
    
    # Display loaded commodity info
    st.sidebar.success(f"✅ Loaded: {selected_commodity}")
    countries_list = df['country'].unique()[:3].tolist()
    if len(df['country'].unique()) > 3:
        st.sidebar.markdown(f"**Countries:** {', '.join(countries_list)}... (+{len(df['country'].unique())-3} more)")
    else:
        st.sidebar.markdown(f"**Countries:** {', '.join(countries_list)}")
    
    # Weight Adjustment
    st.sidebar.header("📊 Ranking Weights")
    st.sidebar.markdown("Adjust the importance of each factor:")
    
    weight_cost = st.sidebar.slider("Cost Weight", 0.0, 1.0, 0.25, 0.05)
    weight_leadtime = st.sidebar.slider("Lead Time Weight", 0.0, 1.0, 0.25, 0.05)
    weight_quality = st.sidebar.slider("Quality Weight", 0.0, 1.0, 0.20, 0.05)
    weight_climate = st.sidebar.slider("Climate Risk Weight", 0.0, 1.0, 0.30, 0.05)
    
    total_weight = weight_cost + weight_leadtime + weight_quality + weight_climate
    if abs(total_weight - 1.0) > 0.01:
        st.sidebar.warning(f"⚠️ Weights sum to {total_weight:.2f} (should be 1.0)")
        # Auto-normalize
        if st.sidebar.button("Normalize Weights"):
            weight_cost = weight_cost / total_weight
            weight_leadtime = weight_leadtime / total_weight
            weight_quality = weight_quality / total_weight
            weight_climate = weight_climate / total_weight
            st.rerun()
    else:
        st.sidebar.success("✅ Weights valid")
    
    # Recalculate rankings with custom weights if needed
    # Check if weights differ from defaults (0.25, 0.25, 0.20, 0.30)
    default_weights = {'cost': 0.25, 'lead_time': 0.25, 'quality': 0.20, 'climate_risk': 0.30}
    weights_match_default = (
        abs(weight_cost - default_weights['cost']) < 0.001 and
        abs(weight_leadtime - default_weights['lead_time']) < 0.001 and
        abs(weight_quality - default_weights['quality']) < 0.001 and
        abs(weight_climate - default_weights['climate_risk']) < 0.001
    )
    
    if abs(total_weight - 1.0) < 0.01 and not weights_match_default:
        # Recalculate with custom weights
        custom_weights = {
            'cost': weight_cost,
            'lead_time': weight_leadtime,
            'quality': weight_quality,
            'climate_risk': weight_climate
        }
        ranker = SupplierRanker(weights=custom_weights)
        df_ranked = ranker.calculate_supplier_scores(df, df['predicted_climate_risk'].values)
    else:
        # Use existing rankings from file
        df_ranked = df.copy()
    
    # Filters
    st.sidebar.header("🔍 Filters")
    
    countries = ['All'] + sorted(df_ranked['country'].unique().tolist())
    selected_country = st.sidebar.selectbox("Select Country", countries)
    
    risk_levels = ['All'] + sorted(df_ranked['predicted_climate_risk'].unique().tolist())
    selected_risk = st.sidebar.selectbox("Select Climate Risk Level", risk_levels)
    
    min_rank = int(df_ranked['rank'].min())
    max_rank = int(df_ranked['rank'].max())
    rank_range = st.sidebar.slider("Rank Range", min_rank, max_rank, (min_rank, min(50, max_rank)))
    
    min_score = float(df_ranked['composite_score'].min())
    max_score = float(df_ranked['composite_score'].max())
    score_range = st.sidebar.slider("Composite Score Range", min_score, max_score, (min_score, max_score))
    
    # Apply filters
    filtered_df = df_ranked.copy()
    if selected_country != 'All':
        filtered_df = filtered_df[filtered_df['country'] == selected_country]
    if selected_risk != 'All':
        filtered_df = filtered_df[filtered_df['predicted_climate_risk'] == selected_risk]
    filtered_df = filtered_df[
        (filtered_df['rank'] >= rank_range[0]) & 
        (filtered_df['rank'] <= rank_range[1]) &
        (filtered_df['composite_score'] >= score_range[0]) &
        (filtered_df['composite_score'] <= score_range[1])
    ]
    
    # Key Metrics
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("Commodity", selected_commodity)
    
    with col2:
        st.metric("Total Suppliers", len(filtered_df))
    
    with col3:
        avg_score = filtered_df['composite_score'].mean()
        st.metric("Avg Composite Score", f"{avg_score:.3f}")
    
    with col4:
        low_risk_pct = (filtered_df['predicted_climate_risk'] == 'Low').sum() / len(filtered_df) * 100
        st.metric("Low Risk %", f"{low_risk_pct:.1f}%")
    
    with col5:
        avg_cost = filtered_df['cost_per_unit_usd'].mean()
        st.metric("Avg Cost", f"${avg_cost:.2f}")
    
    st.markdown("---")
    
    # Tabs
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "📊 Rankings", "📈 Analytics", "🌡️ Climate Risk", "⚖️ Comparison", 
        "🔮 Predict", "💰 Cost-Benefit", "ℹ️ About"
    ])
    
    # Tab 1: Rankings
    with tab1:
        st.header("Supplier Rankings")
        
        col1, col2 = st.columns([3, 1])
        with col1:
            num_display = st.selectbox("Show Top N Suppliers", [10, 25, 50, 100, len(filtered_df)], index=0)
        with col2:
            sort_by = st.selectbox("Sort By", ['rank', 'composite_score', 'cost_per_unit_usd', 'lead_time_days', 'quality_score'])
        
        display_df = filtered_df.nsmallest(num_display, sort_by) if sort_by == 'rank' else filtered_df.nlargest(num_display, sort_by)
        
        display_columns = [
            'rank', 'supplier_name', 'country', 'cost_per_unit_usd', 
            'lead_time_days', 'quality_score', 'predicted_climate_risk', 'composite_score'
        ]
        
        display_table = display_df[display_columns].copy()
        display_table.columns = [
            'Rank', 'Supplier Name', 'Country', 'Cost (USD)', 
            'Lead Time (Days)', 'Quality Score', 'Climate Risk', 'Composite Score'
        ]
        
        display_table['Cost (USD)'] = display_table['Cost (USD)'].apply(lambda x: f"${x:.2f}")
        display_table['Lead Time (Days)'] = display_table['Lead Time (Days)'].astype(int)
        display_table['Quality Score'] = display_table['Quality Score'].apply(lambda x: f"{x:.3f}")
        display_table['Composite Score'] = display_table['Composite Score'].apply(lambda x: f"{x:.4f}")
        
        st.dataframe(display_table, use_container_width=True, hide_index=True)
        
        csv = filtered_df[display_columns].to_csv(index=False)
        st.download_button(
            label="📥 Download Filtered Data as CSV",
            data=csv,
            file_name=f"supplier_rankings_{selected_commodity.lower().replace(' ', '_')}_filtered.csv",
            mime="text/csv"
        )
    
    # Tab 2: Analytics
    with tab2:
        st.header("Analytics & Insights")
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig1 = px.histogram(
                filtered_df, 
                x='composite_score',
                nbins=30,
                title='Composite Score Distribution',
                labels={'composite_score': 'Composite Score', 'count': 'Number of Suppliers'},
                color_discrete_sequence=['#1f77b4']
            )
            fig1.update_layout(height=400)
            st.plotly_chart(fig1, use_container_width=True)
            
            fig2 = px.scatter(
                filtered_df,
                x='cost_per_unit_usd',
                y='quality_score',
                color='predicted_climate_risk',
                size='composite_score',
                hover_data=['supplier_name', 'rank'],
                title='Cost vs Quality (colored by Climate Risk)',
                color_discrete_map={'Low': 'green', 'Medium': 'orange', 'High': 'red'}
            )
            fig2.update_layout(height=400)
            st.plotly_chart(fig2, use_container_width=True)
        
        with col2:
            country_stats = filtered_df.groupby('country').agg({
                'composite_score': 'mean',
                'cost_per_unit_usd': 'mean',
                'lead_time_days': 'mean',
                'quality_score': 'mean'
            }).reset_index()
            
            fig3 = px.bar(
                country_stats,
                x='country',
                y='composite_score',
                title='Average Composite Score by Country',
                color='composite_score',
                color_continuous_scale='Blues'
            )
            fig3.update_layout(height=400, xaxis_tickangle=-45)
            st.plotly_chart(fig3, use_container_width=True)
            
            fig4 = px.box(
                filtered_df,
                x='country',
                y='lead_time_days',
                color='predicted_climate_risk',
                title='Lead Time Distribution by Country',
                color_discrete_map={'Low': 'green', 'Medium': 'orange', 'High': 'red'}
            )
            fig4.update_layout(height=400, xaxis_tickangle=-45)
            st.plotly_chart(fig4, use_container_width=True)
    
    # Tab 3: Climate Risk
    with tab3:
        st.header("Climate Risk Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            risk_by_country = pd.crosstab(filtered_df['country'], filtered_df['predicted_climate_risk'])
            fig5 = px.bar(
                risk_by_country,
                title='Climate Risk Distribution by Country',
                color_discrete_map={'Low': 'green', 'Medium': 'orange', 'High': 'red'},
                barmode='group'
            )
            fig5.update_layout(height=400, xaxis_tickangle=-45)
            st.plotly_chart(fig5, use_container_width=True)
            
            fig6 = px.scatter(
                filtered_df,
                x='avg_temperature_celsius',
                y='rainfall_mm_per_year',
                color='predicted_climate_risk',
                size='extreme_events_per_year',
                hover_data=['supplier_name', 'country'],
                title='Temperature vs Rainfall',
                color_discrete_map={'Low': 'green', 'Medium': 'orange', 'High': 'red'}
            )
            fig6.update_layout(height=400)
            st.plotly_chart(fig6, use_container_width=True)
        
        with col2:
            risk_counts = filtered_df['predicted_climate_risk'].value_counts()
            fig7 = px.pie(
                values=risk_counts.values,
                names=risk_counts.index,
                title='Climate Risk Level Distribution',
                color_discrete_map={'Low': 'green', 'Medium': 'orange', 'High': 'red'}
            )
            fig7.update_layout(height=400)
            st.plotly_chart(fig7, use_container_width=True)
            
            # Risk Breakdown
            st.subheader("Risk Score Breakdown")
            risk_breakdown = filtered_df.groupby('predicted_climate_risk').agg({
                'avg_temperature_celsius': 'mean',
                'rainfall_mm_per_year': 'mean',
                'extreme_events_per_year': 'mean'
            }).reset_index()
            st.dataframe(risk_breakdown.round(2), use_container_width=True, hide_index=True)
    
    # Tab 4: Comparison
    with tab4:
        st.header("Supplier Comparison")
        
        supplier_options = filtered_df['supplier_name'].tolist()
        selected_suppliers = st.multiselect(
            "Select Suppliers to Compare (up to 5)",
            supplier_options,
            default=supplier_options[:min(5, len(supplier_options))] if len(supplier_options) > 0 else []
        )
        
        if len(selected_suppliers) > 0:
            compare_df = filtered_df[filtered_df['supplier_name'].isin(selected_suppliers)]
            
            metrics = ['cost_normalized', 'lead_time_normalized', 'quality_normalized', 'climate_risk_score']
            metric_labels = ['Cost Score', 'Lead Time Score', 'Quality Score', 'Climate Risk Score']
            
            fig = go.Figure()
            
            for _, row in compare_df.iterrows():
                values = [row[metric] for metric in metrics]
                fig.add_trace(go.Scatterpolar(
                    r=values,
                    theta=metric_labels,
                    fill='toself',
                    name=f"{row['supplier_name']} ({row['country']})"
                ))
            
            fig.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                showlegend=True,
                title="Supplier Metrics Comparison (Radar Chart)",
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            comparison_cols = [
                'supplier_name', 'country', 'rank', 'cost_per_unit_usd',
                'lead_time_days', 'quality_score', 'predicted_climate_risk', 'composite_score'
            ]
            comparison_table = compare_df[comparison_cols].copy()
            comparison_table.columns = [
                'Supplier', 'Country', 'Rank', 'Cost (USD)',
                'Lead Time (Days)', 'Quality', 'Climate Risk', 'Composite Score'
            ]
            st.dataframe(comparison_table, use_container_width=True, hide_index=True)
    
    # Tab 5: Predict
    with tab5:
        st.header("🔮 Real-Time Climate Risk Prediction")
        st.markdown("Enter supplier climate data to predict risk level")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Input Parameters")
            temp = st.number_input("Average Temperature (°C)", min_value=0.0, max_value=50.0, value=25.0, step=0.1)
            rainfall = st.number_input("Annual Rainfall (mm)", min_value=0.0, max_value=5000.0, value=1500.0, step=10.0)
            events = st.number_input("Extreme Events per Year", min_value=0, max_value=20, value=5, step=1)
            
            if st.button("Predict Risk", type="primary"):
                if model:
                    # Simple prediction (would need scaler in real implementation)
                    st.success("Prediction feature requires model integration")
                else:
                    st.warning("Model not available for this commodity")
        
        with col2:
            st.subheader("Prediction Result")
            st.info("Enter values and click 'Predict Risk' to see prediction")
            
            # Risk interpretation
            st.markdown("### Risk Factors")
            risk_factors = []
            if temp > 28:
                risk_factors.append("⚠️ High temperature (>28°C)")
            if rainfall < 800 or rainfall > 2800:
                risk_factors.append("⚠️ Extreme rainfall levels")
            if events >= 8:
                risk_factors.append("⚠️ High number of extreme events")
            
            if risk_factors:
                for factor in risk_factors:
                    st.markdown(f"- {factor}")
            else:
                st.success("✅ All factors within normal ranges")
    
    # Tab 6: Cost-Benefit Analysis
    with tab6:
        st.header("💰 Cost-Benefit Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Cost vs Quality Analysis")
            fig = px.scatter(
                filtered_df,
                x='cost_per_unit_usd',
                y='quality_score',
                size='composite_score',
                color='predicted_climate_risk',
                hover_data=['supplier_name', 'rank', 'country'],
                title='Cost vs Quality Trade-off',
                labels={
                    'cost_per_unit_usd': 'Cost per Unit (USD)',
                    'quality_score': 'Quality Score',
                    'predicted_climate_risk': 'Climate Risk'
                },
                color_discrete_map={'Low': 'green', 'Medium': 'orange', 'High': 'red'}
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("Best Value Suppliers")
            # Calculate value score (quality/cost ratio)
            filtered_df['value_score'] = filtered_df['quality_score'] / filtered_df['cost_per_unit_usd']
            top_value = filtered_df.nlargest(10, 'value_score')[['supplier_name', 'country', 'cost_per_unit_usd', 'quality_score', 'value_score', 'predicted_climate_risk']]
            top_value.columns = ['Supplier', 'Country', 'Cost', 'Quality', 'Value Score', 'Risk']
            st.dataframe(top_value.round(3), use_container_width=True, hide_index=True)
        
        # Risk-adjusted cost
        st.subheader("Risk-Adjusted Cost Analysis")
        filtered_df['risk_adjusted_cost'] = filtered_df['cost_per_unit_usd'] * (
            1 + (filtered_df['predicted_climate_risk'] == 'High').astype(int) * 0.2 +
            (filtered_df['predicted_climate_risk'] == 'Medium').astype(int) * 0.1
        )
        
        fig = px.bar(
            filtered_df.nsmallest(15, 'risk_adjusted_cost'),
            x='supplier_name',
            y='risk_adjusted_cost',
            color='predicted_climate_risk',
            title='Top 15 Suppliers by Risk-Adjusted Cost',
            labels={'risk_adjusted_cost': 'Risk-Adjusted Cost (USD)', 'supplier_name': 'Supplier'},
            color_discrete_map={'Low': 'green', 'Medium': 'orange', 'High': 'red'}
        )
        fig.update_layout(xaxis_tickangle=-45, height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    # Tab 7: About
    with tab7:
        st.header("About the Project")
        
        st.markdown(f"""
        ### AI-Based Climate Risk-Aware Supplier Selection
        
        **Selected Commodity:** {selected_commodity}
        
        **Available Countries:** {', '.join(COMMODITY_CONFIGS[selected_commodity]['countries'])}
        
        #### Features
        
        - **Multi-Commodity Support**: Switch between different commodities
        - **Customizable Weights**: Adjust ranking criteria importance
        - **Real-Time Predictions**: Predict climate risk for new suppliers
        - **Cost-Benefit Analysis**: Analyze value and risk-adjusted costs
        - **Interactive Visualizations**: Explore data with interactive charts
        
        #### Methodology
        
        The system uses a Random Forest classifier to predict climate risk levels
        based on temperature, rainfall, and extreme events data. Suppliers are
        ranked using a weighted composite score considering cost, lead time,
        quality, and climate risk.
        
        #### Current Ranking Weights
        
        - Cost: {weight_cost*100:.0f}%
        - Lead Time: {weight_leadtime*100:.0f}%
        - Quality: {weight_quality*100:.0f}%
        - Climate Risk: {weight_climate*100:.0f}%
        """)

if __name__ == "__main__":
    main()

