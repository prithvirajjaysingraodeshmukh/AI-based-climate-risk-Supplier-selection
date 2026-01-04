"""
Streamlit Dashboard for AI-Based Climate Risk-Aware Supplier Selection

Interactive web dashboard for exploring supplier rankings and climate risk predictions
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
import os

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Page configuration
st.set_page_config(
    page_title="Supplier Selection Dashboard",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 1rem 0;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .stMetric {
        background-color: white;
        padding: 0.5rem;
        border-radius: 0.25rem;
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    """Load supplier rankings data"""
    try:
        df = pd.read_csv('outputs/supplier_rankings.csv')
        return df
    except FileNotFoundError:
        st.error("❌ Rankings file not found. Please run `python main.py` first to generate the data.")
        st.stop()
        return None

@st.cache_data
def load_model_info():
    """Load model information if available"""
    try:
        with open('outputs/summary_report.txt', 'r') as f:
            report = f.read()
        return report
    except FileNotFoundError:
        return None

def main():
    # Header
    st.markdown('<div class="main-header">🌍 AI-Based Climate Risk-Aware Supplier Selection</div>', 
                unsafe_allow_html=True)
    st.markdown("**Commodity:** Coffee | **Focus:** Sustainable Global Supply Chains")
    st.markdown("---")
    
    # Load data
    df = load_data()
    if df is None:
        return
    
    # Sidebar filters
    st.sidebar.header("🔍 Filters")
    
    # Country filter
    countries = ['All'] + sorted(df['country'].unique().tolist())
    selected_country = st.sidebar.selectbox("Select Country", countries)
    
    # Risk level filter
    risk_levels = ['All'] + sorted(df['predicted_climate_risk'].unique().tolist())
    selected_risk = st.sidebar.selectbox("Select Climate Risk Level", risk_levels)
    
    # Rank range filter
    min_rank = int(df['rank'].min())
    max_rank = int(df['rank'].max())
    rank_range = st.sidebar.slider("Rank Range", min_rank, max_rank, (min_rank, min(50, max_rank)))
    
    # Composite score filter
    min_score = float(df['composite_score'].min())
    max_score = float(df['composite_score'].max())
    score_range = st.sidebar.slider("Composite Score Range", min_score, max_score, (min_score, max_score))
    
    # Apply filters
    filtered_df = df.copy()
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
    
    # Main content area
    # Key Metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Suppliers", len(filtered_df))
    
    with col2:
        avg_score = filtered_df['composite_score'].mean()
        st.metric("Avg Composite Score", f"{avg_score:.3f}")
    
    with col3:
        low_risk_pct = (filtered_df['predicted_climate_risk'] == 'Low').sum() / len(filtered_df) * 100
        st.metric("Low Risk %", f"{low_risk_pct:.1f}%")
    
    with col4:
        avg_cost = filtered_df['cost_per_unit_usd'].mean()
        st.metric("Avg Cost (USD)", f"${avg_cost:.2f}")
    
    st.markdown("---")
    
    # Tabs for different views
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Rankings", "📈 Analytics", "🌡️ Climate Risk", "⚖️ Comparison", "ℹ️ About"])
    
    # Tab 1: Rankings Table
    with tab1:
        st.header("Supplier Rankings")
        
        # Display options
        col1, col2 = st.columns([3, 1])
        with col1:
            num_display = st.selectbox("Show Top N Suppliers", [10, 25, 50, 100, len(filtered_df)], index=0)
        with col2:
            sort_by = st.selectbox("Sort By", ['rank', 'composite_score', 'cost_per_unit_usd', 'lead_time_days', 'quality_score'])
        
        # Sort data
        display_df = filtered_df.nsmallest(num_display, sort_by) if sort_by == 'rank' else filtered_df.nlargest(num_display, sort_by)
        
        # Select columns to display
        display_columns = [
            'rank', 'supplier_name', 'country', 'cost_per_unit_usd', 
            'lead_time_days', 'quality_score', 'predicted_climate_risk', 'composite_score'
        ]
        
        display_table = display_df[display_columns].copy()
        display_table.columns = [
            'Rank', 'Supplier Name', 'Country', 'Cost (USD)', 
            'Lead Time (Days)', 'Quality Score', 'Climate Risk', 'Composite Score'
        ]
        
        # Format numbers
        display_table['Cost (USD)'] = display_table['Cost (USD)'].apply(lambda x: f"${x:.2f}")
        display_table['Lead Time (Days)'] = display_table['Lead Time (Days)'].astype(int)
        display_table['Quality Score'] = display_table['Quality Score'].apply(lambda x: f"{x:.3f}")
        display_table['Composite Score'] = display_table['Composite Score'].apply(lambda x: f"{x:.4f}")
        
        st.dataframe(display_table, use_container_width=True, hide_index=True)
        
        # Download button
        csv = filtered_df[display_columns].to_csv(index=False)
        st.download_button(
            label="📥 Download Filtered Data as CSV",
            data=csv,
            file_name="supplier_rankings_filtered.csv",
            mime="text/csv"
        )
    
    # Tab 2: Analytics
    with tab2:
        st.header("Analytics & Insights")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Composite score distribution
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
            
            # Cost vs Quality scatter
            fig2 = px.scatter(
                filtered_df,
                x='cost_per_unit_usd',
                y='quality_score',
                color='predicted_climate_risk',
                size='composite_score',
                hover_data=['supplier_name', 'rank'],
                title='Cost vs Quality (colored by Climate Risk)',
                labels={
                    'cost_per_unit_usd': 'Cost per Unit (USD)',
                    'quality_score': 'Quality Score',
                    'predicted_climate_risk': 'Climate Risk'
                },
                color_discrete_map={'Low': 'green', 'Medium': 'orange', 'High': 'red'}
            )
            fig2.update_layout(height=400)
            st.plotly_chart(fig2, use_container_width=True)
        
        with col2:
            # Country comparison
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
                labels={'composite_score': 'Average Composite Score', 'country': 'Country'},
                color='composite_score',
                color_continuous_scale='Blues'
            )
            fig3.update_layout(height=400, xaxis_tickangle=-45)
            st.plotly_chart(fig3, use_container_width=True)
            
            # Lead time distribution
            fig4 = px.box(
                filtered_df,
                x='country',
                y='lead_time_days',
                color='predicted_climate_risk',
                title='Lead Time Distribution by Country',
                labels={'lead_time_days': 'Lead Time (Days)', 'country': 'Country'},
                color_discrete_map={'Low': 'green', 'Medium': 'orange', 'High': 'red'}
            )
            fig4.update_layout(height=400, xaxis_tickangle=-45)
            st.plotly_chart(fig4, use_container_width=True)
    
    # Tab 3: Climate Risk Analysis
    with tab3:
        st.header("Climate Risk Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Risk distribution by country
            risk_by_country = pd.crosstab(filtered_df['country'], filtered_df['predicted_climate_risk'])
            fig5 = px.bar(
                risk_by_country,
                title='Climate Risk Distribution by Country',
                labels={'value': 'Number of Suppliers', 'country': 'Country'},
                color_discrete_map={'Low': 'green', 'Medium': 'orange', 'High': 'red'},
                barmode='group'
            )
            fig5.update_layout(height=400, xaxis_tickangle=-45)
            st.plotly_chart(fig5, use_container_width=True)
            
            # Temperature vs Rainfall scatter
            fig6 = px.scatter(
                filtered_df,
                x='avg_temperature_celsius',
                y='rainfall_mm_per_year',
                color='predicted_climate_risk',
                size='extreme_events_per_year',
                hover_data=['supplier_name', 'country'],
                title='Temperature vs Rainfall (colored by Risk)',
                labels={
                    'avg_temperature_celsius': 'Average Temperature (°C)',
                    'rainfall_mm_per_year': 'Rainfall (mm/year)',
                    'extreme_events_per_year': 'Extreme Events/Year',
                    'predicted_climate_risk': 'Climate Risk'
                },
                color_discrete_map={'Low': 'green', 'Medium': 'orange', 'High': 'red'}
            )
            fig6.update_layout(height=400)
            st.plotly_chart(fig6, use_container_width=True)
        
        with col2:
            # Risk level pie chart
            risk_counts = filtered_df['predicted_climate_risk'].value_counts()
            fig7 = px.pie(
                values=risk_counts.values,
                names=risk_counts.index,
                title='Climate Risk Level Distribution',
                color_discrete_map={'Low': 'green', 'Medium': 'orange', 'High': 'red'}
            )
            fig7.update_layout(height=400)
            st.plotly_chart(fig7, use_container_width=True)
            
            # Extreme events analysis
            events_by_risk = filtered_df.groupby('predicted_climate_risk')['extreme_events_per_year'].mean().reset_index()
            fig8 = px.bar(
                events_by_risk,
                x='predicted_climate_risk',
                y='extreme_events_per_year',
                title='Average Extreme Events by Risk Level',
                labels={
                    'extreme_events_per_year': 'Average Extreme Events/Year',
                    'predicted_climate_risk': 'Climate Risk Level'
                },
                color='predicted_climate_risk',
                color_discrete_map={'Low': 'green', 'Medium': 'orange', 'High': 'red'}
            )
            fig8.update_layout(height=400)
            st.plotly_chart(fig8, use_container_width=True)
    
    # Tab 4: Supplier Comparison
    with tab4:
        st.header("Supplier Comparison")
        
        # Select suppliers to compare
        supplier_options = filtered_df['supplier_name'].tolist()
        selected_suppliers = st.multiselect(
            "Select Suppliers to Compare (up to 5)",
            supplier_options,
            default=supplier_options[:min(5, len(supplier_options))] if len(supplier_options) > 0 else []
        )
        
        if len(selected_suppliers) > 0:
            compare_df = filtered_df[filtered_df['supplier_name'].isin(selected_suppliers)]
            
            # Radar chart for comparison
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
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, 1]
                    )),
                showlegend=True,
                title="Supplier Metrics Comparison (Radar Chart)",
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Comparison table
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
    
    # Tab 5: About
    with tab5:
        st.header("About the Project")
        
        st.markdown("""
        ### AI-Based Climate Risk-Aware Supplier Selection for Sustainable Global Supply Chains
        
        **Commodity:** Coffee  
        **Supplier Countries:** Brazil, Colombia, Vietnam, Ethiopia, Honduras
        
        #### Methodology
        
        This dashboard uses an AI/ML-based decision support system that:
        
        1. **Predicts Climate Risk** using a Random Forest Classifier trained on:
           - Average temperature
           - Annual rainfall
           - Extreme climate events per year
        
        2. **Ranks Suppliers** using a weighted composite score:
           - **Cost:** 25%
           - **Lead Time:** 25%
           - **Quality:** 20%
           - **AI-Predicted Climate Risk:** 30%
        
        3. **Provides Insights** through interactive visualizations and analytics
        
        #### Model Performance
        
        The Random Forest classifier achieves high accuracy in predicting climate risk levels,
        enabling data-driven supplier selection decisions that balance traditional supply chain
        metrics with climate resilience.
        
        #### How to Use
        
        - Use the **Filters** in the sidebar to explore specific countries, risk levels, or score ranges
        - View **Rankings** to see supplier performance
        - Explore **Analytics** for insights on cost, quality, and lead times
        - Analyze **Climate Risk** patterns and distributions
        - **Compare** multiple suppliers side-by-side
        """)
        
        # Model info
        model_info = load_model_info()
        if model_info:
            with st.expander("View Model Details"):
                st.text(model_info)

if __name__ == "__main__":
    main()

