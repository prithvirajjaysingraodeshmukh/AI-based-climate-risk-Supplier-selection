# New Features & Enhancements

## Overview

The dashboard has been significantly enhanced with multi-commodity support and advanced features for better supplier selection analysis.

## ✨ New Features

### 1. **Multi-Commodity Support** 🛒

**Available Commodities:**
- ☕ **Coffee** (Brazil, Colombia, Vietnam, Ethiopia, Honduras)
- 🍫 **Cocoa** (Ivory Coast, Ghana, Indonesia, Ecuador, Cameroon)
- 🌾 **Cotton** (China, India, United States, Brazil, Pakistan)
- 🍚 **Rice** (China, India, Thailand, Vietnam, Bangladesh)
- 🛢️ **Palm Oil** (Indonesia, Malaysia, Thailand, Colombia, Nigeria)

**How to Use:**
- Select commodity from the sidebar dropdown
- Dashboard automatically loads/generates data for selected commodity
- Each commodity has country-specific supplier profiles

### 2. **Customizable Ranking Weights** ⚖️

**Features:**
- Interactive sliders to adjust weight of each factor:
  - Cost (default: 25%)
  - Lead Time (default: 25%)
  - Quality (default: 20%)
  - Climate Risk (default: 30%)
- Real-time recalculation of rankings
- Automatic weight normalization
- Visual validation (weights must sum to 1.0)

**Use Case:**
- Prioritize cost for budget-constrained projects
- Emphasize climate risk for sustainability-focused sourcing
- Balance factors based on business priorities

### 3. **Real-Time Climate Risk Prediction** 🔮

**Features:**
- Input custom supplier climate parameters:
  - Average temperature (°C)
  - Annual rainfall (mm)
  - Extreme events per year
- Instant risk assessment
- Risk factor identification
- Visual feedback on risk indicators

**Use Case:**
- Evaluate new suppliers before adding to database
- Assess risk for specific regions
- Scenario planning for climate changes

### 4. **Cost-Benefit Analysis** 💰

**New Tab Features:**
- **Cost vs Quality Trade-off Analysis**
  - Scatter plot showing cost-quality relationship
  - Color-coded by climate risk
  - Sized by composite score
  
- **Best Value Suppliers**
  - Value score calculation (quality/cost ratio)
  - Top 10 value suppliers table
  - Quick identification of cost-effective options

- **Risk-Adjusted Cost Analysis**
  - Automatic cost adjustment based on climate risk
  - High risk: +20% cost penalty
  - Medium risk: +10% cost penalty
  - Bar chart of top 15 suppliers by risk-adjusted cost

**Use Case:**
- Identify best value suppliers
- Account for hidden climate-related costs
- Make informed cost decisions considering risk

### 5. **Enhanced Risk Breakdown** 📊

**Features:**
- Detailed risk score breakdown table
- Average metrics by risk level:
  - Temperature
  - Rainfall
  - Extreme events
- Visual risk distribution charts
- Country-level risk analysis

### 6. **Improved Visualizations** 📈

**Enhancements:**
- All charts are interactive (Plotly)
- Hover tooltips with detailed information
- Zoom and pan capabilities
- Export charts as images
- Better color schemes for accessibility

### 7. **Enhanced Data Management** 💾

**Features:**
- Commodity-specific data files
- Automatic data generation if missing
- Cached data loading for performance
- Organized file structure:
  - `outputs/supplier_rankings_[commodity].csv`
  - `models/climate_risk_model_[commodity].pkl`
  - `data/supplier_data_[commodity].csv`

## 🚀 Usage Instructions

### Running the Enhanced Dashboard

```bash
# Launch dashboard
streamlit run app.py

# Select commodity from sidebar
# Adjust weights if needed
# Explore different tabs
```

### Generating Data for New Commodity

```bash
# Via command line
python main.py --commodity Cocoa

# Via dashboard
# Just select the commodity - data will be generated automatically if missing
```

## 📋 Dashboard Tabs Overview

1. **📊 Rankings** - Supplier rankings table with filtering
2. **📈 Analytics** - Insights and distribution charts
3. **🌡️ Climate Risk** - Risk analysis and breakdown
4. **⚖️ Comparison** - Side-by-side supplier comparison
5. **🔮 Predict** - Real-time risk prediction for new suppliers
6. **💰 Cost-Benefit** - Value analysis and risk-adjusted costs
7. **ℹ️ About** - Project information and methodology

## 🎯 Key Improvements

### User Experience
- ✅ Intuitive commodity selection
- ✅ Real-time weight adjustments
- ✅ Interactive visualizations
- ✅ Better organization with tabs
- ✅ More actionable insights

### Functionality
- ✅ Multi-commodity support
- ✅ Customizable ranking criteria
- ✅ Predictive capabilities
- ✅ Cost-benefit analysis
- ✅ Risk-adjusted metrics

### Data Quality
- ✅ Realistic commodity-specific profiles
- ✅ Country-specific characteristics
- ✅ Comprehensive feature sets
- ✅ Accurate risk modeling

## 💡 Best Practices

1. **Start with Default Weights**: Use default weights first, then adjust based on business needs
2. **Compare Commodities**: Generate data for multiple commodities to compare sourcing options
3. **Use Risk-Adjusted Costs**: Consider hidden climate-related costs in decisions
4. **Explore Cost-Benefit**: Find best value suppliers, not just cheapest
5. **Validate Predictions**: Use prediction tool to validate new supplier assessments

## 🔄 Migration Notes

- Old data files (`supplier_rankings.csv`) are still compatible
- New commodities require data generation (automatic in dashboard)
- Models are commodity-specific for better accuracy
- All new features are backward compatible

## 📝 Technical Details

- **Framework**: Streamlit with Plotly
- **Data Processing**: Pandas, NumPy
- **ML Model**: Random Forest (per commodity)
- **Caching**: Streamlit caching for performance
- **File Organization**: Commodity-based file naming

## 🎓 Example Workflows

### Workflow 1: Compare Multiple Commodities
1. Select Commodity 1 (e.g., Coffee)
2. Note top suppliers and metrics
3. Switch to Commodity 2 (e.g., Cocoa)
4. Compare rankings and costs
5. Make informed sourcing decision

### Workflow 2: Custom Weight Analysis
1. Select commodity
2. Adjust weights (e.g., 40% climate risk)
3. Observe ranking changes
4. Export filtered results
5. Present findings to stakeholders

### Workflow 3: New Supplier Evaluation
1. Go to Predict tab
2. Enter supplier climate data
3. Review risk assessment
4. Compare with existing suppliers
5. Make sourcing decision

## 🐛 Troubleshooting

**Issue**: Commodity data not found
- **Solution**: Dashboard will automatically generate data, or run `python main.py --commodity [name]`

**Issue**: Weights don't sum to 1.0
- **Solution**: Click "Normalize Weights" button in sidebar

**Issue**: Predictions not working
- **Solution**: Ensure model exists for selected commodity (generate data first)

## 📚 Additional Resources

- See `README.md` for full project documentation
- See `DASHBOARD_GUIDE.md` for dashboard-specific guide
- See `QUICK_START.md` for quick setup instructions

