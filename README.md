
# Food Sustainability Decision Dashboard

Portfolio-grade Streamlit dashboard for evaluating environmental impacts of food production, simulating policy interventions, and interpreting model predictions with SHAP and LIME.

## What this app does

- Loads the **Food_Production.csv** dataset
- Uses lifecycle production stages to model total greenhouse gas emissions
- Simulates policy scenarios by reducing emissions at selected lifecycle stages
- Builds a **Sustainability Index (0–100)** using a weighted combination of:
  - greenhouse gas emissions
  - freshwater withdrawals
  - land use
- Explains model behaviour using:
  - **SHAP** for global and local feature importance
  - **LIME** for case-specific local explanations

## Main sections

- **Overview**  
  Highest-impact foods, food type summary, and lifecycle stage contributions

- **Policy Simulator**  
  Adjust reductions for land use change, animal feed, farm, processing, transport, packaging, and retail stages

- **Sustainability Ranking**  
  Dynamic ranking based on user-defined GHG, water, and land weights

- **Model Explainability**  
  Combined SHAP and LIME interpretation for transparent decision support

## Files

- `app.py` — main Streamlit app
- `Food_Production.csv` — dataset
- `requirements.txt` — Python dependencies
- `.streamlit/config.toml` — theme config for a cleaner UI

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy on Streamlit Community Cloud

1. Push these files to a GitHub repository
2. Open Streamlit Community Cloud
3. Create a new app from the repo
4. Set `app.py` as the entry point
5. Deploy

## Suggested repository name

`food-sustainability-dashboard`

## Suggested portfolio description

An interactive decision-support dashboard that models food-related greenhouse gas emissions, simulates policy interventions across lifecycle stages, and ranks foods using a composite sustainability index based on emissions, water use, and land use.

## Important note

This project is built on the uploaded food production dataset and uses a Random Forest model for predictive and interpretability purposes. Explanations are model-based and should not be treated as causal claims without external validation.
