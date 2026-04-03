
import io
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from lime.lime_tabular import LimeTabularExplainer
import shap

st.set_page_config(
    page_title="Food Sustainability Decision Dashboard",
    page_icon="🌍",
    layout="wide"
)

# -------------------------------
# HELPERS
# -------------------------------
@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    original_cols = df.columns.tolist()
    df.columns = (
        df.columns.str.strip()
        .str.replace(" ", "_", regex=False)
        .str.replace("-", "_", regex=False)
        .str.replace("(", "", regex=False)
        .str.replace(")", "", regex=False)
        .str.replace("/", "_per_", regex=False)
        .str.replace("²", "2", regex=False)
        .str.replace("₄", "4", regex=False)
        .str.replace("₂", "2", regex=False)
        .str.replace("₀", "0", regex=False)
        .str.replace("₁", "1", regex=False)
        .str.replace("₃", "3", regex=False)
        .str.replace("₅", "5", regex=False)
        .str.replace("₆", "6", regex=False)
        .str.replace("₇", "7", regex=False)
        .str.replace("₈", "8", regex=False)
        .str.replace("₉", "9", regex=False)
        .str.replace("kgCO₂eq", "kgCO2eq", regex=False)
        .str.replace("gPO₄eq", "gPO4eq", regex=False)
    )
    df.attrs["original_cols"] = original_cols
    return df

def detect_columns(df: pd.DataFrame):
    emission_cols = [
        "Land_use_change",
        "Animal_Feed",
        "Farm",
        "Processing",
        "Transport",
        "Packging",
        "Retail"
    ]
    emission_cols = [c for c in emission_cols if c in df.columns]

    total_col = "Total_emissions" if "Total_emissions" in df.columns else None
    if total_col is None:
        df["Total_emissions"] = df[emission_cols].sum(axis=1)
        total_col = "Total_emissions"

    water_candidates = [c for c in df.columns if "Freshwater_withdrawals_per_kilogram" in c]
    if not water_candidates:
        water_candidates = [c for c in df.columns if "water" in c.lower() and "kilogram" in c.lower()]
    water_col = water_candidates[0]

    land_candidates = [c for c in df.columns if "Land_use_per_kilogram" in c]
    if not land_candidates:
        land_candidates = [c for c in df.columns if "land_use" in c.lower() and "kilogram" in c.lower()]
    land_col = land_candidates[0]

    food_col = "Food_product"

    return emission_cols, total_col, water_col, land_col, food_col

def normalize(series: pd.Series) -> pd.Series:
    denom = series.max() - series.min()
    if denom == 0:
        return pd.Series(np.zeros(len(series)), index=series.index)
    return (series - series.min()) / denom

def classify_food(name: str) -> str:
    value = str(name).lower()
    animal = ["beef", "lamb", "mutton", "pork", "chicken", "poultry", "fish", "prawns", "shrimp", "milk", "cheese", "eggs"]
    plant_protein = ["beans", "peas", "lentils", "tofu", "nuts", "groundnuts"]
    if any(x in value for x in animal):
        return "Animal-based"
    if any(x in value for x in plant_protein):
        return "Plant protein"
    return "Plant-based"

@st.cache_resource
def train_model(X: pd.DataFrame, y: pd.Series):
    model = RandomForestRegressor(
        n_estimators=500,
        max_depth=8,
        min_samples_leaf=1,
        random_state=42
    )
    model.fit(X, y)
    return model

def render_barh(data: pd.DataFrame, x: str, y: str, title: str, xlab: str = ""):
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(data[y], data[x])
    ax.set_title(title)
    ax.set_xlabel(xlab or x)
    ax.invert_yaxis()
    st.pyplot(fig, use_container_width=True)

def shap_bar_plot(shap_values, X):
    fig = plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X, plot_type="bar", show=False)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

def shap_beeswarm_plot(shap_values, X):
    fig = plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X, show=False)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

# -------------------------------
# LOAD AND PREP
# -------------------------------
DATA_PATH = "Food_Production.csv"
df = load_data(DATA_PATH)
emission_cols, total_col, water_col, land_col, food_col = detect_columns(df)

for c in emission_cols + [total_col, water_col, land_col]:
    df[c] = pd.to_numeric(df[c], errors="coerce")

df["Food_Type"] = df[food_col].apply(classify_food)
df["GHG_norm"] = normalize(df[total_col])
df["Water_norm"] = normalize(df[water_col].fillna(df[water_col].median()))
df["Land_norm"] = normalize(df[land_col].fillna(df[land_col].median()))

# -------------------------------
# SIDEBAR CONTROLS
# -------------------------------
st.sidebar.title("Policy Controls")
st.sidebar.caption("Adjust lifecycle-stage reductions and sustainability weights.")

stage_reductions = {}
for col in emission_cols:
    stage_reductions[col] = st.sidebar.slider(f"{col.replace('_', ' ')} reduction (%)", 0, 60, 0, 5)

st.sidebar.markdown("---")
w_ghg = st.sidebar.slider("Weight: GHG", 0.0, 1.0, 0.5, 0.05)
w_water = st.sidebar.slider("Weight: Water", 0.0, 1.0, 0.3, 0.05)
w_land = st.sidebar.slider("Weight: Land", 0.0, 1.0, 0.2, 0.05)

weight_sum = w_ghg + w_water + w_land
if weight_sum == 0:
    w_ghg, w_water, w_land = 0.5, 0.3, 0.2
    weight_sum = 1.0

w_ghg /= weight_sum
w_water /= weight_sum
w_land /= weight_sum

# -------------------------------
# MODEL
# -------------------------------
X = df[emission_cols].fillna(0)
y = df[total_col].fillna(df[total_col].median())
model = train_model(X, y)

pred_baseline = model.predict(X)

scenario_X = X.copy()
for col, pct in stage_reductions.items():
    scenario_X[col] = scenario_X[col] * (1 - pct / 100.0)

pred_scenario = model.predict(scenario_X)

results = df.copy()
results["Predicted_GHG_Baseline"] = pred_baseline
results["Predicted_GHG_Scenario"] = pred_scenario
results["Predicted_Reduction"] = results["Predicted_GHG_Baseline"] - results["Predicted_GHG_Scenario"]

results["Scenario_GHG_norm"] = normalize(results["Predicted_GHG_Scenario"])
results["Environmental_Score"] = (
    w_ghg * results["Scenario_GHG_norm"] +
    w_water * results["Water_norm"] +
    w_land * results["Land_norm"]
)
results["Sustainability_Index"] = (1 - results["Environmental_Score"]) * 100
results["Rank"] = results["Sustainability_Index"].rank(ascending=False, method="dense").astype(int)

rmse = float(np.sqrt(mean_squared_error(y, pred_baseline)))
r2 = float(r2_score(y, pred_baseline))

# -------------------------------
# HEADER
# -------------------------------
st.title("🌍 Food Sustainability Decision Dashboard")
st.markdown(
    """
    **Created by Powell Andile Ndlovu**  
    Decision-support app for comparing food environmental impacts, simulating policy levers, and interpreting model predictions with **SHAP + LIME**.
    This project develops a machine learning–driven decision-support system that evaluates and simulates the environmental impacts of food production, enabling scenario-based policy analysis across carbon, water, and land footprints. 
    Data Source: 
    https://ourworldindata.org

    """
)

m1, m2, m3, m4 = st.columns(4)
m1.metric("Foods", f"{len(results)}")
m2.metric("Model R²", f"{r2:.3f}")
m3.metric("Model RMSE", f"{rmse:.3f}")
m4.metric("Average scenario reduction", f"{results['Predicted_Reduction'].mean():.2f} kgCO2e/kg")

tab1, tab2, tab3, tab4 = st.tabs(
    ["Overview", "Policy Simulator", "Sustainability Ranking", "Model Explainability"]
)

# -------------------------------
# TAB 1 OVERVIEW
# -------------------------------
with tab1:
    left, right = st.columns([1.1, 1])
    with left:
        top_emitters = results[[food_col, total_col]].sort_values(total_col, ascending=False).head(12)
        render_barh(
            top_emitters,
            x=total_col,
            y=food_col,
            title="Highest-impact foods by observed total emissions",
            xlab="kgCO2e per kg product"
        )
    with right:
        food_type_summary = (
            results.groupby("Food_Type")[total_col]
            .agg(["mean", "median", "max"])
            .reset_index()
            .rename(columns={"mean": "Mean_GHG", "median": "Median_GHG", "max": "Max_GHG"})
        )
        st.subheader("Food type summary")
        st.dataframe(food_type_summary, use_container_width=True)

    stage_summary = results[emission_cols].sum().sort_values(ascending=False).reset_index()
    stage_summary.columns = ["Stage", "Total_Contribution"]
    render_barh(
        stage_summary,
        x="Total_Contribution",
        y="Stage",
        title="Lifecycle stages contributing most to emissions",
        xlab="Total kgCO2e contribution across foods"
    )

# -------------------------------
# TAB 2 POLICY SIMULATOR
# -------------------------------
with tab2:
    st.subheader("Scenario impact under selected policy levers")

    c1, c2 = st.columns(2)
    with c1:
        top_reduction = results[[food_col, "Predicted_Reduction", "Predicted_GHG_Baseline", "Predicted_GHG_Scenario"]].sort_values(
            "Predicted_Reduction", ascending=False
        ).head(10)
        render_barh(
            top_reduction,
            x="Predicted_Reduction",
            y=food_col,
            title="Foods with the largest predicted reduction",
            xlab="Predicted reduction in kgCO2e/kg"
        )
    with c2:
        sample_foods = results[[food_col, "Predicted_GHG_Baseline", "Predicted_GHG_Scenario"]].copy()
        sample_foods = sample_foods.sort_values("Predicted_GHG_Baseline", ascending=False).head(8)
        fig, ax = plt.subplots(figsize=(9, 6))
        idx = np.arange(len(sample_foods))
        width = 0.38
        ax.barh(idx - width/2, sample_foods["Predicted_GHG_Baseline"], height=width, label="Baseline")
        ax.barh(idx + width/2, sample_foods["Predicted_GHG_Scenario"], height=width, label="Scenario")
        ax.set_yticks(idx)
        ax.set_yticklabels(sample_foods[food_col])
        ax.set_title("Baseline vs scenario emissions")
        ax.set_xlabel("Predicted kgCO2e/kg")
        ax.legend()
        ax.invert_yaxis()
        st.pyplot(fig, use_container_width=True)

    st.dataframe(
        results[[food_col, "Food_Type", "Predicted_GHG_Baseline", "Predicted_GHG_Scenario", "Predicted_Reduction"]]
        .sort_values("Predicted_Reduction", ascending=False),
        use_container_width=True,
        height=360
    )

# -------------------------------
# TAB 3 SUSTAINABILITY RANKING
# -------------------------------
with tab3:
    st.subheader("Composite sustainability scoring")
    st.caption(
        f"Weights after normalization → GHG: {w_ghg:.2f}, Water: {w_water:.2f}, Land: {w_land:.2f}"
    )

    top_ranked = results[[food_col, "Food_Type", "Sustainability_Index", "Rank", total_col, water_col, land_col]].sort_values(
        "Sustainability_Index", ascending=False
    ).head(15)

    bottom_ranked = results[[food_col, "Food_Type", "Sustainability_Index", "Rank"]].sort_values(
        "Sustainability_Index", ascending=True
    ).head(10)

    c1, c2 = st.columns(2)
    with c1:
        render_barh(
            top_ranked,
            x="Sustainability_Index",
            y=food_col,
            title="Top foods by sustainability index",
            xlab="Index score (0–100)"
        )
    with c2:
        render_barh(
            bottom_ranked,
            x="Sustainability_Index",
            y=food_col,
            title="Lowest-ranked foods",
            xlab="Index score (0–100)"
        )

    st.dataframe(
        results[[food_col, "Food_Type", total_col, water_col, land_col, "Sustainability_Index", "Rank"]]
        .sort_values("Rank"),
        use_container_width=True,
        height=360
    )

# -------------------------------
# TAB 4 EXPLAINABILITY
# -------------------------------
with tab4:
    st.subheader("SHAP + LIME combined interpretation")

    shap_values = shap.TreeExplainer(model).shap_values(X)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Global importance: SHAP bar summary**")
        shap_bar_plot(shap_values, X)
    with c2:
        st.markdown("**Global behaviour: SHAP beeswarm**")
        shap_beeswarm_plot(shap_values, X)

    selected_food = st.selectbox("Choose a food for local explanation", results[food_col].tolist(), index=0)
    selected_idx = results.index[results[food_col] == selected_food][0]

    lime_explainer = LimeTabularExplainer(
        training_data=X.values,
        feature_names=X.columns.tolist(),
        mode="regression",
        random_state=42
    )

    lime_exp = lime_explainer.explain_instance(
        X.iloc[selected_idx].values,
        model.predict,
        num_features=len(emission_cols)
    )

    lime_df = pd.DataFrame(lime_exp.as_list(), columns=["Feature_Rule", "Contribution"])
    lime_df["Abs_Contribution"] = lime_df["Contribution"].abs()
    lime_df = lime_df.sort_values("Abs_Contribution", ascending=True)

    c3, c4 = st.columns(2)
    with c3:
        fig, ax = plt.subplots(figsize=(9, 6))
        ax.barh(lime_df["Feature_Rule"], lime_df["Contribution"])
        ax.set_title(f"LIME local explanation: {selected_food}")
        ax.set_xlabel("Contribution to prediction")
        st.pyplot(fig, use_container_width=True)

    with c4:
        local_shap = pd.DataFrame({
            "Feature": X.columns,
            "SHAP_Value": shap_values[selected_idx]
        }).sort_values("SHAP_Value", key=np.abs, ascending=True)

        fig, ax = plt.subplots(figsize=(9, 6))
        ax.barh(local_shap["Feature"], local_shap["SHAP_Value"])
        ax.set_title(f"Local SHAP values: {selected_food}")
        ax.set_xlabel("SHAP value")
        st.pyplot(fig, use_container_width=True)

    st.markdown("**LIME explanation table**")
    st.dataframe(lime_df.drop(columns="Abs_Contribution"), use_container_width=True)

# -------------------------------
# DOWNLOAD
# -------------------------------
st.markdown("---")
csv_data = results.to_csv(index=False).encode("utf-8")
st.download_button(
    label="Download scenario results as CSV",
    data=csv_data,
    file_name="food_sustainability_scenario_results.csv",
    mime="text/csv"
)
