import streamlit as st
import pandas as pd
import requests
from bs4 import BeautifulSoup
import matplotlib.pyplot as plt

# ---------------------------------------------------------
# TEAM LOGOS (Wikipedia SVG/PNG links)
# ---------------------------------------------------------
TEAM_LOGOS = {
    "Arsenal": "https://upload.wikimedia.org/wikipedia/en/5/53/Arsenal_FC.svg",
    "Manchester City": "https://upload.wikimedia.org/wikipedia/en/e/eb/Manchester_City_FC_badge.svg",
    "Manchester United": "https://upload.wikimedia.org/wikipedia/en/7/7a/Manchester_United_FC_crest.svg",
    "Liverpool": "https://upload.wikimedia.org/wikipedia/en/0/0c/Liverpool_FC.svg",
    "Chelsea": "https://upload.wikimedia.org/wikipedia/en/c/cc/Chelsea_FC.svg",
    "Tottenham Hotspur": "https://upload.wikimedia.org/wikipedia/en/b/b4/Tottenham_Hotspur.svg",
    "Aston Villa": "https://upload.wikimedia.org/wikipedia/en/f/f9/Aston_Villa_FC_crest_%282016%29.svg",
    "Newcastle United": "https://upload.wikimedia.org/wikipedia/en/5/56/Newcastle_United_Logo.svg",
    "Brighton & Hove Albion": "https://upload.wikimedia.org/wikipedia/en/f/fd/Brighton_%26_Hove_Albion_logo.svg",
    "Brentford": "https://upload.wikimedia.org/wikipedia/en/2/2a/Brentford_FC_crest.svg",
    "Fulham": "https://upload.wikimedia.org/wikipedia/en/e/eb/Fulham_FC_%28shield%29.svg",
    "Crystal Palace": "https://upload.wikimedia.org/wikipedia/en/0/0c/Crystal_Palace_FC_logo.svg",
    "Everton": "https://upload.wikimedia.org/wikipedia/en/7/7c/Everton_FC_logo.svg",
    "Leeds United": "https://upload.wikimedia.org/wikipedia/en/4/4c/Leeds_United_F.C._logo.svg",
    "West Ham United": "https://upload.wikimedia.org/wikipedia/en/c/c2/West_Ham_United_FC_logo.svg",
    "Nottingham Forest": "https://upload.wikimedia.org/wikipedia/en/7/79/Nottingham_Forest_logo.svg",
    "AFC Bournemouth": "https://upload.wikimedia.org/wikipedia/en/e/e5/AFC_Bournemouth_%282013%29.svg",
    "Wolverhampton Wanderers": "https://upload.wikimedia.org/wikipedia/en/f/fc/Wolverhampton_Wanderers.svg",
    "Burnley": "https://upload.wikimedia.org/wikipedia/en/6/62/Burnley_FC_badge.png",
    "Sunderland": "https://upload.wikimedia.org/wikipedia/en/7/77/Sunderland_A.F.C._logo.svg"
}

# ---------------------------------------------------------
# LOAD PREDICTIONS
# ---------------------------------------------------------
def load_predictions(file) -> pd.DataFrame:
    df = pd.read_excel(file)
    df.columns = df.columns.str.strip()
    return df

# ---------------------------------------------------------
# SCRAPE EPL STANDINGS
# ---------------------------------------------------------
def scrape_actual_standings() -> pd.DataFrame:
    url = "https://www.espn.com/soccer/standings/_/league/eng.1"

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0.0.0 Safari/537.36"
        ),
        "Accept-Language": "en-US,en;q=0.9",
    }

    response = requests.get(url, headers=headers, timeout=10)

    # If ESPN blocks us, return empty DataFrame instead of crashing
    if response.status_code != 200:
        return pd.DataFrame({"Team": [], "ActualStanding": []})

    soup = BeautifulSoup(response.text, "html.parser")

    teams = []

    # ESPN's current structure
    for team_cell in soup.select("div.team-link a"):
        name = team_cell.text.strip()
        if name:
            teams.append(name)

    # If scraping fails, return empty DF
    if not teams:
        return pd.DataFrame({"Team": [], "ActualStanding": []})

    return pd.DataFrame({
        "Team": teams,
        "ActualStanding": list(range(1, len(teams) + 1))
    })



# ---------------------------------------------------------
# NORMALIZE TEAM NAMES
# ---------------------------------------------------------
def normalize_team_names(df: pd.DataFrame) -> pd.DataFrame:
    mapping = {
        "Man City": "Manchester City",
        "Man United": "Manchester United",
        "Spurs": "Tottenham Hotspur",
        "Wolves": "Wolverhampton Wanderers",
        "Newcastle": "Newcastle United",
        "West Ham": "West Ham United",
        "Brighton": "Brighton & Hove Albion",
        "Forest": "Nottingham Forest",
        "Bournemouth": "AFC Bournemouth",
        "Leeds": "Leeds United",
        "Everton FC": "Everton",
        "Chelsea FC": "Chelsea",
        "Arsenal FC": "Arsenal",
        "Liverpool FC": "Liverpool",
    }
    df["Team"] = df["Team"].replace(mapping)
    return df

# ---------------------------------------------------------
# COMPARE PREDICTIONS
# ---------------------------------------------------------
def compare_predictions(pred_df: pd.DataFrame, actual_df: pd.DataFrame) -> pd.DataFrame:
    # Merge predictions with actual standings
    merged = pred_df.merge(actual_df, on="Team", how="left")

    # Identify predictor columns (everything except Team + ActualStanding)
    predictors = [c for c in merged.columns if c not in ["Team", "ActualStanding"]]

    # Convert predictor columns to numeric (fixes PyArrow type errors)
    for p in predictors:
        merged[p] = pd.to_numeric(merged[p], errors="coerce")

    # Compute absolute errors safely
    for p in predictors:
        merged[f"{p}_Error"] = (merged[p] - merged["ActualStanding"]).abs()

    return merged

# ---------------------------------------------------------
# RANK PREDICTORS
# ---------------------------------------------------------
def rank_predictors(result_df: pd.DataFrame) -> pd.DataFrame:
    error_cols = [c for c in result_df.columns if c.endswith("_Error")]
    ranking = result_df[error_cols].sum().sort_values().reset_index()
    ranking.columns = ["Predictor", "TotalError"]
    ranking["Predictor"] = ranking["Predictor"].str.replace("_Error", "")
    return ranking

# ---------------------------------------------------------
# PLOT ACCURACY
# ---------------------------------------------------------
def plot_prediction_accuracy(ranking_df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(ranking_df["Predictor"], ranking_df["TotalError"], color="royalblue")
    ax.set_title("Prediction Accuracy (Lower = Better)")
    ax.set_xlabel("Predictor")
    ax.set_ylabel("Total Error")

    for i, val in enumerate(ranking_df["TotalError"]):
        ax.text(i, val + 0.2, str(val), ha="center")

    return fig

# ---------------------------------------------------------
# STREAMLIT UI
# ---------------------------------------------------------
st.set_page_config(page_title="EPL Prediction Dashboard", layout="wide")
st.title("⚽ EPL 2025–2026 Prediction Accuracy Dashboard")

uploaded_file = st.file_uploader("Upload predictions Excel (.xlsx)", type=["xlsx"])

if uploaded_file:
    pred_df = load_predictions(uploaded_file)
    pred_df = normalize_team_names(pred_df)
    pred_df["Logo"] = pred_df["Team"].map(TEAM_LOGOS)

    st.subheader("📄 Uploaded Predictions")
    st.dataframe(
        pred_df[["Logo", "Team"] + [c for c in pred_df.columns if c not in ["Logo", "Team"]]],
        column_config={"Logo": st.column_config.ImageColumn("Logo", width="small")},
        use_container_width=True
    )

    actual_df = scrape_actual_standings()
    actual_df = normalize_team_names(actual_df)
    actual_df["Logo"] = actual_df["Team"].map(TEAM_LOGOS)

    st.subheader("📊 Actual EPL Standings (Scraped)")
    st.dataframe(
        actual_df[["Logo", "Team", "ActualStanding"]],
        column_config={"Logo": st.column_config.ImageColumn("Logo", width="small")},
        use_container_width=True
    )

    result_df = compare_predictions(pred_df, actual_df)

    st.subheader("🔍 Prediction vs Actual")
    st.dataframe(result_df, use_container_width=True)

    ranking_df = rank_predictors(result_df)

    st.subheader("🏆 Predictor Ranking")
    st.dataframe(ranking_df, use_container_width=True)

    st.subheader("📉 Total Error by Predictor")
    fig = plot_prediction_accuracy(ranking_df)
    st.pyplot(fig)

else:
    st.info("Upload a predictions Excel file to begin.")
