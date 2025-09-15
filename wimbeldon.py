"""
Wimbledon 2025 final-qualification probabilities using tennis_atp-master (2019–2024).

Excludes retired/non-participating players. Only matches from 2019-01-01 to 2024-12-31 are considered.

Requirements:
  pip install pandas numpy scikit-learn catboost tqdm
"""

import pandas as pd
import numpy as np
import glob
import random
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss
from tqdm import tqdm

try:
    from catboost import CatBoostClassifier, CatBoostError
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False

# List of known retired or non‐participating players to exclude
RETIRED_PLAYERS = {
    "Roger Federer",
    "Rafael Nadal",
    "Andy Roddick",
    "Mardy Fish",
    "Cezar Cretu",
    # add others as needed
}

########################################
# LOAD MATCHES (ONLY 2019–2024)
########################################
def load_matches(folder: str = "tennis_atp-master") -> pd.DataFrame:
    files = glob.glob(f"{folder}/atp_matches_*.csv")
    df_list = [pd.read_csv(f, low_memory=False) for f in files]
    df = pd.concat(df_list, ignore_index=True)
    df = df.rename(columns={
        "tourney_date": "Date",
        "surface": "Surface",
        "winner_name": "PlayerA",
        "loser_name": "PlayerB",
        "w_ace": "AcesA",
        "l_ace": "AcesB",
        "w_df": "DoubleFaultsA",
        "l_df": "DoubleFaultsB",
        "winner_rank": "RankA",
        "loser_rank": "RankB",
        "winner_rank_points": "RankPointsA",
        "loser_rank_points": "RankPointsB"
    })
    df["Date"] = pd.to_datetime(df["Date"], format="%Y%m%d", errors="coerce")
    # Restrict to 2019–2024 data only
    df = df[(df["Date"] >= "2019-01-01") & (df["Date"] <= "2024-12-31")]
    # Exclude any matches involving retired players
    df = df[~df["PlayerA"].isin(RETIRED_PLAYERS) & ~df["PlayerB"].isin(RETIRED_PLAYERS)]
    # Optionally print the first/last dates to confirm filter
    print("Using match data from", df["Date"].min().date(), "to", df["Date"].max().date())
    return df.dropna(subset=["PlayerA","PlayerB","Surface"]).reset_index(drop=True)

########################################
# COMPUTE ELO
########################################
def compute_elo(matches: pd.DataFrame, k=10) -> dict:
    elo = {}
    for _, r in matches.sort_values("Date").iterrows():
        a, b = r["PlayerA"], r["PlayerB"]
        elo.setdefault(a, 1500)
        elo.setdefault(b, 1500)
        ea = 1/(1 + 10**((elo[b]-elo[a])/400))
        sa, sb = 1, 0
        elo[a] += k*(sa-ea)
        elo[b] += k*(sb-(1-ea))
    return elo

########################################
# BUILD FEATURES
########################################
def build_features(matches: pd.DataFrame, elo: dict) -> pd.DataFrame:
    players = pd.unique(matches[["PlayerA","PlayerB"]].values.ravel())
    rows = []
    for p in players:
        if p in RETIRED_PLAYERS:
            continue
        m = matches[(matches["PlayerA"]==p)|(matches["PlayerB"]==p)]
        total = len(m)
        wins = (m["PlayerA"]==p).sum()
        grass = m[m["Surface"].str.lower().str.contains("grass",na=False)]
        grass_pct = grass["PlayerA"].eq(p).sum()/len(grass) if len(grass)>0 else 0.5

        aces = m.apply(lambda r: r["AcesA"] if r["PlayerA"]==p else r["AcesB"], axis=1)
        dfaults = m.apply(lambda r: r["DoubleFaultsA"] if r["PlayerA"]==p else r["DoubleFaultsB"], axis=1)
        ranks = m.apply(lambda r: r.get("RankA",np.nan) if r["PlayerA"]==p else r.get("RankB",np.nan), axis=1)
        rpts  = m.apply(lambda r: r.get("RankPointsA",np.nan) if r["PlayerA"]==p else r.get("RankPointsB",np.nan), axis=1)

        rows.append({
            "Player": p,
            "Elo": elo.get(p,1500),
            "WinPct": wins/total if total>0 else 0.5,
            "GrassPct": grass_pct,
            "Aces": aces.replace([np.inf,-np.inf],np.nan).dropna().mean() or 0.0,
            "DF": dfaults.replace([np.inf,-np.inf],np.nan).dropna().mean() or 0.0,
            "AvgRank": ranks.replace([np.inf,-np.inf],np.nan).dropna().mean() or 999.0,
            "AvgRankPts": rpts.replace([np.inf,-np.inf],np.nan).dropna().mean() or 0.0
        })
    return pd.DataFrame(rows)

########################################
# TRAIN MODEL (Balanced)
########################################
def train_match_model(matches: pd.DataFrame, feat: pd.DataFrame):
    X_rows, y = [], []
    for _, r in matches.iterrows():
        pa, pb = r["PlayerA"], r["PlayerB"]
        if pa in RETIRED_PLAYERS or pb in RETIRED_PLAYERS:
            continue
        if pa not in feat["Player"].values or pb not in feat["Player"].values:
            continue
        fa = feat[feat["Player"]==pa].iloc[0]
        fb = feat[feat["Player"]==pb].iloc[0]
        def diffs(win, lose):
            return {
                "EloDiff": win["Elo"]-lose["Elo"],
                "GrassDiff": win["GrassPct"]-lose["GrassPct"],
                "WinDiff": win["WinPct"]-lose["WinPct"],
                "AcesDiff": win["Aces"]-lose["Aces"],
                "DFDiff": win["DF"]-lose["DF"],
                "RankDiff": lose["AvgRank"]-win["AvgRank"],
                "RankPtsDiff": win["AvgRankPts"]-lose["AvgRankPts"]
            }
        X_rows.append(diffs(fa, fb)); y.append(1)
        X_rows.append(diffs(fb, fa)); y.append(0)

    X = pd.DataFrame(X_rows).fillna(0.0)
    y = np.array(y)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    if CATBOOST_AVAILABLE:
        try:
            model = CatBoostClassifier(iterations=500, learning_rate=0.05, depth=6, verbose=False)
            model.fit(Xtr, ytr, eval_set=(Xte,yte), early_stopping_rounds=50)
        except CatBoostError:
            model = RandomForestClassifier(n_estimators=300, max_depth=12, random_state=42)
            model.fit(Xtr, ytr)
    else:
        model = RandomForestClassifier(n_estimators=300, max_depth=12, random_state=42)
        model.fit(Xtr, ytr)

    preds = model.predict_proba(Xte)[:,1]
    print("Log loss:", round(log_loss(yte,preds),4))
    return model, list(X.columns)

########################################
# MATCH WIN PROBABILITY
########################################
def predict_match_prob(model, cols, pa, pb):
    vec = {
        "EloDiff": pa["Elo"]-pb["Elo"],
        "GrassDiff": pa["GrassPct"]-pb["GrassPct"],
        "WinDiff": pa["WinPct"]-pb["WinPct"],
        "AcesDiff": pa["Aces"]-pb["Aces"],
        "DFDiff": pa["DF"]-pb["DF"],
        "RankDiff": pb["AvgRank"]-pa["AvgRank"],
        "RankPtsDiff": pa["AvgRankPts"]-pb["AvgRankPts"]
    }
    row = pd.DataFrame([vec])[cols].fillna(0.0)
    return model.predict_proba(row)[0,1]

########################################
# MONTE CARLO FINAL QUALIFICATION (Top 10 with normalization)
########################################
def monte_carlo(draw, feat_df, model, cols, sims=1000):
    if len(draw) == 0:
        return {}
    
    # Pad draw to power of 2 for tournament bracket
    while len(draw) & (len(draw) - 1) != 0:
        draw.append(draw[0])
    
    reach = {p: 0 for p in set(draw)}

    for _ in tqdm(range(sims), desc="Simulating"):
        bracket = [p for p in draw if p not in RETIRED_PLAYERS]
        random.shuffle(bracket)
        
        # Tournament simulation
        while len(bracket) > 1:
            next_bracket = []
            for i in range(0, len(bracket), 2):
                if i + 1 >= len(bracket):
                    next_bracket.append(bracket[i])
                    break
                pa, pb = bracket[i], bracket[i+1]
                try:
                    fa = feat_df[feat_df["Player"]==pa].iloc[0]
                    fb = feat_df[feat_df["Player"]==pb].iloc[0]
                    pA = predict_match_prob(model, cols, fa, fb)
                    winner = pa if random.random() < pA else pb
                except:
                    winner = random.choice([pa, pb])
                next_bracket.append(winner)
            bracket = next_bracket
        
        # Winner reaches final
        if bracket:
            reach[bracket[0]] += 1

    # Normalize probabilities to sum to 1 for top 10 players
    total = sum(reach.values())
    if total > 0:
        return {p: reach[p]/total for p in reach}
    else:
        return {p: 0 for p in reach}

########################################
# LOAD DRAW (Top 10 from features, excluding Cezar Cretu)
########################################
def load_draw(feats, top_n=10):
    """Load top 10 players from features, excluding retired players and Cezar Cretu"""
    try:
        # Filter out retired players including Cezar Cretu
        active_players = feats[~feats["Player"].isin(RETIRED_PLAYERS)]
        active_players = active_players[active_players["Player"] != "Cezar Cretu"]
        
        # Sort by Elo and take top N
        top_players = active_players.nlargest(top_n, "Elo")
        return top_players["Player"].tolist()
    except:
        return []

########################################
# MAIN
########################################
if __name__ == "__main__":
    print("Loading matches...")
    matches = load_matches()

    print("Computing Elo...")
    elo = compute_elo(matches)

    print("Building features...")
    feats = build_features(matches, elo)
    print(f"{len(feats)} active players loaded.")

    print("Training model...")
    model, cols = train_match_model(matches, feats)

    print("Loading draw (top 10 active players)...")
    draw = load_draw(feats, top_n=10)
    print(f"Draw size: {len(draw)}")
    print(f"Top 10 players: {draw}")

    if len(draw) > 0:
        print("Simulating probability of reaching FINAL...")
        results = monte_carlo(draw, feats, model, cols, sims=1000)

        print("\n--- Top 10 Chances to Reach FINAL ---")
        total_prob = 0
        for p, prob in sorted(results.items(), key=lambda x: x[1], reverse=True):
            print(f"{p:25s} : {prob*100:.2f}%")
            total_prob += prob
        print(f"\nTotal probability check: {total_prob*100:.2f}%")
    else:
        print("No active players found for simulation!")
