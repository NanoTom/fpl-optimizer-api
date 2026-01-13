from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import solver
import difflib

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class UserTeamInput(BaseModel):
    player_names: List[str]
    bank: float


# --- HELPER: Match Names to Real Data ---
def match_names_to_data(names, df):
    name_map = {}
    for _, row in df.iterrows():
        pid = row['id']
        web_name = row['web_name'].lower()
        name_map[web_name] = pid
        if " " in web_name:
            name_map[web_name.split(' ')[-1]] = pid

    matched_ids = []
    found_names = set()

    for scraped_name in names:
        clean_name = scraped_name.lower().strip().split('£')[0].strip()
        if len(clean_name) < 3: continue

        matched_id = None
        if clean_name in name_map:
            matched_id = name_map[clean_name]
        else:
            matches = difflib.get_close_matches(clean_name, name_map.keys(), n=1, cutoff=0.7)
            if matches:
                matched_id = name_map[matches[0]]

        if matched_id and matched_id not in matched_ids:
            matched_ids.append(matched_id)

    return matched_ids


# --- ENDPOINT 1: Best Current Lineup ---
@app.post("/optimize-lineup")
def current_lineup(data: UserTeamInput):
    print(f"📥 LINEUP REQUEST: {len(data.player_names)} players")

    df = solver.get_live_data()
    if df is None: raise HTTPException(500, "FPL API Error")

    current_ids = match_names_to_data(data.player_names, df)

    if len(current_ids) < 11:
        raise HTTPException(400, f"Only identified {len(current_ids)} players.")

    # Get full data for these 15 players
    current_squad_list = df[df['id'].isin(current_ids)].to_dict('records')

    # Calculate best lineup from THIS exact list
    tactics = solver.get_best_lineup(current_squad_list)

    return {"tactics": tactics}


# --- ENDPOINT 2: Transfers ---
@app.post("/recommend-transfers")
def transfers(data: UserTeamInput):
    print(f"📥 TRANSFER REQUEST: {len(data.player_names)} players")

    df = solver.get_live_data()
    if df is None: raise HTTPException(500, "FPL API Error")

    current_ids = match_names_to_data(data.player_names, df)

    if len(current_ids) < 11:
        raise HTTPException(400, f"Only identified {len(current_ids)} players.")

    # 1. Run Solver (Market Search)
    optimal_team_list = solver.solve_squad(df)

    # 2. Get Transfers
    suggestions = solver.calculate_transfers(df, current_ids, optimal_team_list, data.bank)

    # 3. Get Lineup for the NEW hypothetical team
    # (We combine kept players + buy players to show the future lineup)
    tactics = solver.get_best_lineup(optimal_team_list)

    return {
        "transfers": suggestions,
        "tactics": tactics
    }