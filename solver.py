import pandas as pd
import pulp
import requests
import warnings
import numpy as np

warnings.filterwarnings('ignore')

# ==========================================
# --- CONFIGURATION ---
# ==========================================
API_URL = "https://fantasy.premierleague.com/api/bootstrap-static/"
FIXTURES_URL = "https://fantasy.premierleague.com/api/fixtures/"

# Strategy Constraints
BUDGET = 1000  # £100.0m
MIN_TRANSFER_GAIN = 4.0
W_UPCOMING = {'ep_next': 0.60, 'form': 0.30, 'fixture': 0.10}
W_SEASON = {'total_points': 0.50, 'ict_index': 0.30, 'minutes': 0.20}


# ==========================================
# --- 1. DATA ENGINE ---
# ==========================================
def get_live_data():
    """
    Fetches data from FPL API, calculates custom scores, and returns a DataFrame.
    Returns None on failure.
    """
    try:
        # 1. Fetch Data
        static = requests.get(API_URL).json()
        fixtures = requests.get(FIXTURES_URL).json()

        teams = pd.DataFrame(static['teams'])
        teams_map = teams.set_index('id')['name'].to_dict()

        events = pd.DataFrame(static['events'])
        next_gw_row = events[events['is_next'] == True]

        gw_id = 38  # Default fallback
        if not next_gw_row.empty:
            gw_id = next_gw_row.iloc[0]['id']

        # 2. Calculate Fixture Difficulty (FDR)
        fix_df = pd.DataFrame(fixtures)
        current_fix = fix_df[fix_df['event'] == gw_id]

        team_fdr = {}
        for _, row in current_fix.iterrows():
            # Invert difficulty: 1 (hard) to 5 (easy)
            team_fdr[row['team_h']] = 6 - row['team_h_difficulty']
            team_fdr[row['team_a']] = 6 - row['team_a_difficulty']

        # 3. Process Players
        df = pd.DataFrame(static['elements'])
        cols = ['id', 'web_name', 'element_type', 'team', 'now_cost',
                'total_points', 'form', 'ep_next', 'ict_index',
                'minutes', 'chance_of_playing_next_round']
        df = df[cols].copy()

        df['team_name'] = df['team'].map(teams_map)
        df['position'] = df['element_type'].map({1: 'GK', 2: 'DEF', 3: 'MID', 4: 'FWD'})
        df['fixture_ease'] = df['team'].map(team_fdr).fillna(2.0)

        # Clean numeric columns
        for c in ['form', 'ep_next', 'ict_index', 'total_points', 'minutes']:
            df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0)

        df['chance_of_playing_next_round'] = df['chance_of_playing_next_round'].fillna(100)

        # 4. Calculate Scores
        def norm(s):
            if s.max() == s.min(): return 0
            return (s - s.min()) / (s.max() - s.min())

        df['score_upcoming'] = (
                                       (norm(df['ep_next']) * W_UPCOMING['ep_next']) +
                                       (norm(df['form']) * W_UPCOMING['form']) +
                                       (norm(df['fixture_ease']) * W_UPCOMING['fixture'])
                               ) * 100

        df['score_season'] = (
                                     (norm(df['total_points']) * W_SEASON['total_points']) +
                                     (norm(df['ict_index']) * W_SEASON['ict_index']) +
                                     (norm(df['minutes']) * W_SEASON['minutes'])
                             ) * 100

        # Positional Bias (Forwards tend to be undervalued by raw stats)
        df.loc[df['position'] == 'FWD', 'score_upcoming'] *= 1.10
        df.loc[df['position'] == 'FWD', 'score_season'] *= 1.05

        return df

    except Exception as e:
        print(f"Server Error in get_live_data: {e}")
        return None


# ==========================================
# --- 2. OPTIMIZER ---
# ==========================================
def solve_squad(df, score_col='score_upcoming', budget=BUDGET):
    """
    Solves the Knapsack problem to find the best possible 15-man squad.
    Returns: List of Dictionaries (The optimal squad players).
    """
    # Filter for available players
    pool = df[df['chance_of_playing_next_round'] >= 75].copy()

    prob = pulp.LpProblem("FPL_Solver", pulp.LpMaximize)
    players = pool['id'].tolist()

    score = dict(zip(players, pool[score_col]))
    cost = dict(zip(players, pool['now_cost']))
    pos = dict(zip(players, pool['position']))
    team_names = dict(zip(players, pool['team_name']))

    # Binary Variable: 1 if player is selected, 0 otherwise
    x = pulp.LpVariable.dicts("p", players, 0, 1, pulp.LpBinary)

    # Objective: Maximize Score
    prob += pulp.lpSum([score[i] * x[i] for i in players])

    # Constraint 1: Budget
    prob += pulp.lpSum([cost[i] * x[i] for i in players]) <= budget

    # Constraint 2: Squad Size (15 players)
    prob += pulp.lpSum([x[i] for i in players]) == 15

    # Constraint 3: Position Limits
    prob += pulp.lpSum([x[i] for i in players if pos[i] == 'GK']) == 2
    prob += pulp.lpSum([x[i] for i in players if pos[i] == 'DEF']) == 5
    prob += pulp.lpSum([x[i] for i in players if pos[i] == 'MID']) == 5
    prob += pulp.lpSum([x[i] for i in players if pos[i] == 'FWD']) == 3

    # Constraint 4: Max 3 players per team
    unique_teams = pool['team_name'].unique()
    for t in unique_teams:
        t_ids = [p_id for p_id in players if team_names[p_id] == t]
        prob += pulp.lpSum([x[i] for i in t_ids]) <= 3

    # Solve silently
    prob.solve(pulp.PULP_CBC_CMD(msg=0))

    # Extract Results
    selected_ids = [i for i in players if x[i].varValue == 1]
    result_df = df[df['id'].isin(selected_ids)].copy()

    result_df['p_sort'] = result_df['position'].map({'GK': 0, 'DEF': 1, 'MID': 2, 'FWD': 3})
    result_df = result_df.sort_values(['p_sort', score_col], ascending=[True, False])

    return result_df.to_dict(orient='records')


# ==========================================
# --- 3. TRANSFER LOGIC ---
# ==========================================
def calculate_transfers(df, current_team_ids, optimal_team_list, bank):
    """
    Compares user's team vs optimal team and suggests transfers.
    """
    optimal_ids = set([p['id'] for p in optimal_team_list])
    current_ids = set(current_team_ids)

    to_sell_ids = list(current_ids - optimal_ids)
    to_buy_ids = list(optimal_ids - current_ids)

    if not to_sell_ids:
        return []

    sell_players = df[df['id'].isin(to_sell_ids)]
    buy_players = df[df['id'].isin(to_buy_ids)]

    suggestions = []

    for _, sell in sell_players.iterrows():
        for _, buy in buy_players.iterrows():
            if sell['position'] == buy['position']:
                if buy['now_cost'] <= (sell['now_cost'] + bank):
                    gain = buy['score_upcoming'] - sell['score_upcoming']

                    if gain >= MIN_TRANSFER_GAIN:
                        suggestions.append({
                            'sell_id': int(sell['id']),
                            'sell_name': sell['web_name'],
                            'sell_cost': float(sell['now_cost']) / 10,
                            'buy_id': int(buy['id']),
                            'buy_name': buy['web_name'],
                            'buy_cost': float(buy['now_cost']) / 10,
                            'gain': round(gain, 2)
                        })

    suggestions.sort(key=lambda x: x['gain'], reverse=True)
    return suggestions[:5]


# ==========================================
# --- 4. LINEUP LOGIC (NEW) ---
# ==========================================
def get_best_lineup(squad_list):
    """
    Takes a list of 15 players and returns the optimal Starting XI + Bench.
    """
    df = pd.DataFrame(squad_list)
    df = df.sort_values('score_upcoming', ascending=False)

    captain_id = df.iloc[0]['id']
    vice_id = df.iloc[1]['id']

    gks = df[df['position'] == 'GK'].sort_values('score_upcoming', ascending=False)
    defs = df[df['position'] == 'DEF'].sort_values('score_upcoming', ascending=False)
    mids = df[df['position'] == 'MID'].sort_values('score_upcoming', ascending=False)
    fwds = df[df['position'] == 'FWD'].sort_values('score_upcoming', ascending=False)

    formations = [(3, 4, 3), (3, 5, 2), (4, 3, 3), (4, 4, 2), (4, 5, 1), (5, 3, 2), (5, 4, 1)]
    best_xi_score = -1
    final_starters = []

    for n_d, n_m, n_f in formations:
        current_xi = pd.concat([
            gks.head(1),
            defs.head(n_d),
            mids.head(n_m),
            fwds.head(n_f)
        ])

        score = current_xi['score_upcoming'].sum()
        if score > best_xi_score:
            best_xi_score = score
            final_starters = current_xi.to_dict('records')

    starter_ids = [p['id'] for p in final_starters]
    final_bench = df[~df['id'].isin(starter_ids)].to_dict('records')

    return {
        "starters": final_starters,
        "bench": final_bench,
        "captain_id": int(captain_id),
        "vice_id": int(vice_id)
    }