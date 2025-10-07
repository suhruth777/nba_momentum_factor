from scripts.momentum_utils import fetch_process_by_game_id, feature_pack

def test_single_game_smoke():
    df = fetch_process_by_game_id("0022300001")
    assert {"home_score","away_score","score_diff","momentum_index"}.issubset(df.columns)
    feats = feature_pack(df)
    assert "momentum_l2_mean" in feats
    print("OK")

if __name__ == "__main__":
    test_single_game_smoke()