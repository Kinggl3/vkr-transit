"""
Расписания событий на площадках рядом со станциями метро.

Источники (все известны заранее, поэтому честно используются в прогнозе):
  * MLB Stats API (statsapi.mlb.com): домашние матчи Yankees (Yankee Stadium) и Mets (Citi Field),
    а также Brooklyn Cyclones (младшая лига, Maimonides Park у Coney Island);
  * ESPN API: домашние матчи Brooklyn Nets (NBA) и New York Liberty (WNBA) в Barclays Center;
  * US Open (теннис, у станции Mets-Willets Point): даты задаются вручную по официальному сайту.

Берутся только состоявшиеся матчи: перенесённые и отменённые не попадают.

Запуск (из корня проекта):
  python src/download_events.py
Результат: data/raw/events.csv — одна строка на событие:
  kind, venue, start (местное время Нью-Йорка), source_id
"""

import time

import pandas as pd
import requests

from config import RAW

SEASONS = [2023, 2024, 2025, 2026]

# kind, sportId, teamId, venueId (домашняя площадка)
MLB_TEAMS = [
    ("mlb_yankees", 1, 147, 3313),   # Yankee Stadium
    ("mlb_mets", 1, 121, 3289),      # Citi Field
    ("milb_cyclones", 13, 453, 2795),  # Maimonides Park, Coney Island
]
MLB_GAME_TYPES = "R,F,D,L,W"         # регулярный сезон + плей-офф

# kind, лига ESPN, teamId, сезоны ESPN (NBA-сезон 2024 = 2023/24)
ESPN_TEAMS = [
    ("nba_nets", "nba", 17, [2023, 2024, 2025, 2026, 2027]),
    ("wnba_liberty", "wnba", 9, [2023, 2024, 2025, 2026]),
]

# US Open: (неделя болельщиков с квалификацией), (основная сетка). usopen.org; 2023–2025 сверены
# со всплесками входов на Mets-Willets Point в данных MTA.
US_OPEN = {
    2023: (("2023-08-22", "2023-08-27"), ("2023-08-28", "2023-09-10")),
    2024: (("2024-08-19", "2024-08-25"), ("2024-08-26", "2024-09-08")),
    2025: (("2025-08-18", "2025-08-23"), ("2025-08-24", "2025-09-07")),
    2026: (("2026-08-23", "2026-08-29"), ("2026-08-30", "2026-09-13")),
}


def get_json(url, params=None):
    for attempt in range(1, 6):
        try:
            r = requests.get(url, params=params, timeout=60)
            r.raise_for_status()
            return r.json()
        except requests.RequestException as e:
            print(f"  попытка {attempt}/5: {e}; жду {5 * attempt} с", flush=True)
            time.sleep(5 * attempt)
    raise RuntimeError(f"Не удалось получить {url}")


def to_local(utc_str: str) -> pd.Timestamp:
    return pd.Timestamp(utc_str).tz_convert("America/New_York").tz_localize(None)


def mlb_events() -> list:
    rows = []
    for kind, sport, team, venue in MLB_TEAMS:
        for season in SEASONS:
            data = get_json("https://statsapi.mlb.com/api/v1/schedule",
                            {"sportId": sport, "teamId": team, "season": season,
                             "gameType": MLB_GAME_TYPES})
            n = 0
            for d in data.get("dates", []):
                for g in d.get("games", []):
                    if g.get("venue", {}).get("id") != venue:
                        continue                      # выездной матч
                    state = g.get("status", {}).get("detailedState", "")
                    if not (state.startswith("Final") or state in ("Game Over", "Completed Early")):
                        continue                      # перенесён, отменён или ещё не сыгран
                    rows.append({"kind": kind, "venue": g["venue"]["name"],
                                 "start": to_local(g["gameDate"]), "source_id": f"mlb:{g['gamePk']}"})
                    n += 1
            print(f"{kind} {season}: {n} домашних матчей", flush=True)
            time.sleep(0.5)
    return rows


def espn_events() -> list:
    rows = []
    for kind, league, team, seasons in ESPN_TEAMS:
        for season in seasons:
            n = 0
            for stype in (2, 3):                      # регулярный сезон, плей-офф
                url = f"https://site.api.espn.com/apis/site/v2/sports/basketball/{league}/teams/{team}/schedule"
                data = get_json(url, {"season": season, "seasontype": stype})
                for ev in data.get("events", []):
                    comp = ev.get("competitions", [{}])[0]
                    venue = comp.get("venue", {}).get("fullName", "")
                    status = comp.get("status") or ev.get("status") or {}   # у ESPN бывает в обоих местах
                    done = status.get("type", {}).get("completed", False)
                    if "Barclays" not in venue or not done:
                        continue
                    rows.append({"kind": kind, "venue": venue, "start": to_local(ev["date"]),
                                 "source_id": f"espn:{ev['id']}"})
                    n += 1
                time.sleep(0.5)
            print(f"{kind} сезон {season}: {n} домашних матчей в Barclays Center", flush=True)
    return rows


def us_open_events() -> list:
    rows = []
    for year, ((fw_a, fw_b), (md_a, md_b)) in US_OPEN.items():
        for d in pd.date_range(fw_a, fw_b):
            rows.append({"kind": "usopen_fanweek", "venue": "USTA Billie Jean King NTC",
                         "start": d + pd.Timedelta(hours=11), "source_id": f"usopen:{d:%Y%m%d}"})
        for d in pd.date_range(md_a, md_b):
            for h in (11, 19):                        # дневная и вечерняя сессии
                rows.append({"kind": "usopen_main", "venue": "USTA Billie Jean King NTC",
                             "start": d + pd.Timedelta(hours=h), "source_id": f"usopen:{d:%Y%m%d}:{h}"})
    print(f"US Open: {len(rows)} сессий")
    return rows


def main():
    rows = mlb_events() + espn_events() + us_open_events()
    ev = pd.DataFrame(rows).drop_duplicates("source_id").sort_values("start")
    out = RAW / "events.csv"
    ev.to_csv(out, index=False)
    print(f"\nВсего событий: {len(ev)}")
    print(ev.groupby("kind").agg(событий=("start", "size"), с=("start", "min"), по=("start", "max")))
    print(f"Сохранено: {out}")


if __name__ == "__main__":
    main()
