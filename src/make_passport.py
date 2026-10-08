"""
Паспорт данных: описательная статистика собранного датасета в одном markdown-файле.

Запуск (из корня проекта, после build_dataset.py и build_features.py):
  python src/make_passport.py
Результат: reports/data_passport.md (коммитится в git — в отличие от data/).
"""

import json

import numpy as np
import pandas as pd

from config import (CONGESTION_PRICING_START, PROCESSED, RAW, REPORTS, SPLITS, STUDENT_OMNY_START,
                    split_of)


def fmt(x, nd=0):
    if pd.isna(x):
        return "—"
    return f"{x:,.{nd}f}".replace(",", " ")


def md_table(df: pd.DataFrame) -> str:
    cols = [str(c) for c in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join("---" for _ in cols) + " |"]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(str(v) for v in r.values) + " |")
    return "\n".join(lines)


def target_stats(y: pd.Series) -> dict:
    return {"среднее": fmt(y.mean(), 1), "ст. откл.": fmt(y.std(), 1), "медиана": fmt(y.median()),
            "95-й перц.": fmt(y.quantile(0.95)), "99-й перц.": fmt(y.quantile(0.99)), "макс.": fmt(y.max()),
            "доля нулей": f"{(y == 0).mean():.1%}", "асимметрия": fmt(y.skew(), 2)}


def main():
    REPORTS.mkdir(exist_ok=True)
    panel = pd.read_parquet(PROCESSED / "panel.parquet")
    panel["split"] = split_of(panel["transit_timestamp"])
    st = pd.read_parquet(PROCESSED / "station_static.parquet")
    cal = pd.read_parquet(PROCESSED / "calendar_ext.parquet")
    closures = pd.read_csv(PROCESSED / "closures.csv")

    n_st, n_h = panel["station_complex_id"].nunique(), panel["transit_timestamp"].nunique()
    opn = panel[panel["is_closed"] == 0]
    out = []
    w = out.append

    w("# Паспорт данных: почасовой пассажиропоток метро Нью-Йорка\n")
    w(f"Сгенерировано скриптом `src/make_passport.py` {pd.Timestamp.now():%d.%m.%Y}.\n")

    w("## 1. Общие сведения\n")
    w(md_table(pd.DataFrame([
        ("Объект", "станционные комплексы метро Нью-Йорка (MTA), без Staten Island Railway"),
        ("Целевая переменная", "`ridership` — число входов в метро на станции за час (все способы оплаты и тарифы)"),
        ("Источник", "MTA Subway Hourly Ridership, data.ny.gov (`wujg-7c2s`, `5wq4-mkjj`)"),
        ("Период", f"{panel['transit_timestamp'].min():%d.%m.%Y %H:%M} – {panel['transit_timestamp'].max():%d.%m.%Y %H:%M}"),
        ("Частота", "1 час, местное время America/New_York без часового пояса"),
        ("Станций", fmt(n_st)),
        ("Часов", fmt(n_h)),
        ("Строк (станция × час)", fmt(len(panel))),
        ("Суммарно входов", fmt(panel["ridership"].sum())),
    ], columns=["Параметр", "Значение"])))
    w("\nВремя у MTA округлено вниз до часа (вход в 13:37 → 13:00). Учитываются только входы, выходов в данных нет.\n")

    w("## 2. Разбиение по времени\n")
    rows = []
    for name, (a, b) in SPLITS.items():
        p = panel[panel["split"] == name]
        po = p[p["is_closed"] == 0]
        rows.append({"выборка": name, "с": f"{a:%d.%m.%Y}", "по": f"{b:%d.%m.%Y}",
                     "часов": fmt(p["transit_timestamp"].nunique()), "строк": fmt(len(p)),
                     "доля строк": f"{len(p) / len(panel):.1%}",
                     "входов в час (среднее)": fmt(po["ridership"].mean(), 1),
                     "закрыто, % станционных дней": f"{p['is_closed'].mean():.2%}"})
    w(md_table(pd.DataFrame(rows)))
    w("\nРазбиение строго по времени, без перемешивания. Валидация используется для подбора гиперпараметров, "
      "тест — один раз для итоговых метрик.\n")

    w("## 3. Целевая переменная\n")
    w("По строкам открытых станций (`is_closed = 0`):\n")
    t = pd.DataFrame({"все часы": target_stats(opn["ridership"])})
    for name in SPLITS:
        t[name] = target_stats(opn.loc[opn["split"] == name, "ridership"])
    w(md_table(t.reset_index().rename(columns={"index": "статистика"})))
    hourly = opn.groupby(opn["transit_timestamp"].dt.hour)["ridership"].mean()
    w(f"\nСредние входы по часам суток: минимум {fmt(hourly.min(), 1)} в {hourly.idxmin()}:00, "
      f"максимум {fmt(hourly.max(), 1)} в {hourly.idxmax()}:00.\n")

    s = st["scale_weekday"].dropna().sort_values()
    top10 = s.nlargest(max(1, len(s) // 10)).sum() / s.sum()
    w("**Масштаб станций** (средние входы в будний день, обучающий период):\n")
    w(md_table(pd.DataFrame([("минимум", fmt(s.min()), st.loc[st["scale_weekday"].idxmin(), "station_complex"]),
                             ("медиана", fmt(s.median()), ""),
                             ("максимум", fmt(s.max()), st.loc[st["scale_weekday"].idxmax(), "station_complex"]),
                             ("доля 10% крупнейших станций во всех входах", f"{top10:.0%}", "")],
                            columns=["", "входов в день", "станция"])))
    w("\nМасштаб различается на три порядка: метрики считаются в относительной форме (WAPE) "
      "и отдельно по группам станций.\n")

    w("## 4. Пропуски, закрытия, особые часы\n")
    zero_hours = (panel["ridership"] == 0).mean()
    w(md_table(pd.DataFrame([
        ("Часов без единого входа (заполнены нулём)", f"{zero_hours:.1%} строк",
         "MTA не публикует часы без входов; в основном ночь на малых станциях"),
        ("Закрытые станционные дни (`is_closed`)", f"{panel['is_closed'].mean():.2%} строк",
         "поток < 10% медианы станции за тот же год и день недели"),
        ("Отрезков закрытий", fmt(len(closures)), f"у {closures['station_complex_id'].nunique()} станций"),
        ("Сдвоенный час осеннего перевода времени", fmt(cal["is_dst_double_hour"].sum()) + " ч",
         "в 01:00 сложены два часа; фактически входов в ~1,3 раза больше соседних часов; флаг `is_dst_double_hour`"),
        ("Весенний перевод времени", "час 02:00 отсутствует", "несуществующий час, не пропуск"),
    ], columns=["Что", "Объём", "Комментарий"])))
    w("\nЗакрытые дни не удаляются: при обучении их строки исключаются из целевой переменной, "
      "а лаги, попадающие на них, заменяются пропусками.\n")

    w("## 5. Структурные сдвиги и аномалии\n")
    w(md_table(pd.DataFrame([
        (f"{STUDENT_OMNY_START:%d.%m.%Y}", "карты OMNY школьникам: бесплатно, круглосуточно, по выходным",
         "ступенька спроса, сильнее ночью, в выходные и вне Манхэттена; флаг `regime_student_omny`"),
        (f"{CONGESTION_PRICING_START:%d.%m.%Y}", "плата за въезд в центр Манхэттена",
         "на данных о входах отдельный эффект не выделяется; флаг `cp_active`"),
        ("29.09.2023", "ливень и наводнение", "−31% за сутки"),
        ("13.02.2024", "снегопад, школы на удалёнке", "−37%"),
        ("25–26.01.2026", "сильный снегопад", "−57% и −51%"),
        ("23.02.2026", "буран", "−78%, худший день периода"),
        ("январь–май 2025", "закрытие линии на Рокавэй (ремонт)", "11 станций, помечены `is_closed`"),
        ("лето 2024", "частичное закрытие линии G", "помечено `is_closed`"),
        ("май 2025", "Canarsie-Rockaway Pkwy (L): уровень вырос в ~5 раз за неделю и остался",
         "скорее изменение учёта/входов станции, а не спроса; в train есть оба уровня"),
        ("2023 – авг. 2024", "Woodhaven Blvd, 75 St-Elderts Ln (J, Z): спрос ниже вдвое, затем возврат",
         "частичные закрытия (одно направление) правилом `is_closed` не ловятся"),
        ("2023–2026", "52 St и 69 St (7): уровень упал вдвое",
         "вероятно, работы на станциях; ещё 3 станции с ростом в 1,7–1,9 раза"),
    ], columns=["Дата", "Событие", "Как видно в данных"])))
    w("")

    w("## 6. Признаки (ковариаты)\n")
    w("«Известен заранее» — признак доступен на момент прогноза без допущений.\n")
    w(md_table(pd.DataFrame([
        ("календарь", "hour, dow, month, dayofyear, week_of_year, циклические sin/cos", "да"),
        ("праздники", "is_holiday, holiday_name, is_day_before/after_holiday, xmas_week", "да"),
        ("школа", "is_school_day (официальные календари NYC Public Schools)", "да"),
        ("режимы", "regime_student_omny, cp_active", "да"),
        ("городские события", "city_event, is_city_event, holiday_shopping_weekend", "да"),
        ("события на площадках", "ev_today, ev_active, ev_hours_from_start, ev_kind (MLB, MiLB, NBA, WNBA, US Open)",
         "да (расписания); отмены учтены по факту"),
        ("погода", "температура, осадки, снег, ветер и др.: город / сетка 0.1° / сетка 0.25° (Open-Meteo, ERA5)",
         "нет — фактическая погода; допущение «идеальный прогноз погоды»"),
        ("станция", "scale_weekday, commute_index, cluster, n_lines, in_cbd, borough, координаты, venue", "да (по train)"),
        ("граф", "od_out_flow, od_in_flow, od_out_degree; матрицы смежности в `graph/`", "да (по OD 2024)"),
        ("закрытия", "is_closed (по дням)",
         "нет: флаг вычислен по фактическому спросу этого же дня — использовать только как маску "
         "(исключать из обучения и метрик), но не как входной признак, иначе утечка"),
    ], columns=["Группа", "Признаки", "Известен заранее"])))
    w("")

    gdir = PROCESSED / "graph"
    if gdir.exists():
        w("## 7. Граф станций\n")
        rows = []
        for f in sorted(gdir.glob("adj_*.npy")):
            A = np.load(f)
            n = A.shape[0]
            nz = (A > 0).sum()
            rows.append({"матрица": f.stem, "узлов": n, "рёбер": fmt(nz), "плотность": f"{nz / (n * (n - 1)):.2%}",
                         "ср. степень": fmt((A > 0).sum(axis=1).mean(), 1),
                         "изолированных": int(((A > 0).sum(axis=1) == 0).sum())})
        w(md_table(pd.DataFrame(rows)))
        w("\n`adj_dist` — гауссово ядро по расстоянию; `adj_line` — соседи по линиям; "
          "`adj_od`, `adj_od_am` — средний поток поездок в будний день (весь день и 6–9 ч), направленные, почти полные; "
          "`adj_od_top10`, `adj_od_am_top10` — для каждой станции 10 направлений с наибольшим потоком "
          "(рекомендуются для графовых моделей).\n")

    lab = PROCESSED / "lab"
    if (lab / "split.json").exists():
        meta = json.loads((lab / "split.json").read_text(encoding="utf-8"))
        w("## 8. Файлы для моделей временных рядов (`data/processed/lab/`)\n")
        w(md_table(pd.DataFrame([
            ("ridership_wide.parquet", f"{meta['n_hours']} часов × {meta['n_stations']} станций, int32"),
            ("closed_wide.parquet", "та же форма, True = станция закрыта"),
            ("covariates_time.parquet", "ковариаты по часам: календарь, режимы, события, погода (город)"),
            ("station_static.parquet", "статические признаки станций"),
            ("split.json", "границы выборок и порядок станций"),
            ("../graph/nodes.csv, adj_*.npy", "граф; порядок узлов = порядок столбцов ridership_wide"),
        ], columns=["Файл", "Содержимое"])))
        w("")

    w("## 9. Источники\n")
    w("- MTA Subway Hourly Ridership: https://data.ny.gov/d/wujg-7c2s, https://data.ny.gov/d/5wq4-mkjj")
    w("- MTA Subway Origin-Destination Ridership Estimate 2024: https://data.ny.gov/d/jsu2-fbtj")
    w("- Open-Meteo Historical Weather API: https://open-meteo.com/en/docs/historical-weather-api")
    w("- Календари NYC Public Schools 2022/23 – 2026/27: https://www.schools.nyc.gov/calendar")
    w("- MLB Stats API: https://statsapi.mlb.com; ESPN API (NBA, WNBA); US Open: https://www.usopen.org")

    path = REPORTS / "data_passport.md"
    path.write_text("\n".join(out) + "\n", encoding="utf-8")
    print(f"Сохранено: {path}")


if __name__ == "__main__":
    main()
