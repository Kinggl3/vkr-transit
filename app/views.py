"""Страницы прототипа."""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import data as D

C = {"actual": "#1f1f1f", "hyb": "#2a78d6", "lgbm": "#eb6834", "chronos": "#1baf7a", "norm": "#9a988f", "band": "rgba(42,120,214,0.15)"}
DIVERGING = [[0, "#2166ac"], [0.25, "#92c5de"], [0.5, "#f2f1ec"], [0.75, "#f4a582"], [1, "#b2182b"]]
NYC = {"lat": 40.73, "lon": -73.94}
CSS = """
<style>
.block-container {padding-top: 2rem; max-width: 1400px;}
div[data-testid="stMetricValue"] {font-size: 1.9rem;}
.lead {font-size: 1.05rem; color: #55534c; max-width: 900px; margin-bottom: 1.2rem;}
.note {font-size: 0.9rem; color: #77756c;}
</style>
"""


def lead(text: str):
    st.markdown(f'<div class="lead">{text}</div>', unsafe_allow_html=True)


def note(text: str):
    st.markdown(f'<div class="note">{text}</div>', unsafe_allow_html=True)


def pct(x, signed=False, d=1):
    s = f"{100 * x:+.{d}f}%" if signed else f"{100 * x:.{d}f}%"
    return s.replace(".", ",").replace("-", "−")


def num(x):
    return f"{x:,.0f}".replace(",", " ").replace("-", "−")


def layout(fig, h=380, legend=True, dates=False):
    fig.update_layout(height=h, margin=dict(l=10, r=10, t=30, b=10), plot_bgcolor="white", hovermode="x unified",
                      legend=dict(orientation="h", y=1.08, x=0) if legend else None, showlegend=legend,
                      font=dict(size=13))
    fig.update_xaxes(showgrid=True, gridcolor="#ecebe6")
    fig.update_yaxes(showgrid=True, gridcolor="#ecebe6", rangemode="tozero")
    if dates:
        fig.update_xaxes(tickformat="%d.%m", hoverformat="%d.%m %H:00")
    return fig


def day_picker(label="День", key=None, options=None, default=None):
    days = list(options if options is not None else D.days())
    idx = days.index(pd.Timestamp(default)) if default is not None and pd.Timestamp(default) in days else 0
    return st.selectbox(label, days, index=idx, format_func=D.fmt_day, key=key)


def network_trace():
    """Серые линии метро: соседние станции на одной линии."""
    stn, e = D.stations(), D.edges()
    xs, ys = [], []
    for i, j in e:
        xs += [stn["lon"].iat[i], stn["lon"].iat[j], None]
        ys += [stn["lat"].iat[i], stn["lat"].iat[j], None]
    return go.Scatter(x=xs, y=ys, mode="lines", line=dict(color="#c9c7be", width=1.5), hoverinfo="skip", showlegend=False)


def scatter_map(df, color, size, hover, range_color, color_title, animation=None, h=640, scale=DIVERGING):
    """Схема сети: станции на координатах поверх линий метро. С галочкой «подложка» — карта улиц (нужен интернет)."""
    streets = st.session_state.get("streets", False)
    kw = dict(color=color, size=size, size_max=18, hover_name="станция", hover_data=hover,
              color_continuous_scale=scale, range_color=range_color, animation_frame=animation)
    if streets:
        fig = px.scatter_map(df, lat="lat", lon="lon", center=NYC, zoom=10.2, map_style="carto-positron", **kw)
        fig.update_layout(height=h, margin=dict(l=0, r=0, t=0, b=0))
    else:
        fig = px.scatter(df, x="lon", y="lat", **kw)
        fig.add_trace(network_trace())
        fig.data = (fig.data[-1],) + fig.data[:-1]                    # линии под станциями
        if animation:
            for fr in fig.frames:
                fr.traces = [1]
        fig.update_traces(marker=dict(line=dict(width=0.5, color="white")), selector=dict(mode="markers"))
        fig.update_xaxes(visible=False, range=[-74.06, -73.74])
        fig.update_yaxes(visible=False, scaleanchor="x", scaleratio=1.32, range=[40.56, 40.91])
        fig.update_layout(height=h, margin=dict(l=0, r=0, t=0, b=0), plot_bgcolor="white")
    fig.update_layout(coloraxis_colorbar=dict(title=color_title, ticksuffix="%", len=0.6))
    return fig


def map_toggle():
    st.toggle("Подложка с улицами (нужен интернет)", key="streets")


# ---------------------------------------------------------------------------
def overview():
    st.title("Прогноз пассажиропотока метро Нью-Йорка")
    lead("Каждый час модель прогнозирует, сколько человек войдёт на каждую из 424 станций в следующие 24 часа. "
         "Итоговая модель — гибрид: смешивание LightGBM (календарь, события, погода), графовой сети "
         "(связи между станциями) и Chronos-2 (предобученная модель для рядов). Ниже — тестовый период "
         "1 июля – 24 сентября 2026 года, который модели не видели при обучении.")
    s = D.series()
    m = D.result("metrics_test.csv")
    c1, c2, c3, c4 = st.columns(4)
    if m is not None:
        g = m[(m.subset == "все точки") & (m.slice == "всё")].set_index("model")["wape"]
        c1.metric("Ошибка гибрида (WAPE, тест)", pct(g.get("hyb_blend", np.nan)))
        c2.metric("Точнее LightGBM", pct(g["hyb_blend"] / g["lgbm"] - 1, True), delta_color="off")
        c3.metric("Точнее эталона (норма + поправка)", pct(g["hyb_blend"] / g["norm_level"] - 1, True), delta_color="off")
    agg = D.result("report_test_aggregation.csv")
    if agg is not None and "hyb_blend" in set(agg["model"]):
        c4.metric("Город за сутки", pct(agg.set_index("model").loc["hyb_blend", "город × сутки"]))
    note("WAPE — сумма абсолютных ошибок, делённая на сумму фактических входов. 9% ≈ 30 пассажиров на станцию в час.")

    st.subheader("Весь город по дням")
    day = s["hours"].normalize()
    tot = pd.DataFrame({"факт": np.nansum(s["actual"], 1), "прогноз на сутки": np.nansum(s["da_hyb"], 1),
                        "обычный уровень": np.nansum(s["norm"], 1)}, index=s["hours"]).groupby(day).sum() / 1e6
    fig = go.Figure()
    fig.add_scatter(x=tot.index, y=tot["обычный уровень"], name="обычный уровень (норма)", line=dict(color=C["norm"], dash="dot"))
    fig.add_scatter(x=tot.index, y=tot["прогноз на сутки"], name="прогноз гибрида на сутки", line=dict(color=C["hyb"], width=2))
    fig.add_scatter(x=tot.index, y=tot["факт"], name="факт", line=dict(color=C["actual"], width=1.5))
    fig.update_yaxes(title="млн входов в сутки")
    st.plotly_chart(layout(fig, 360, dates=True), use_container_width=True)
    note("Провалы — выходные и праздники (4 июля, День труда 7 сентября). Прогноз на сутки делается в 00:00 каждого дня.")

    st.subheader("Неделя по часам")
    wk = day_picker("Начало недели", key="ov_week", default=pd.Timestamp("2026-07-13"))
    i = s["hours"].searchsorted(wk)
    sl = slice(i, i + 24 * 7)
    fig = go.Figure()
    fig.add_scatter(x=s["hours"][sl], y=np.nansum(s["h1_hyb"][sl], 1) / 1e3, name="прогноз на час вперёд", line=dict(color=C["hyb"], width=2))
    fig.add_scatter(x=s["hours"][sl], y=np.nansum(s["actual"][sl], 1) / 1e3, name="факт", line=dict(color=C["actual"], width=1.5))
    fig.update_yaxes(title="тыс. входов в час")
    st.plotly_chart(layout(fig, 320, dates=True), use_container_width=True)


# ---------------------------------------------------------------------------
def city_map():
    st.title("Карта")
    lead("Прогноз на выбранный час по каждой станции. Цвет — насколько поток отличается от обычного для этой станции "
         "в этот час (медиана за 4 недели): красный — больше обычного, синий — меньше. Размер — число входов.")
    s, stn = D.series(), D.stations()
    c1, c2, c3 = st.columns([2, 3, 3])
    with c1:
        day = day_picker(key="map_day", default=pd.Timestamp("2026-07-04"))
    with c2:
        hour = st.slider("Час", 0, 23, 18, key="map_hour")
    with c3:
        layer = st.radio("Что показать", ["прогноз на час вперёд", "факт", "ошибка прогноза"], horizontal=True)
    t = D.day_slice(day).start + hour
    y, p, n = s["actual"][t], s["h1_hyb"][t], s["norm"][t]
    df = stn.assign(станция=stn["short"], факт=np.round(y), прогноз=np.round(p), норма=np.round(n))
    if layer == "ошибка прогноза":
        df["цвет"] = 100 * (p - y) / np.maximum(y, 30)
        title, rng = "прогноз − факт", (-40, 40)
    else:
        v = p if layer.startswith("прогноз") else y
        df["цвет"] = 100 * ((v + D.EPS) / (n + D.EPS) - 1)
        title, rng = "к обычному", (-50, 50)
    df["размер"] = np.sqrt(np.nan_to_num(np.where(layer == "факт", y, p)) + 1)
    df = df.dropna(subset=["цвет"])
    fig = scatter_map(df, "цвет", "размер", {"факт": True, "прогноз": True, "норма": True, "линии": True, "размер": False, "цвет": ":.0f"},
                      rng, title)
    ev = st.plotly_chart(fig, use_container_width=True, on_select="rerun", selection_mode="points", key="map_chart")
    map_toggle()
    tot_y, tot_p = np.nansum(y), np.nansum(p)
    c1, c2, c3 = st.columns(3)
    c1.metric("Входов за час по городу (факт)", num(tot_y))
    c2.metric("Прогноз на час", num(tot_p), pct(tot_p / tot_y - 1, True), delta_color="off")
    c3.metric("Ошибка по станциям (WAPE)", pct(D.wape(y, p)))
    pts = (ev or {}).get("selection", {}).get("points", []) if isinstance(ev, dict) else getattr(getattr(ev, "selection", None), "points", [])
    if pts:
        name = pts[0].get("hovertext")
        row = stn[stn["short"] == name].iloc[0]
        st.session_state["station_id"] = row["id"]
        st.subheader(row["name"])
        _station_day_chart(row, day)
        note("Эта станция уже выбрана на странице «Станция» — там можно посмотреть любые дни и сравнить модели.")
    else:
        note("Нажмите на станцию, чтобы увидеть её сутки.")


def _station_day_chart(row, day):
    s = D.series()
    j = int(np.where(D.stations()["id"] == row["id"])[0][0])
    sl = D.day_slice(day)
    hrs = s["hours"][sl].hour
    fig = go.Figure()
    fig.add_scatter(x=hrs, y=s["norm"][sl, j], name="обычный уровень", line=dict(color=C["norm"], dash="dot"))
    fig.add_scatter(x=hrs, y=s["da_hyb"][sl, j], name="прогноз на сутки", line=dict(color=C["hyb"], width=2, dash="dash"))
    fig.add_scatter(x=hrs, y=s["h1_hyb"][sl, j], name="прогноз на час", line=dict(color=C["hyb"], width=2))
    fig.add_scatter(x=hrs, y=s["actual"][sl, j], name="факт", line=dict(color=C["actual"], width=1.5), mode="lines+markers", marker=dict(size=4))
    fig.update_xaxes(title="час", dtick=2)
    fig.update_yaxes(title="входов в час")
    st.plotly_chart(layout(fig, 300), use_container_width=True)


# ---------------------------------------------------------------------------
def station():
    st.title("Станция")
    s, stn = D.series(), D.stations()
    order = stn.sort_values("name")
    ids = list(order["id"])
    default = st.session_state.get("station_id", stn.loc[stn["weekday_entries"].idxmax(), "id"])
    c1, c2, c3 = st.columns([4, 2, 2])
    with c1:
        sid = st.selectbox("Станция", ids, index=ids.index(default) if default in ids else 0,
                           format_func=lambda i: stn.set_index("id").loc[i, "name"])
    st.session_state["station_id"] = sid
    with c2:
        start = day_picker("С", key="st_start", default=pd.Timestamp("2026-07-13"))
    with c3:
        ndays = st.select_slider("Дней", [1, 3, 7, 14], value=7)
    mode = st.radio("Горизонт прогноза", ["на час вперёд", "на сутки вперёд (прогноз в 00:00)"], horizontal=True)
    models = st.multiselect("Модели", list(D.MODEL_NAMES), default=["hyb"], format_func=D.MODEL_NAMES.get)
    row = stn.set_index("id").loc[sid]
    j = int(np.where(stn["id"] == sid)[0][0])
    pre = "h1" if mode.startswith("на час") else "da"
    i = s["hours"].searchsorted(start)
    sl = slice(i, i + 24 * ndays)
    x = s["hours"][sl]

    info = f"{row['borough']} · линии {row['lines']} · в среднем {num(row['weekday_entries'])} входов в будний день"
    if row["venue"]:
        info += f" · рядом {row['venue']}"
    note(info)

    fig = go.Figure()
    band = s[f"band_{pre}"][:, j]
    base = s[f"{pre}_hyb"][sl, j]
    lo, hi = (base + D.EPS) * band[0] - D.EPS, (base + D.EPS) * band[1] - D.EPS
    fig.add_scatter(x=x, y=np.clip(hi, 0, None), line=dict(width=0), showlegend=False, hoverinfo="skip")
    fig.add_scatter(x=x, y=np.clip(lo, 0, None), fill="tonexty", fillcolor=C["band"], line=dict(width=0),
                    name="80%-интервал гибрида", hoverinfo="skip")
    fig.add_scatter(x=x, y=s["norm"][sl, j], name="обычный уровень", line=dict(color=C["norm"], dash="dot"))
    for k in models:
        if f"{pre}_{k}" in s:
            fig.add_scatter(x=x, y=s[f"{pre}_{k}"][sl, j], name=D.MODEL_NAMES[k], line=dict(color=C[k], width=2))
    fig.add_scatter(x=x, y=s["actual"][sl, j], name="факт", line=dict(color=C["actual"], width=1.5))
    ev = D.events()
    if row["venue"] and len(ev):
        e = ev[ev["kind"].map(D.KIND_VENUE) == row["venue"]]
        e = e[(e.start >= x[0]) & (e.start <= x[-1])]
        for _, r in e.iterrows():
            fig.add_vline(x=r["start"], line=dict(color="#b2182b" if r["status"] == "final" else "#eda100", dash="dash", width=1))
        if len(e):
            note("Пунктирные вертикальные линии — матчи: красные — состоялись, жёлтые — перенесены или отменены.")
    fig.update_yaxes(title="входов в час")
    st.plotly_chart(layout(fig, 420, dates=True), use_container_width=True)

    st.subheader("Точность на этой станции за весь тестовый период")
    rows = []
    for k, nm in D.MODEL_NAMES.items():
        if f"h1_{k}" in s:
            rows.append({"модель": nm, "на час вперёд": pct(D.wape(s["actual"][:, j], s[f"h1_{k}"][:, j])),
                         "на сутки вперёд": pct(D.wape(s["actual"][:, j], s[f"da_{k}"][:, j]))})
    rows.append({"модель": "обычный уровень (норма)", "на час вперёд": pct(D.wape(s["actual"][:, j], s["norm"][:, j])),
                 "на сутки вперёд": pct(D.wape(s["actual"][:, j], s["norm"][:, j]))})
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=False)
    note("Интервал — 10-й и 90-й процентили отношения факта к прогнозу гибрида на этой станции на валидационной выборке.")


# ---------------------------------------------------------------------------
PRESETS = {"4 июля, День независимости": "2026-07-04", "18 июля, ливень 31 мм": "2026-07-18",
           "7 сентября, День труда": "2026-09-07", "14 июля, обычный вторник": "2026-07-14",
           "13 сентября, воскресенье с ливнем": "2026-09-13"}


def replay():
    st.title("Повтор дня")
    lead("Как менялся поток по городу в течение суток и что ожидала модель. Цвет станции — отклонение от обычного "
         "уровня в этот час. Нажмите ▶ под картой, чтобы проиграть сутки.")
    s, stn = D.series(), D.stations()
    c1, c2 = st.columns([3, 3])
    with c1:
        preset = st.selectbox("Интересные дни", ["выбрать любой"] + list(PRESETS), index=1)
    with c2:
        day = day_picker(key="rp_day", default=PRESETS.get(preset, "2026-07-04")) if preset == "выбрать любой" \
            else pd.Timestamp(PRESETS[preset])
    what = st.radio("На карте", ["факт", "прогноз гибрида на сутки"], horizontal=True)
    sl = D.day_slice(day)
    y, p, n = s["actual"][sl], s["da_hyb"][sl], s["norm"][sl]
    v = y if what == "факт" else p
    hrs = s["hours"][sl].hour
    frames = []
    for k, hh in enumerate(hrs):
        frames.append(stn.assign(станция=stn["short"], час=f"{hh:02d}:00", значение=np.round(v[k]),
                                 отклонение=100 * ((v[k] + D.EPS) / (n[k] + D.EPS) - 1),
                                 размер=np.sqrt(np.nan_to_num(v[k]) + 1)))
    df = pd.concat(frames).dropna(subset=["отклонение"])
    fig = scatter_map(df, "отклонение", "размер", {"значение": True, "линии": True, "размер": False, "отклонение": ":.0f"},
                      (-60, 60), "к обычному", animation="час", h=600)
    st.plotly_chart(fig, use_container_width=True)
    map_toggle()

    c1, c2 = st.columns([3, 2])
    with c1:
        fig = go.Figure()
        fig.add_scatter(x=hrs, y=np.nansum(n, 1) / 1e3, name="обычный уровень", line=dict(color=C["norm"], dash="dot"))
        fig.add_scatter(x=hrs, y=np.nansum(s["da_lgbm"][sl], 1) / 1e3, name="LightGBM на сутки", line=dict(color=C["lgbm"], width=1.5))
        fig.add_scatter(x=hrs, y=np.nansum(p, 1) / 1e3, name="гибрид на сутки", line=dict(color=C["hyb"], width=2))
        fig.add_scatter(x=hrs, y=np.nansum(y, 1) / 1e3, name="факт", line=dict(color=C["actual"], width=2))
        fig.update_xaxes(title="час", dtick=2)
        fig.update_yaxes(title="тыс. входов в час")
        st.plotly_chart(layout(fig, 340), use_container_width=True)
    with c2:
        st.metric("Входов за сутки", num(np.nansum(y)))
        st.metric("Прогноз гибрида на сутки", num(np.nansum(p)), pct(np.nansum(p) / np.nansum(y) - 1, True), delta_color="off")
        st.metric("Ошибка по станциям и часам: гибрид / обычный уровень", f"{pct(D.wape(y, p))} / {pct(D.wape(y, n))}")
    err = pd.DataFrame({"станция": stn["short"], "факт": np.nansum(y, 0), "прогноз": np.nansum(p, 0)})
    err["разница"] = err["прогноз"] - err["факт"]
    st.subheader("Где прогноз на этот день ошибся сильнее всего")
    t = err.reindex(err["разница"].abs().sort_values(ascending=False).index).head(10)
    t["ошибка"] = (t["разница"] / t["факт"].clip(lower=1)).map(lambda x: pct(x, True, 0))
    st.dataframe(t.assign(факт=t["факт"].map(num), прогноз=t["прогноз"].map(num), разница=t["разница"].map(num)),
                 hide_index=True, use_container_width=True)


# ---------------------------------------------------------------------------
def detector():
    st.title("Детектор сбоев")
    lead("Если на станции несколько часов подряд входит намного меньше людей, чем ожидала модель, вероятна проблема: "
         "сбой на линии, закрытый вход, перекрытие. Детектор сравнивает факт за последние 3 часа с прогнозом на час вперёд "
         "и отмечает станции, где расхождение выходит за обычный для них разброс (больше 2,5 стандартных отклонений).")
    s = D.series()
    cnt = D.anomaly_counts()
    st.subheader("Сколько станций одновременно «проседали» — по часам тестового периода")
    fig = go.Figure(go.Scatter(x=cnt.index, y=cnt.values, line=dict(color="#b2182b", width=1), fill="tozeroy",
                               fillcolor="rgba(178,24,43,0.12)", name="станций с провалом"))
    fig.update_yaxes(title="станций")
    st.plotly_chart(layout(fig, 240, legend=False, dates=True), use_container_width=True)
    top = cnt[cnt > 0].sort_values(ascending=False)
    picks = []
    for t in top.index:
        if all(abs((t - u).total_seconds()) > 6 * 3600 for u in picks):
            picks.append(t)
        if len(picks) == 8:
            break
    labels = {t: f"{D.fmt_day(t)}, {t:%H}:00 — {int(cnt[t])} ст." for t in picks}
    c1, c2, c3 = st.columns([3, 2, 2])
    with c1:
        pick = st.selectbox("Крупнейшие эпизоды", ["выбрать время вручную"] + picks,
                            format_func=lambda t: t if isinstance(t, str) else labels[t])
    if isinstance(pick, str):
        with c2:
            day = day_picker(key="dt_day", default=pd.Timestamp("2026-07-14"))
        with c3:
            hour = st.slider("Час", 0, 23, 8, key="dt_hour")
        t = D.day_slice(day).start + hour
    else:
        t = int(s["hours"].get_loc(pick))
    st.markdown(f"**{D.fmt_day(s['hours'][t])}, {s['hours'][t]:%H}:00** — окно {s['hours'][max(0, t - 2)]:%H}:00–{s['hours'][t]:%H}:59")
    an = D.anomalies(t)
    drops, surges = an[an["z"] < 0], an[an["z"] > 0]
    c1, c2, c3 = st.columns(3)
    c1.metric("Станций с провалом", len(drops))
    c2.metric("Не вошло пассажиров (оценка)", num(-drops["разница"].sum()) if len(drops) else "0")
    c3.metric("Станций со всплеском", len(surges))
    if len(an):
        df = an.assign(сила=an["z"].abs(), тип=np.where(an["z"] < 0, "провал", "всплеск"))
        fig = px.scatter(df, x="lon", y="lat", color="тип", size="сила", size_max=24, hover_name="станция",
                         hover_data={"факт": ":.0f", "прогноз": ":.0f", "линии": True, "сила": False, "lat": False, "lon": False},
                         color_discrete_map={"провал": "#2166ac", "всплеск": "#b2182b"})
        fig.add_trace(network_trace())
        fig.data = (fig.data[-1],) + fig.data[:-1]
        fig.update_xaxes(visible=False, range=[-74.06, -73.74])
        fig.update_yaxes(visible=False, scaleanchor="x", scaleratio=1.32, range=[40.56, 40.91])
        fig.update_layout(height=480, margin=dict(l=0, r=0, t=0, b=0), legend=dict(orientation="h", y=1.02), plot_bgcolor="white")
        st.plotly_chart(fig, use_container_width=True)
        tb = an[["станция", "район", "линии", "факт", "прогноз", "разница", "z"]].copy()
        for c in ("факт", "прогноз", "разница"):
            tb[c] = tb[c].map(num)
        tb["z"] = tb["z"].round(1)
        st.dataframe(tb, hide_index=True, use_container_width=True)
        lines = pd.Series([ln for ls in drops["линии"].fillna("") for ln in ls.split(",") if ln]).value_counts()
        if len(lines) and lines.iloc[0] >= 3:
            st.info(f"Провалы сосредоточены на линиях: {', '.join(lines.index[lines >= 3])} — похоже на сбой на линии.")
    else:
        st.success("В этот час аномалий нет.")
    note("Прогноз на час делается по данным до начала часа, поэтому детектор работает в реальном времени, "
         "как только MTA публикует входы за прошедший час. Порог и окно — параметры прототипа.")


# ---------------------------------------------------------------------------
def what_if():
    st.title("Что если")
    lead("Как изменится прогноз на сутки, если поменять условия: пойдёт дождь, наступит жара, день станет "
         "праздничным или на стадионе пройдёт матч. Эффект считает LightGBM — та часть гибрида, которая знает о календаре, "
         "погоде и событиях; он применяется к прогнозу гибрида на этот день.")
    sc, s, stn = D.scenarios(), D.series(), D.stations()
    if sc is None:
        st.warning("Нет файла сценариев: запустите src/app_data.py с параметром --model.")
        return
    c1, c2 = st.columns([2, 3])
    with c1:
        day = day_picker(key="wi_day", options=sc["days"], default=pd.Timestamp("2026-07-14"))
    with c2:
        k = st.selectbox("Сценарий", sc["keys"][1:], format_func=lambda x: sc["names"][sc["keys"].index(x)])
    di, ki = list(sc["days"]).index(day), sc["keys"].index(k)
    r = sc["ratio"][di, ki].astype("float32")                       # (24, N)
    sl = D.day_slice(day)
    base = np.nan_to_num(s["da_hyb"][sl])
    scen = (base + D.EPS) * r - D.EPS
    hrs = np.arange(base.shape[0])
    d_city = scen.sum() - base.sum()
    c1, c2, c3 = st.columns(3)
    c1.metric("Прогноз на сутки, весь город", num(scen.sum()), f"{num(d_city)} ({pct(scen.sum() / base.sum() - 1, True)})", delta_color="off")
    c2.metric("Без изменений", num(base.sum()))
    c3.metric("Фактически было", num(np.nansum(s["actual"][sl])))
    fig = go.Figure()
    fig.add_scatter(x=hrs, y=base.sum(1) / 1e3, name="прогноз без изменений", line=dict(color=C["norm"], width=2))
    fig.add_scatter(x=hrs, y=scen.sum(1) / 1e3, name="прогноз по сценарию", line=dict(color=C["hyb"], width=2.5))
    fig.update_xaxes(title="час", dtick=2)
    fig.update_yaxes(title="тыс. входов в час")
    st.plotly_chart(layout(fig, 320), use_container_width=True)

    eff = pd.DataFrame({"станция": stn["short"], "lat": stn["lat"], "lon": stn["lon"], "линии": stn["lines"],
                        "без изменений": base.sum(0), "по сценарию": scen.sum(0)})
    eff["изменение, %"] = 100 * (eff["по сценарию"] / eff["без изменений"].clip(lower=1) - 1)
    eff["изменение"] = eff["по сценарию"] - eff["без изменений"]
    eff["размер"] = np.sqrt(eff["без изменений"].clip(lower=0) + 1)
    c1, c2 = st.columns([3, 2])
    with c1:
        lim = float(np.clip(np.nanpercentile(np.abs(eff["изменение, %"]), 98), 5, 300))
        fig = scatter_map(eff, "изменение, %", "размер", {"изменение": ":.0f", "линии": True, "размер": False,
                                                           "изменение, %": ":.0f"}, (-lim, lim), "за сутки", h=480)
        st.plotly_chart(fig, use_container_width=True)
        map_toggle()
    with c2:
        st.markdown("**Станции с наибольшим изменением за сутки**")
        t = eff.reindex(eff["изменение"].abs().sort_values(ascending=False).index).head(12)
        st.dataframe(pd.DataFrame({"станция": t["станция"], "изменение": t["изменение"].map(num),
                                   "%": t["изменение, %"].map(lambda x: f"{x:+.0f}%".replace("-", "−"))}),
                     hide_index=True, use_container_width=True, height=450)
    note("Сценарии меняют только входные признаки модели (прогноз погоды, флаг праздника, расписание события) "
         "и не учитывают, например, дополнительные поезда. Это оценка модели, а не наблюдение.")


# ---------------------------------------------------------------------------
NAMES = {"snaive168": "Тот же час неделю назад", "ha": "Историческое среднее", "median4w": "Медиана 4 недель",
         "norm_level": "Эталон (норма + поправка)", "chronos2": "Chronos-2", "gnn_none": "Графовая сеть без графа",
         "gnn_od_in": "Графовая сеть, граф поездок", "lgbm": "LightGBM", "lgbm_final": "LightGBM финальный",
         "hyb_full": "Гибрид: мета-модель", "hybr_full": "Гибрид: поправка к смешиванию", "hyb_blend": "Гибрид (итоговая)"}


def quality():
    st.title("Качество моделей")
    lead("Все модели оценены одним кодом на одних и тех же точках прогноза. Модель выбиралась по валидационной выборке "
         "(апрель–июнь 2026), тестовая (июль–сентябрь) использовалась один раз.")
    mv, mt = D.result("metrics_val.csv"), D.result("metrics_test.csv")
    if mt is None:
        st.warning("Нет таблиц results/metrics_*.csv")
        return
    H_ = ["горизонт h1", "горизонт h2-3", "горизонт h4-6", "горизонт h7-12", "горизонт h13-24"]
    g = lambda m, sl: m[(m.subset == "все точки") & (m.slice == sl)].set_index("model")["wape"]
    rows = []
    for k, nm in NAMES.items():
        if k in g(mt, "всё"):
            rows.append({"модель": nm, "валидация": g(mv, "всё").get(k), "тест": g(mt, "всё")[k],
                         "тест, 1 ч": g(mt, "горизонт h1")[k], "тест, 13–24 ч": g(mt, "горизонт h13-24")[k],
                         "тест, праздники": g(mt, "день: праздник").get(k)})
    t = pd.DataFrame(rows).sort_values("тест", ascending=False)
    st.subheader("Ошибка WAPE")
    st.dataframe(t.style.format({c: lambda x: pct(x) for c in t.columns[1:]}).highlight_min(subset=t.columns[1:], color="#d6e6f8"),
                 hide_index=True, use_container_width=True, height=36 * (len(t) + 1) + 3)

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("По горизонту прогноза (тест)")
        fig = go.Figure()
        for k, col in (("norm_level", C["norm"]), ("chronos2", C["chronos"]), ("gnn_od_in", "#e87ba4"), ("lgbm", C["lgbm"]), ("hyb_blend", C["hyb"])):
            v = mt[(mt.subset == "все точки") & (mt.model == k)].set_index("slice")["wape"].reindex(H_) * 100
            fig.add_scatter(x=["1", "2–3", "4–6", "7–12", "13–24"], y=v.values, name=NAMES[k], line=dict(color=col, width=2), mode="lines+markers")
        fig.update_xaxes(title="часов вперёд")
        fig = layout(fig, 380)
        fig.update_yaxes(title="WAPE, %", rangemode="normal")
        fig.update_layout(legend=dict(orientation="h", y=-0.25, x=0))
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        st.subheader("Уровни агрегации (гибрид, тест)")
        agg = D.result("report_test_aggregation.csv")
        if agg is not None:
            a = agg.set_index("model").loc["hyb_blend"].drop(labels=[c for c in agg.columns if c in ("Unnamed: 0", "model")], errors="ignore")
            fig = go.Figure(go.Bar(x=a.index, y=a.values.astype(float) * 100, marker_color=C["hyb"],
                                   text=[pct(v) for v in a.values.astype(float)], textposition="outside"))
            fig.update_yaxes(title="WAPE, %")
            st.plotly_chart(layout(fig, 360, legend=False), use_container_width=True)

    st.subheader("Сравнение с опубликованными работами")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**HW-DMD (Ханчжоу, 30 мин, входы), WMAPE**")
        ext = D.result("external_hangzhou_30_in.csv")
        rows = [{"модель": "HW-DMD (статья)", "1 шаг": 7.38, "2 шага": 8.05, "3 шага": 8.12},
                {"модель": "Историческое среднее (статья)", "1 шаг": 11.09, "2 шага": 11.09, "3 шага": 11.09}]
        if ext is not None:
            e = ext[ext["выборка"] == "test"]
            for k, nm in (("blend", "Наш подход: смешивание"), ("lgbm", "Наш LightGBM"), ("ha (тот же тип дня)", "Историческое среднее (наше)")):
                v = e[e["модель"] == k].set_index("шаг")["WMAPE, %"]
                rows.append({"модель": nm, "1 шаг": v[1], "2 шага": v[2], "3 шага": v[3]})
        st.dataframe(pd.DataFrame(rows).style.format({c: "{:.2f}%" for c in ("1 шаг", "2 шага", "3 шага")}), hide_index=True, use_container_width=True)
    with c2:
        st.markdown("**PVCGN (MAE, пассажиров за 15 мин), 15 / 60 минут**")
        rows = [{"модель": "PVCGN (статья)", "Ханчжоу": "22,68 / 24,93", "Шанхай": "23,29 / 26,29"},
                {"модель": "Graph WaveNet (статья)", "Ханчжоу": "24,07 / 29,14", "Шанхай": "24,91 / 30,90"}]
        hz, sh = D.result("external_hangzhou_15.csv"), D.result("external_shanghai_15.csv")
        if hz is not None and sh is not None:
            f = lambda r, k, h: r[(r["выборка"] == "test") & (r["модель"] == k) & (r["шаг"] == h)]["MAE"].iloc[0]
            for k, nm in (("blend", "Наш подход: смешивание"), ("lgbm4", "Наш LightGBM, только последний час")):
                rows.append({"модель": nm, "Ханчжоу": f"{f(hz, k, 1):.2f} / {f(hz, k, 4):.2f}".replace(".", ","),
                             "Шанхай": f"{f(sh, k, 1):.2f} / {f(sh, k, 4):.2f}".replace(".", ",")})
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
    note("Метод заново обучен на данных Ханчжоу и Шанхая; модель Нью-Йорка не переносилась.")
