# %% [markdown]
# # ЛР2 · эксперимент 2 · lab_08 — новые модальности и абляция (CPU)
#
# **Inputs:** датасет `VKR_PM232_Pevin_MTA`, выводы `lab-00-data`, `lab-06-sources`, `lab-07-radar`.
# **Accelerator:** нет. Время: ~1–1,5 ч.
#
# Те же объекты, разбиение и LightGBM, что в ЛР2 (lab_02). Новые группы признаков:
# - **R** — радарные осадки MRMS у станции (числа), **RI** — вырезка карты осадков вокруг станции (изображение);
# - **W** — предупреждения NWS по району (структура), **WT** — их текст (эмбеддинг BGE → PCA 8);
# - **E** — мероприятия с разрешением города у станции (структура), **ET** — их текст (BGE → PCA 16);
# - **S** — плановые работы: число остановок по расписанию в этот день против обычного.
#
# Абляция: U1 (числа) → U1 + каждая группа → ALL → ALL без каждого источника; контроли: шум и другой seed.
# Две цели: провал (PR-AUC, как в ЛР2) и регрессия y/ref (WAPE): сбои и погода дают падения на единицы
# процентов, и порог 25% их может не замечать. Источник, который не удалось получить, пропускается.

# %%
import os, sys, json, time, warnings
from pathlib import Path
import numpy as np, pandas as pd
import lightgbm as lgb
from sklearn.decomposition import PCA
from scipy import ndimage
warnings.filterwarnings("ignore", category=RuntimeWarning)


def _find_labkit():
    for root in ["/kaggle/input", "/kaggle/working", "."]:
        for dp, dn, fn in os.walk(root, followlinks=True):
            if "labkit.py" in fn:
                return dp
    raise FileNotFoundError("labkit.py не найден: подключите вывод ноутбука lab-00-data")


sys.path.insert(0, _find_labkit())
import labkit as lk
OUT = lk.results_dir()
PARTS = ["train", "val", "test", "holdout"]
d = lk.load()
IDX = d.y.index
H, S = len(IDX), d.y.shape[1]
BORO = d.static["borough"].to_numpy()
BOROS = sorted(set(BORO))
B_OF_S = np.array([BOROS.index(b) for b in BORO])
A_LINE = (d.adj_line > 0).astype("float32")
O = {p: lk.load_objects(p) for p in PARTS}
for p in PARTS:
    O[p]["z"] = np.log((O[p]["y"] + 10) / (O[p]["ref"] + 10)).astype("float32")
    print(p, len(O[p]), "провалов", round(O[p]["y_drop"].mean(), 4))


def hour_pos(ts):
    """Метка времени → номер часа в индексе ridership_wide (индекс — полная почасовая сетка)."""
    return ((pd.DatetimeIndex(ts) - IDX[0]) / pd.Timedelta(hours=1)).astype(int)


def maybe(name):
    try:
        return lk.find_file(name)
    except FileNotFoundError:
        return None


TRAIN_END = pd.Timestamp(lk.SPLIT["train"][1])
NEW = {}            # группа → список колонок
STATUS = {}

# %% [markdown]
# ## R, RI — радарные осадки MRMS

# %%
def p1h_mm(idx):
    idx = np.asarray(idx, dtype="float32")
    mm = np.where(idx <= 100, 0.25 * idx, np.where(idx <= 180, 25 + 1.25 * (idx - 100), 125 + 5 * (idx - 180)))
    return np.where(idx >= 255, np.nan, mm).astype("float32")


RAD = None
if maybe("radar_nyc.npy") is not None:
    crops = np.load(maybe("radar_nyc.npy"))
    meta = json.load(open(maybe("radar_meta.json")))
    rh = pd.read_parquet(maybe("radar_hours.parquet"))
    lat, lon = d.static["latitude"].to_numpy(), d.static["longitude"].to_numpy()
    rr = np.round((lat - meta["lat0"]) / meta["dy"]).astype(int)
    cc = np.round((lon - meta["lon0"]) / meta["dx"]).astype(int)
    assert rr.min() >= 16 and cc.min() >= 16 and rr.max() < meta["h"] - 16 and cc.max() < meta["w"] - 16
    F = len(crops)
    fr_feat = np.full((F, S, 5), np.nan, dtype="float32")     # пиксель, max 3×3, max 7×7, среднее 15×15, доля ≥1 мм 11×11
    fr_img = np.zeros((F, S, 16), dtype="float16")             # вырезка 32×32 → среднее по блокам 8×8
    fr_city = np.full(F, np.nan, dtype="float32")              # среднее по всем станциям
    fr_line = np.full((F, S), np.nan, dtype="float32")         # среднее по соседям по линии
    t0 = time.time()
    for f in range(F):
        mm = p1h_mm(crops[f])
        if np.isnan(mm).mean() > 0.5:
            continue
        m0 = np.nan_to_num(mm)
        fr_feat[f, :, 0] = m0[rr, cc]
        fr_feat[f, :, 1] = ndimage.maximum_filter(m0, 3)[rr, cc]
        fr_feat[f, :, 2] = ndimage.maximum_filter(m0, 7)[rr, cc]
        fr_feat[f, :, 3] = ndimage.uniform_filter(m0, 15)[rr, cc]
        fr_feat[f, :, 4] = ndimage.uniform_filter((m0 >= 1).astype("float32"), 11)[rr, cc]
        lg = np.log1p(m0)
        fr_city[f] = m0[rr, cc].mean()
        fr_line[f] = (A_LINE @ m0[rr, cc]) / np.maximum(A_LINE.sum(1), 1)
        for k in range(S):
            blk = lg[rr[k] - 16:rr[k] + 16, cc[k] - 16:cc[k] + 16].reshape(4, 8, 4, 8).mean((1, 3))
            fr_img[f, k] = blk.ravel()
    print(f"радар: {F} кадров, признаки за {time.time() - t0:.0f} c")
    # плотные массивы по часам индекса: кадр «сейчас» и «час назад» для каждого часа T
    cur = np.full(H, -1); prev = np.full(H, -1)
    pos = hour_pos(rh["T"]); okp = (pos >= 0) & (pos < H)
    cur[pos[okp]] = rh["cur"].to_numpy()[okp]; prev[pos[okp]] = rh["prev"].to_numpy()[okp]
    RAD = {"cur": cur, "prev": prev, "feat": fr_feat, "img": fr_img, "city": fr_city, "line": fr_line}
    STATUS["radar"] = {"frames": F, "hours_with_frame": float((cur >= 0).mean())}


def radar_block(t, s):
    c, pv = RAD["cur"][t], RAD["prev"][t]
    f = np.where((c >= 0)[:, None], RAD["feat"][np.clip(c, 0, None), s], np.nan)
    fp = np.where((pv >= 0)[:, None], RAD["feat"][np.clip(pv, 0, None), s], np.nan)
    cols = {"rad_pix": f[:, 0], "rad_max3": f[:, 1], "rad_max7": f[:, 2], "rad_mean15": f[:, 3],
            "rad_wet11": f[:, 4], "rad_prev_pix": fp[:, 0], "rad_prev_mean15": fp[:, 3],
            "rad_missing": (c < 0).astype("float32")}
    # средние по городу и по соседям по линии — из того же кадра
    cols["rad_city"] = np.where(c >= 0, RAD["city"][np.clip(c, 0, None)], np.nan)
    cols["rad_line"] = np.where(c >= 0, RAD["line"][np.clip(c, 0, None), s], np.nan)
    img = np.concatenate([RAD["img"][np.clip(c, 0, None), s], RAD["img"][np.clip(pv, 0, None), s]], 1).astype("float32")
    img[(c < 0)] = 0
    return pd.DataFrame(cols), img

# %% [markdown]
# ## W, WT — предупреждения NWS по району станции
# Активно в час цели [T, T+1): выпущено до T, начало < T+1 ч, конец > T. «Скоро»: выпущено до T,
# начало в ближайшие 24 ч. Текст: среднее эмбеддингов действующих и ожидаемых предупреждений → PCA 8.

# %%
NWS_GROUPS = ["flood", "storm", "heat", "winter", "cold", "wind", "tropical", "coastal", "other"]
NWS = None
if maybe("nws.parquet") is not None:
    nw = pd.read_parquet(maybe("nws.parquet"))
    keep_b = nw["borough"].isin(BOROS).to_numpy()            # на Статен-Айленде станций метро нет
    E = np.load(maybe("nws_emb.npy")).astype("float32")[keep_b] if maybe("nws_emb.npy") is not None else None
    nw = nw[keep_b].reset_index(drop=True)
    nb_ = len(BOROS)
    act = np.zeros((H, nb_, len(NWS_GROUPS)), "float32")
    soon = np.zeros((H, nb_), "float32")
    since = np.full((H, nb_), np.nan, "float32")
    to_begin = np.full((H, nb_), np.nan, "float32")
    emb_sum = np.zeros((H, nb_, E.shape[1]), "float32") if E is not None else None
    emb_w = np.zeros((H, nb_), "float32")
    hrs = IDX
    for i, r in nw.iterrows():
        b = BOROS.index(r["borough"]); g = NWS_GROUPS.index(r["group"]) if r["group"] in NWS_GROUPS else len(NWS_GROUPS) - 1
        lo = max(r["issued"], r["begin"] - pd.Timedelta(hours=1))
        a0 = int(np.searchsorted(hrs, lo, side="right")); a1 = int(np.searchsorted(hrs, r["end"], side="left"))
        if a1 > a0:
            act[a0:a1, b, g] = 1
            m = (hrs[a0:a1] - r["issued"]) / pd.Timedelta(minutes=1)
            since[a0:a1, b] = np.fmin(since[a0:a1, b], m.to_numpy(dtype="float32"))
        u0 = int(np.searchsorted(hrs, max(r["issued"], r["begin"] - pd.Timedelta(hours=24)), side="right"))
        u1 = int(np.searchsorted(hrs, r["begin"] - pd.Timedelta(hours=1), side="right"))
        if u1 > u0:
            soon[u0:u1, b] += 1
            h_to = (r["begin"] - hrs[u0:u1]) / pd.Timedelta(hours=1)
            to_begin[u0:u1, b] = np.fmin(to_begin[u0:u1, b], h_to.to_numpy(dtype="float32"))
        if emb_sum is not None:
            for x0, x1 in [(a0, a1), (u0, u1)]:
                if x1 > x0:
                    emb_sum[x0:x1, b] += E[i]; emb_w[x0:x1, b] += 1
    if emb_sum is not None:
        tr_rows = emb_w[:hour_pos([TRAIN_END])[0]] > 0
        X = (emb_sum[:hour_pos([TRAIN_END])[0]][tr_rows] / emb_w[:hour_pos([TRAIN_END])[0]][tr_rows][:, None])
        pca_w = PCA(8, random_state=lk.SEED).fit(X) if len(X) >= 8 else None
    NWS = {"act": act, "soon": soon, "since": since, "to_begin": to_begin, "emb_sum": emb_sum, "emb_w": emb_w,
           "pca": pca_w if emb_sum is not None else None}
    STATUS["nws"] = {"events": len(nw), "hours_any_active": float((act.sum((1, 2)) > 0).mean())}


def nws_block(t, s):
    b = B_OF_S[s]
    a = NWS["act"][t, b]
    cols = {f"nws_{g}": a[:, k] for k, g in enumerate(NWS_GROUPS)}
    cols["nws_n_active"] = a.sum(1)
    cols["nws_n_soon"] = NWS["soon"][t, b]
    cols["nws_min_since"] = NWS["since"][t, b]
    cols["nws_h_to_begin"] = NWS["to_begin"][t, b]
    txt = np.zeros((len(t), 8), "float32")
    if NWS["pca"] is not None:
        w = NWS["emb_w"][t, b]; m = w > 0
        txt[m] = NWS["pca"].transform(NWS["emb_sum"][t[m], b[m]] / w[m, None])
    return pd.DataFrame(cols), txt

# %% [markdown]
# ## E, ET — мероприятия у станции
# Активно в час цели: начало < T+1 ч, конец > T. «Скоро» — начало в ближайшие 3 ч, «недавно закончилось» —
# конец за последние 2 ч (расходятся люди). Прямая привязка — вес 1, станции в 800 м — вес 0,5.

# %%
EVT = None
if maybe("permits.parquet") is not None:
    ev = pd.read_parquet(maybe("permits.parquet"))
    ps = pd.read_parquet(maybe("permit_station.parquet"))
    E = np.load(maybe("permits_emb.npy")).astype("float32") if maybe("permits_emb.npy") is not None else None
    train_ev = (ev["start"] <= TRAIN_END).to_numpy()
    pca_e = PCA(16, random_state=lk.SEED).fit(E[train_ev]) if E is not None and train_ev.sum() > 100 else None
    EP = pca_e.transform(E).astype("float32") if pca_e is not None else None
    parade = ev["type"].str.contains("parade", case=False).to_numpy()
    full = ev["full_closure"].astype(bool).to_numpy()
    direct_n = np.zeros((H, S), "float32"); near_n = np.zeros((H, S), "float32")
    full_n = np.zeros((H, S), "float32"); parade_n = np.zeros((H, S), "float32")
    soon_n = np.zeros((H, S), "float32"); after_n = np.zeros((H, S), "float32")
    city_n = np.zeros(H, "float32")
    emb = np.zeros((H, S, 16), "float32") if EP is not None else None
    emb_w = np.zeros((H, S), "float32")
    st0 = np.searchsorted(IDX, ev["start"] - pd.Timedelta(hours=1), side="right")   # T > start − 1 ч
    en0 = np.searchsorted(IDX, ev["end"], side="left")                              # T < end
    so0 = np.searchsorted(IDX, ev["start"] - pd.Timedelta(hours=3), side="left")    # start − 3 ч ≤ T
    af1 = np.searchsorted(IDX, ev["end"] + pd.Timedelta(hours=2), side="left")      # T < end + 2 ч
    for i in range(len(ev)):
        if en0[i] > st0[i]:
            city_n[st0[i]:en0[i]] += 1
    t0 = time.time()
    for (i, lvl), g in ps.groupby(["event", "level"]):
        sts = g["s"].to_numpy()
        a, b = st0[i], en0[i]
        w = 1.0 if lvl == 2 else 0.5
        if b > a:
            if lvl == 2:
                direct_n[a:b, sts] += 1
                if full[i]: full_n[a:b, sts] += 1
                if parade[i]: parade_n[a:b, sts] += 1
            else:
                near_n[a:b, sts] += 1
            if emb is not None:
                emb[a:b, sts] += w * EP[i]; emb_w[a:b, sts] += w
        if lvl == 2:
            if st0[i] > so0[i]: soon_n[so0[i]:st0[i], sts] += 1
            if af1[i] > en0[i]: after_n[en0[i]:af1[i], sts] += 1
    print(f"мероприятия: {len(ev)} событий, {len(ps)} пар событие–станция, {time.time() - t0:.0f} c")
    EVT = dict(direct=direct_n, near=near_n, full=full_n, parade=parade_n, soon=soon_n, after=after_n,
               city=city_n, emb=emb, emb_w=emb_w)
    STATUS["permits"] = {"events": len(ev), "objects_hours_with_direct": float((direct_n.sum(1) > 0).mean())}


def evt_block(t, s):
    cols = {f"evt_{k}": EVT[k][t, s] for k in ["direct", "near", "full", "parade", "soon", "after"]}
    cols["evt_city"] = EVT["city"][t]
    txt = np.zeros((len(t), 16), "float32")
    if EVT["emb"] is not None:
        w = EVT["emb_w"][t, s]; m = w > 0
        txt[m] = EVT["emb"][t[m], s[m]] / w[m, None]
    return pd.DataFrame(cols), txt

# %% [markdown]
# ## S — плановые работы по расписанию
# Отношение числа остановок по расписанию в этот день к медиане того же дня недели за 8 предыдущих недель.
# 0 — поезда на станции не останавливаются, < 1 — часть линий или поездов снята.

# %%
SCH = None
if maybe("sched_daily.parquet") is not None:
    sd = pd.read_parquet(maybe("sched_daily.parquet"))
    piv = sd.pivot_table(index="day", columns="s", values="n", aggfunc="sum").reindex(columns=range(S)).fillna(0)
    piv = piv.asfreq("D")                                    # дни без данных — NaN по всем станциям
    typ = piv.groupby(piv.index.dayofweek).transform(lambda x: x.rolling(8, min_periods=3).median().shift(1))
    ratio = (piv / typ.replace(0, np.nan)).clip(0, 3)
    nbr = (np.nan_to_num(ratio.to_numpy()) @ A_LINE.T) / np.maximum(A_LINE.sum(1), 1)
    SCH = {"days": ratio.index, "ratio": ratio.to_numpy("float32"), "nbr": nbr.astype("float32"),
           "n": piv.to_numpy("float32")}
    STATUS["schedules"] = {"days": len(piv), "station_days_ratio_lt_0.5": float((ratio < 0.5).to_numpy().mean())}


def sched_block(T, s):
    di = np.searchsorted(SCH["days"], pd.DatetimeIndex(T).normalize())
    ok = (di < len(SCH["days"])) & (SCH["days"][np.clip(di, 0, len(SCH["days"]) - 1)] == pd.DatetimeIndex(T).normalize())
    di = np.clip(di, 0, len(SCH["days"]) - 1)
    r = np.where(ok, SCH["ratio"][di, s], np.nan)
    return pd.DataFrame({"sch_ratio": r, "sch_zero": np.where(ok, (r < 0.05).astype("float32"), np.nan),
                         "sch_nbr": np.where(ok, SCH["nbr"][di, s], np.nan)})

# %% [markdown]
# ## Сборка признаков объектов

# %%
IMG_PC = [f"rimg_{i}" for i in range(16)]
WT_PC = [f"wtxt_{i}" for i in range(8)]
ET_PC = [f"etxt_{i}" for i in range(16)]
pca_r = None
for p in PARTS:
    o = O[p]; t = o["t"].to_numpy(); s = o["s"].to_numpy()
    add = []
    if RAD is not None:
        f, img = radar_block(t, s)
        if pca_r is None:
            pca_r = PCA(16, random_state=lk.SEED).fit(img[np.abs(img).sum(1) > 0][::3])
        add += [f, pd.DataFrame(pca_r.transform(img), columns=IMG_PC)]
    if NWS is not None:
        f, txt = nws_block(t, s); add += [f, pd.DataFrame(txt, columns=WT_PC)]
    if EVT is not None:
        f, txt = evt_block(t, s); add += [f, pd.DataFrame(txt, columns=ET_PC)]
    if SCH is not None:
        add.append(sched_block(o["T"], s))
    if add:
        O[p] = pd.concat([o.reset_index(drop=True)] + [a.reset_index(drop=True) for a in add], axis=1)
cols = O["train"].columns
NEW = {k: [c for c in cols if c.startswith(pref) and c not in excl]
       for k, pref, excl in [("R", "rad_", []), ("RI", "rimg_", []), ("W", "nws_", []), ("WT", "wtxt_", []),
                             ("E", "evt_", []), ("ET", "etxt_", []), ("S", "sch_", [])]}
NEW = {k: v for k, v in NEW.items() if v}
print({k: len(v) for k, v in NEW.items()})
NOISE = [f"noise_{i}" for i in range(16)]
rng = np.random.default_rng(lk.SEED)
for p in PARTS:
    O[p][NOISE] = rng.normal(size=(len(O[p]), 16)).astype("float32")

# покрытие: доля объектов, где источник «сработал»
cov_rows = []
for p in PARTS:
    o = O[p]
    cov_rows.append({"part": p, "n": len(o),
                     "дождь ≥1 мм у станции": float((o.get("rad_pix", pd.Series(np.nan)) >= 1).mean()) if "R" in NEW else np.nan,
                     "есть предупреждение NWS": float((o.get("nws_n_active", 0) + o.get("nws_n_soon", 0) > 0).mean()) if "W" in NEW else np.nan,
                     "мероприятие у станции": float((o.get("evt_direct", 0) > 0).mean()) if "E" in NEW else np.nan,
                     "расписание < 80% обычного": float((o.get("sch_ratio", pd.Series(np.nan)) < 0.8).mean()) if "S" in NEW else np.nan})
coverage = pd.DataFrame(cov_rows)
coverage.to_csv(OUT / "coverage_ext.csv", index=False)
print(coverage.round(4).to_string(index=False))

# %% [markdown]
# ## Проверка сдвига (S16): корреляция источника с отклонением входов y/ref − 1 при сдвиге −3…+3 ч
# Если время источника перепутано (как UTC в оповещениях MTA), пик будет не на нуле.

# %%
def lag_corr(name, dense, p="train"):
    o = O[p].sample(min(300_000, len(O[p])), random_state=lk.SEED)
    t, s = o["t"].to_numpy(), o["s"].to_numpy()
    r = (o["y"] / o["ref"] - 1).clip(-1, 2).to_numpy()
    out = []
    for k in range(-3, 4):
        tt = np.clip(t - k, 0, H - 1)
        x = dense(tt, s)
        m = np.isfinite(x) & np.isfinite(r)
        out.append({"источник": name, "сдвиг, ч": k,
                    "corr": float(np.corrcoef(x[m], r[m])[0, 1]) if m.sum() > 100 and np.nanstd(x[m]) > 0 else np.nan})
    return out


sync = []
if RAD is not None:
    sync += lag_corr("радар: осадки у станции", lambda tt, s: np.where(RAD["cur"][tt] >= 0,
                     RAD["feat"][np.clip(RAD["cur"][tt], 0, None), s, 0], np.nan))
if NWS is not None:
    sync += lag_corr("NWS: действующих предупреждений", lambda tt, s: NWS["act"][tt, B_OF_S[s]].sum(1))
if EVT is not None:
    sync += lag_corr("мероприятия у станции", lambda tt, s: EVT["direct"][tt, s])
sync = pd.DataFrame(sync)
sync.to_csv(OUT / "sync_ext.csv", index=False)
if len(sync):
    print(sync.pivot(index="сдвиг, ч", columns="источник", values="corr").round(4).to_string())

# %% [markdown]
# ## Модели: LightGBM, как в lab_02 (ранняя остановка на val; holdout — переобучение на train + val)

# %%
NUM = lk.NUM_FEATURES
ALLNEW = sum(NEW.values(), [])
SRC = {"радар": NEW.get("R", []) + NEW.get("RI", []), "NWS": NEW.get("W", []) + NEW.get("WT", []),
       "мероприятия": NEW.get("E", []) + NEW.get("ET", []), "расписание": NEW.get("S", [])}
SRC = {k: v for k, v in SRC.items() if v}
FEATS = {"U1_num": NUM}
for g, c in NEW.items():
    FEATS[f"+{g}"] = NUM + c
PAIRS = {"радар": ["R", "RI"], "NWS": ["W", "WT"], "мероприятия": ["E", "ET"]}
for src, gs in PAIRS.items():
    if all(g in NEW for g in gs):
        FEATS[f"+{src}"] = NUM + SRC[src]          # структура + текст / изображение вместе
FEATS["ALL"] = NUM + ALLNEW
for src, c in SRC.items():
    FEATS[f"ALL−{src}"] = NUM + [x for x in ALLNEW if x not in c]
FEATS["NEW_only"] = ALLNEW
FEATS["C0_U1_plus_noise"] = NUM + NOISE
FEATS["C1_U1_seed2"] = NUM
print(len(FEATS), "моделей:", list(FEATS))

BASE = dict(learning_rate=0.05, num_leaves=63, min_data_in_leaf=200, feature_fraction=0.8, bagging_fraction=0.8,
            bagging_freq=1, lambda_l2=1.0, verbose=-1, seed=lk.SEED, num_threads=4)
TASKS = {"cls": dict(objective="binary", metric="average_precision", target="y_drop"),
         "reg": dict(objective="regression", metric="l1", target="z")}
MAX_ROUNDS = int(os.environ.get("LAB_MAX_ROUNDS", 3000))
PRED, INFO = {}, {}


def fit(name, feats, task):
    cfg = TASKS[task]
    params = {**BASE, "objective": cfg["objective"], "metric": cfg["metric"],
              "seed": lk.SEED + (1 if name.startswith("C1_") else 0)}
    tr, va = O["train"], O["val"]
    y = cfg["target"]
    dtr = lgb.Dataset(tr[feats], tr[y], weight=tr["weight"], free_raw_data=False)
    dva = lgb.Dataset(va[feats], va[y], reference=dtr)
    t0 = time.time()
    m = lgb.train(params, dtr, MAX_ROUNDS, valid_sets=[dva], callbacks=[lgb.early_stopping(100, verbose=False)])
    for p in ["val", "test"]:
        PRED[(task, name, p)] = m.predict(O[p][feats], num_iteration=m.best_iteration)
    trva = pd.concat([tr, va])
    m2 = lgb.train(params, lgb.Dataset(trva[feats], trva[y], weight=trva["weight"]), int(m.best_iteration * 1.1) + 1)
    PRED[(task, name, "holdout")] = m2.predict(O["holdout"][feats])
    INFO[(task, name)] = {"feats": len(feats), "trees": m.best_iteration, "sec": round(time.time() - t0, 1),
                          "model": m}
    print(f"{task} {name:<22} признаков {len(feats):>3} деревьев {m.best_iteration:>4} {time.time() - t0:.0f} c")


for task in TASKS:
    for name, feats in FEATS.items():
        fit(name, feats, task)

# %% [markdown]
# ## Метрики и прирост к U1
# Классификация — PR-AUC с парным блочным бутстрэпом по дням и permutation test (lk.paired_diff).
# Регрессия — WAPE входов (ŷ = exp(ẑ)·(ref + 10) − 10) с тем же блочным бутстрэпом и перестановками.

# %%
def y_hat(o, z):
    return np.clip(np.exp(z) * (o["ref"].to_numpy() + 10) - 10, 0, None)


def wape(y, yh):
    return float(np.abs(y - yh).sum() / y.sum())


def wape_diff(o, za, zb, n_boot=1000, n_perm=1000, seed=lk.SEED):
    """WAPE(A) − WAPE(B) (< 0 — A лучше), ДИ по дням, односторонний p для «A лучше»."""
    y = o["y"].to_numpy(); ya, yb = y_hat(o, za), y_hat(o, zb)
    days, inv = np.unique(o["day"].to_numpy(), return_inverse=True)
    ea = np.bincount(inv, np.abs(y - ya)); eb = np.bincount(inv, np.abs(y - yb)); sy = np.bincount(inv, y)
    est = ea.sum() / sy.sum() - eb.sum() / sy.sum()
    rng = np.random.default_rng(seed)
    W = np.zeros((n_boot, len(days)))
    np.add.at(W, (np.repeat(np.arange(n_boot), len(days)), rng.integers(0, len(days), n_boot * len(days))), 1)
    bs = (W @ ea - W @ eb) / (W @ sy)
    Sw = (rng.random((n_perm, len(days))) < 0.5)
    perm = ((np.where(Sw, eb, ea)).sum(1) - (np.where(Sw, ea, eb)).sum(1)) / sy.sum()
    return {"diff": float(est), "ci_low": float(np.percentile(bs, 2.5)), "ci_high": float(np.percentile(bs, 97.5)),
            "p_perm": float((np.sum(perm <= est) + 1) / (n_perm + 1))}


rows, gains = [], []
for p in ["val", "test", "holdout"]:
    o = O[p]
    for name in FEATS:
        pc, zr = PRED[("cls", name, p)], PRED[("reg", name, p)]
        rows.append({"model": name, "part": p, "pr_auc": lk.metrics(o["y_drop"], pc)["pr_auc"],
                     "auroc": lk.metrics(o["y_drop"], pc)["auroc"], "wape": wape(o["y"].to_numpy(), y_hat(o, zr)),
                     "mae_z": float(np.abs(o["z"] - zr).mean())})
        if p != "val" and name != "U1_num":
            g = lk.paired_diff(o["y_drop"], pc, PRED[("cls", "U1_num", p)], o["day"], n_boot=1000, n_perm=1000)
            gains.append({"model": name, "part": p, "task": "PR-AUC (больше — лучше)", **g})
            g = wape_diff(o, zr, PRED[("reg", "U1_num", p)])
            gains.append({"model": name, "part": p, "task": "WAPE (меньше — лучше)", **g})
res = pd.DataFrame(rows); gains = pd.DataFrame(gains)
res.to_csv(OUT / "metrics_ext.csv", index=False); gains.to_csv(OUT / "gain_ext.csv", index=False)
print(res[res["part"] == "test"].round(4).to_string(index=False))
print(gains[gains["part"] == "test"].round(4).to_string(index=False))

# %% [markdown]
# ## Срезы test: там, где источник «сработал»

# %%
o = O["test"]
SL = {"все": np.ones(len(o), bool),
      "начало провала (в T−1 провала нет)": (o["r_lag1"].fillna(1) >= lk.DROP_RATIO).to_numpy()}
if "R" in NEW:
    SL["дождь ≥ 1 мм/ч у станции"] = (o["rad_pix"] >= 1).to_numpy()
    SL["ливень ≥ 7,6 мм/ч у станции"] = (o["rad_max3"] >= 7.6).to_numpy()
if "W" in NEW:
    SL["действует предупреждение NWS"] = (o["nws_n_active"] > 0).to_numpy()
    SL["NWS: наводнение или гроза"] = ((o["nws_flood"] + o["nws_storm"]) > 0).to_numpy()
if "E" in NEW:
    SL["мероприятие у станции"] = (o["evt_direct"] > 0).to_numpy()
if "S" in NEW:
    SL["расписание < 80% обычного"] = (o["sch_ratio"] < 0.8).to_numpy()
# пересечения с началом провала: там числа прошлого часа ещё ничего не показывают
onset = SL["начало провала (в T−1 провала нет)"]
for k in [k for k in list(SL) if k not in ("все", "начало провала (в T−1 провала нет)")]:
    SL[f"{k} и начало провала"] = SL[k] & onset
models_sl = ["U1_num"] + [m for m in FEATS if m.startswith("+") or m == "ALL"]
srows = []
for nm, m in SL.items():
    g = o[m]
    if len(g) < 50:
        continue
    for mod in models_sl:
        pc, zr = PRED[("cls", mod, "test")][m], PRED[("reg", mod, "test")][m]
        srows.append({"срез": nm, "n": int(m.sum()), "провалов": int(g["y_drop"].sum()), "model": mod,
                      "pr_auc": lk.metrics(g["y_drop"], pc)["pr_auc"] if g["y_drop"].any() else np.nan,
                      "wape": wape(g["y"].to_numpy(), y_hat(g, zr))})
    if g["y_drop"].sum() >= 10:
        gg = lk.paired_diff(g["y_drop"], PRED[("cls", "ALL", "test")][m], PRED[("cls", "U1_num", "test")][m],
                            g["day"], n_boot=1000, n_perm=1000)
        gw = wape_diff(g, PRED[("reg", "ALL", "test")][m], PRED[("reg", "U1_num", "test")][m])
        srows.append({"срез": nm, "n": int(m.sum()), "провалов": int(g["y_drop"].sum()), "model": "ALL − U1",
                      "pr_auc": gg["diff"], "pr_ci": f"[{gg['ci_low']:+.4f}; {gg['ci_high']:+.4f}] p={gg['p_perm']:.3f}",
                      "wape": gw["diff"], "wape_ci": f"[{gw['ci_low']:+.4f}; {gw['ci_high']:+.4f}] p={gw['p_perm']:.3f}"})
slices = pd.DataFrame(srows)
slices.to_csv(OUT / "slices_ext.csv", index=False)
print(slices.pivot_table(index="срез", columns="model", values="pr_auc").round(3).to_string())
print(slices.pivot_table(index="срез", columns="model", values="wape").round(4).to_string())

# %% [markdown]
# ## Вклад групп в модели ALL (gain LightGBM) и итог

# %%
imp = []
for task in TASKS:
    m = INFO[(task, "ALL")]["model"]
    g = pd.Series(m.feature_importance("gain"), index=m.feature_name())
    tot = g.sum()
    imp.append({"task": task, "числа": g[NUM].sum() / tot,
                **{k: g[[c for c in v if c in g.index]].sum() / tot for k, v in NEW.items()}})
imp = pd.DataFrame(imp)
imp.to_csv(OUT / "importance_ext.csv", index=False)
print(imp.round(4).to_string(index=False))

lines = ["# Эксперимент 2: новые модальности — сводка", "",
         f"Источники: {', '.join(SRC) or 'нет'}; статус: {json.dumps(STATUS, ensure_ascii=False, default=str)}", "",
         "## Прирост к U1 на test и holdout", "",
         "| модель | часть | задача | разность | 95% ДИ | p |", "| --- | --- | --- | --- | --- | --- |"]
for _, r in gains.iterrows():
    lines.append(f"| {r['model']} | {r['part']} | {r['task']} | {r['diff']:+.4f} | "
                 f"[{r['ci_low']:+.4f}; {r['ci_high']:+.4f}] | {r['p_perm']:.3f} |")
c0 = gains[(gains["model"].isin(["C0_U1_plus_noise", "C1_U1_seed2"])) & (gains["part"] == "test")]
lines += ["", "Контроли (шум, другой seed) показывают масштаб случайной разницы: прирост меньше их — не эффект.",
          "", "## Вклад групп признаков в ALL (gain)", "", imp.round(4).to_markdown(index=False)]
(OUT / "summary_ext.md").write_text("\n".join(lines), encoding="utf-8")
json.dump({"status": STATUS, "models": list(FEATS), "groups": {k: len(v) for k, v in NEW.items()},
           "info": {f"{a}|{b}": {k: v for k, v in i.items() if k != "model"} for (a, b), i in INFO.items()}},
          open(OUT / "lab08_summary.json", "w"), ensure_ascii=False, indent=1, default=str)
print("\n".join(lines[:60]))
