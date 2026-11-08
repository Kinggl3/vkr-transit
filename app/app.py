"""
Прототип системы прогнозирования пассажиропотока метро Нью-Йорка.

  streamlit run app/app.py

Данные — app/data (готовит src/app_data.py) и таблицы results/.
"""

import streamlit as st

import views

st.set_page_config(page_title="Прогноз пассажиропотока метро", page_icon="🚇", layout="wide")
st.markdown(views.CSS, unsafe_allow_html=True)

pages = [
    st.Page(views.overview, title="Обзор", icon=":material/dashboard:", default=True),
    st.Page(views.city_map, title="Карта", icon=":material/map:", url_path="map"),
    st.Page(views.station, title="Станция", icon=":material/train:", url_path="station"),
    st.Page(views.replay, title="Повтор дня", icon=":material/replay:", url_path="replay"),
    st.Page(views.detector, title="Детектор сбоев", icon=":material/warning:", url_path="detector"),
    st.Page(views.what_if, title="Что если", icon=":material/tune:", url_path="what-if"),
    st.Page(views.quality, title="Качество моделей", icon=":material/insights:", url_path="quality"),
]
st.navigation(pages).run()
