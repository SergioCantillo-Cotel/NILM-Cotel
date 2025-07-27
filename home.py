import streamlit as st
from datetime import datetime
from utils import tools
import pandas as pd
import pytz

st.set_page_config(page_title="Eficiencia Energética + IA Cotel", layout="wide",initial_sidebar_state="collapsed")
tools.load_custom_css()
st.logo("images/cotel-logotipo.png", size="Large")

pages = [st.Page("pages/NILM.py", title="", icon="🏛️"),]
pg = st.navigation({"⚡ Medición Inteligente no Intrusiva": [pages[0]]}, position="top")
pg.run()

zona = pytz.timezone("America/Bogota")
ahora = pd.Timestamp(datetime.now(zona)).floor('15min').strftime("%Y-%m-%d %H:%M:%S")
st.markdown(f"""<div class="footer">🔄 Esta página se actualiza cada 15 minutos. Última actualización: {ahora}</div>""", unsafe_allow_html=True)