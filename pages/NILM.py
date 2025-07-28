import streamlit as st
from utils import tools, viz, ia_model
import pandas as pd
from datetime import datetime

tools.load_custom_css()
tools.quarter_autorefresh()
credentials = tools.bigquery_auth()

st.sidebar.markdown("##### Parámetros de Visualización")

rango = st.sidebar.date_input("Periodo", (pd.Timestamp.now() - pd.Timedelta(hours=5) - pd.Timedelta(weeks=1), pd.Timestamp.now() - pd.Timedelta(hours=5)), min_value='2025-05-15', max_value=pd.Timestamp.now() - pd.Timedelta(hours=5), key='rango_fecha_NILM')   
if isinstance(rango, tuple) and len(rango) == 2:
    fecha_ini, fecha_fin = rango
else:
    st.toast("Esperando selección de un rango válido de fechas.", icon="⏳")
    fecha_ini, fecha_fin = pd.Timestamp.now() - pd.Timedelta(weeks=1) - pd.Timedelta(hours=5), pd.Timestamp.now() - pd.Timedelta(hours=5)

config_perc = st.sidebar.checkbox("Mostrar distribución de consumo energético en periodo de visualización", value=False, key='config_perc_NILM')
config_hist = st.sidebar.checkbox("Mostrar maximos y promedios de consumo energético en periodo de visualización", value=False, key='config_hist_NILM')

nombres_submedidores = {"AC": "Aires Acondicionados","SSFV": "SSFV","otros": "Otras Cargas"}
db_pow = tools.read_bq_db(credentials, fecha_ini, fecha_fin)
lat, lon = 3.4793949016367822, -76.52284557701176
datos = tools.get_climate_data(lat, lon)
try:
    modelo_IA = ia_model.get_IA_model()
    caracteristicas = ia_model.datos_Exog(db_pow, datos).drop(columns=['ds'])
    car2 = caracteristicas.copy()
    Y_hat_raw = modelo_IA.predict(caracteristicas.values.reshape(-1, 1, caracteristicas.shape[1]))
    Y_hat_rec = ia_model.reconcile(car2,Y_hat_raw)
    Y_hat_df2 = pd.DataFrame(Y_hat_rec, columns=['Aires Acondicionados','SSFV','Otros'])
    Y_hat_df2.index = db_pow.loc[db_pow["unique_id"] == 'General', "ds"].reset_index(drop=True)

    metrics = tools.get_metrics(
        db_pow.loc[db_pow["unique_id"] == 'General', "value"].iloc[-1],
        Y_hat_df2['Aires Acondicionados'].iloc[-1],
        Y_hat_df2['SSFV'].iloc[-1],Y_hat_df2['Otros'].iloc[-1]
    )

    submedidores = tools.get_submedidores(metrics)

except (IndexError, AttributeError, TypeError) as e:
    st.error(f"Error al calcular métricas: Intente nuevamente", icon="⚠️")
    st.stop()
viz.render_NILM_tabs(submedidores, nombres_submedidores, viz.get_icons(), metrics, db_pow, Y_hat_df2, fecha_ini, fecha_fin, config_perc, config_hist)
