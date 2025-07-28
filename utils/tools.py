import streamlit as st
from streamlit_autorefresh import st_autorefresh
from datetime import datetime, timedelta, timezone
import requests_cache, openmeteo_requests
from retry_requests import retry
from google.oauth2 import service_account
from google.cloud import bigquery
import pandas as pd
import numpy as np
import polars as pl
from pathlib import Path
from zoneinfo import ZoneInfo

credenciales_json = st.secrets["gcp_service_account"]
BIGQUERY_PROJECT_ID = st.secrets["bigquery"]["project_id"]
BIGQUERY_DATASET_ID = st.secrets["bigquery"]["dataset_id"]
TABLA = st.secrets["bigquery"]["table"]
TABLA_COMPLETA = f"{BIGQUERY_DATASET_ID}.{TABLA}"
CACHE_PATH = "cache/df_power_cache.parquet"

def quarter_autorefresh(key: str = "q", state_key: str = "first") -> None:
    """Refresca en el próximo cuarto de hora exacto y luego cada 15 min."""
    ms_to_q = lambda: ((15 - datetime.now().minute % 15) * 60 - datetime.now().second) * 1000 - datetime.now().microsecond // 1000
    first = st.session_state.setdefault(state_key, True)
    interval = ms_to_q() if first else 15 * 60 * 1000
    st.session_state[state_key] = False
    st_autorefresh(interval=interval, key=key)

def bigquery_auth():
    return service_account.Credentials.from_service_account_info(credenciales_json)

def _get_mapping():
    return {
        'PM_General_Potencia_Activa_Total': 'General',
        'PM_Aires_Potencia_Activa_Total': 'Aires Acondicionados',
        'Inversor_Solar_Potencia_Salida': 'SSFV',
    }

def _get_cache(CACHE_PATH, _POWER):
    if Path(CACHE_PATH).exists():
        df_cache = pl.read_parquet(CACHE_PATH)
        last_date = df_cache.select(pl.col("ds").max()).item()
        where_clause = f"WHERE id IN ({','.join(repr(i) for i in _POWER)}) AND datetime_record > TIMESTAMP('{last_date}')"
    else:
        df_cache = pl.DataFrame()
        where_clause = f"WHERE id IN ({','.join(repr(i) for i in _POWER)})"
    return df_cache, where_clause

def read_bq_db(credentials, fecha_ini=None, fecha_fin=None):
    fecha_fin = fecha_fin + timedelta(days=1)
    _POWER = ['PM_General_Potencia_Activa_Total', 'PM_Aires_Potencia_Activa_Total', 'Inversor_Solar_Potencia_Salida']
    _MAPPING = _get_mapping()
    client = bigquery.Client(project=BIGQUERY_PROJECT_ID, credentials=credentials)
    df_cache, where_clause = _get_cache(CACHE_PATH, _POWER)
    sql_query = f"SELECT * FROM `{TABLA_COMPLETA}` {where_clause} ORDER BY datetime_record ASC, id ASC"
    query_job = client.query(sql_query)
    results = query_job.result()
    data = [dict(row) for row in results]

    if not data:
        return df_cache  # Nada nuevo que agregar
    
    df = pl.DataFrame(data).rename(mapping=({'id': 'unique_id', 'datetime_record': 'ds', }))
    df = df.with_columns(pl.col("unique_id").replace(_MAPPING).alias("unique_id"), pl.col("ds").dt.truncate("15m").alias("ds"))
    df = df.filter((pl.col("ds") >= fecha_ini) & (pl.col("ds") <= fecha_fin))
    df_power = gen_others_load(df)
    df_power_pa = df_power.to_pandas()
    return df_power_pa

def gen_others_load(df):
    pivot = df.pivot(values="value",index=["ds", "company", "headquarters"],on="unique_id",aggregate_function="first")
    pivot = pivot.with_columns((pl.col("General") + pl.col("SSFV") - pl.col("Aires Acondicionados")).alias("Otros"))
    result = pivot.unpivot(index=["ds", "company", "headquarters"],on=["General", "Aires Acondicionados", "SSFV", "Otros"],variable_name="unique_id",value_name="value").sort(["ds", "unique_id"])
    result = result.with_columns((pl.col("value") * 0.25).round(2).alias("value"))
    return result

def get_climate_data(lat, lon):
    TZ = ZoneInfo("America/Bogota")
    # sesión con retry
    session = retry(
        requests_cache.CachedSession('.cache', expire_after=3600),
        retries=5, backoff_factor=0.2
    )()
    client = openmeteo_requests.Client(session=session)

    # ahora en Bogotá
    now_local = datetime.now(TZ)
    start_date = "2025-05-15"  # tu fecha fija
    end_date = now_local.strftime("%Y-%m-%d")

    # llamamos a la API
    response = client.weather_api(
        "https://api.open-meteo.com/v1/forecast",
        params={
            "latitude": lat,
            "longitude": lon,
            "models": "gfs_seamless",
            "minutely_15": ["temperature_2m", "relative_humidity_2m", "precipitation"],
            "start_date": start_date,
            "end_date": end_date
        }
    )[0].Minutely15()

    # convertimos los timestamps respetando TZ Bogota
    start_ts = datetime.fromtimestamp(response.Time(), TZ)
    end_ts   = datetime.fromtimestamp(response.TimeEnd(), TZ)
    st.write(start_ts, end_ts)

    # generamos el rango de timestamps
    interval = timedelta(seconds=response.Interval())
    count = int((end_ts - start_ts) / interval) + 1
    timestamps = [start_ts + i * interval for i in range(count)]

    # construimos el DataFrame en Polars
    df = pl.DataFrame({
        "ds": timestamps,
        "T2M":    response.Variables(0).ValuesAsNumpy(),
        "RH2M":   response.Variables(1).ValuesAsNumpy(),
        "PRECTOTCORR": response.Variables(2).ValuesAsNumpy()
    })

    # filtramos entre tu fecha fija (ajústala a TZ si lo deseas)
    start_filter = datetime(2025, 5, 15, 0, 0, tzinfo=TZ)
    df = df.filter(
        (pl.col("ds") >= start_filter) &
        (pl.col("ds") <= now_local)
    )

    df_pandas = df.to_pandas()
    st.write(df_pandas)
    return df_pandas
   
# Función para obtener las métricas
def get_metrics(general, ac, ssfv, otros):
    return {
        "General": {"energia": f"{general:.1f}"},
        "AC": {"energia": f"{ac:.1f}"},
        "SSFV": {"energia": f"{-ssfv:.1f}"},
        "Otros": {"energia": f"{otros:.1f}"},
    }

def get_percentages(ac, ssfv, otros):
    general = ac + ssfv + otros
    return {
        "AC": round((ac / general) * 100, 2),
        "SSFV": round((ssfv / general) * 100, 2),
        "Otros": round((otros / general) * 100, 2),
    }
    
def get_peak_load(df):
    peak_row = df.loc[df['value'].idxmax()]
    return {'valor': peak_row['value']}

def get_prom_load(df):
    if df.empty:
        return 0
    prom = round(df['value'].mean(), 2)
    return {'valor': prom}

def get_submedidores(metrics):
    return [k for k in metrics if k != "General"]

def load_custom_css(file_path: str = "styles/style.css"):
    with open(file_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
