import streamlit as st
import base64
from datetime import datetime, timedelta
import plotly.graph_objects as go
import numpy as np
import pandas as pd
from utils import tools

COLOR_MAP = {
    'SSFV': "orange",
    'AC': "lightblue",
    'Otros': "gray",
    'General': "black"
}

def mostrar_imagen(path, width=150):
    with open(path, "rb") as f:
        img64 = base64.b64encode(f.read()).decode()
    st.markdown(f'<div style="text-align:center;"><img src="data:image/png;base64,{img64}" width="{width}"></div>', unsafe_allow_html=True)

def graficar_consumo(df,pron,sub,fecha_ini=None, fecha_fin=None, altura=210):
    if fecha_ini is not None and fecha_fin is not None:
        fecha_ini_dt, fecha_fin_dt = pd.to_datetime(fecha_ini), pd.to_datetime(fecha_fin)
        df = df[(df["ds"] >= fecha_ini_dt) & (df["ds"] <= fecha_fin_dt)]
        if not sub:
            pron = pron[(pron.index >= fecha_ini_dt) & (pron.index <= fecha_fin_dt)]
        else:
            # Si pron es una Serie, asegurar que su índice es comparable con df["ds"]
            if hasattr(pron, "index") and hasattr(df, "ds"):
                # Alinear ambos por fecha
                mask = df["ds"] >= fecha_ini_dt
                df = df[mask]
                if isinstance(pron.index, pd.DatetimeIndex):
                    pron = pron[pron.index >= fecha_ini_dt]
                elif hasattr(pron, "index") and isinstance(pron.index, pd.Index):
                    # Si el índice no es fecha, simplemente recorta para igualar longitud
                    pron = pron[-len(df):]
            else:
                pron = pron
    fig = go.Figure()
    if not sub:
        solar = np.minimum(pron["SSFV"].values, df["value"].values)
        otros = np.minimum(pron["Otros"].values, np.maximum(df["value"].values - solar, 0))
        aires = np.maximum(df["value"].values - otros - solar, 0)  # Resta con aires, no con otros

        fig.add_trace(go.Bar(x=df["ds"], y=solar, name="SSFV", marker_color=COLOR_MAP['SSFV']))
        fig.add_trace(go.Bar(x=df["ds"], y=otros, name="Otros", marker_color=COLOR_MAP['Otros']))
        fig.add_trace(go.Bar(x=df["ds"], y=aires, name="Aires Acondicionados", marker_color=COLOR_MAP['AC']))
        fig.add_trace(go.Scatter(x=df["ds"], y=np.round(df["value"],1), mode="lines", name='General', line=dict(color=COLOR_MAP['General'])))
    else:
        fig.add_trace(go.Scatter(x=df["ds"], y=df["value"], mode="lines",name='Real'))
        fig.add_trace(go.Scatter(x=df["ds"], y=pron, mode="lines",name='Pronosticado'))
    
    fig.update_layout(title="", margin=dict(t=40, b=0), barmode="relative",font=dict(family="Poppins", color="black"),
                      xaxis=dict(domain=[0.05, 0.95], title="Fecha", showline=True, linecolor='black', showgrid=False, zeroline=False, title_font=dict(color='black'),tickfont=dict(color='black')),
                      yaxis=dict(title="Consumo (kWh)", title_font=dict(color='black'), tickfont=dict(color='black')),
                      legend=dict(orientation="h", x=0.8, xanchor="right", y=1.1, yanchor="top", font=dict(color="black")), height=altura)
    st.plotly_chart(fig, use_container_width=True)

def display_general(icons, metrics, db, pron, fecha_ini, fecha_fin, config_perc, config_hist):
    total = db.loc[db["unique_id"] == 'General', 'value'].abs().sum()
    with st.container(border=False, key='nilm-gen'):
        colg,colh = st.columns([1, 2], vertical_alignment='center')
        with colg:
            st.markdown("""<div style='text-align: center; margin-bottom:-60px;'><h6 style='margin-bottom: 0;'>Consumo Actual</h6></div>""", unsafe_allow_html=True)
            ca,cb = st.columns([4, 6], vertical_alignment='center')
            with ca:
                mostrar_imagen(icons['General'], 500)
            with cb:
                render_custom_metric(cb, "Medición General", metrics['General']['energia']+" kWh","")

            get_percentages = tools.get_percentages(pron['Aires Acondicionados'].sum(), pron['SSFV'].sum(), pron['Otros'].sum())
            if config_perc:
                st.markdown("<div style='text-align: center; margin-bottom:-30px;'><h6 style='margin-bottom: 0;'>Distribución Consumo Energético</h6></div>", unsafe_allow_html=True)
                display_participacion(get_percentages)
        with colh:
            df = db.loc[db["unique_id"] == 'General',['ds','value']]
            graficar_consumo(df,pron,False,fecha_ini,fecha_fin,360)
        if config_hist:    
            coli,colj = st.columns([1, 1], vertical_alignment='center')
            with coli:
                display_peak_load(df)
            with colj:
                display_prom_load(df)

def display_participacion(percentages):
    colors = [COLOR_MAP.get(label, '#CCCCCC') for label in percentages.keys()]
    participacion = go.Figure(go.Pie(labels=list(percentages.keys()), values=list(percentages.values()), marker=dict(colors=colors),
                                     hole=0.5, textinfo='label+percent', insidetextorientation='horizontal', showlegend=False, textposition='outside'))
    participacion.update_layout(margin=dict(l=20, r=20, t=20, b=20),height=160, font=dict(family="Poppins", size=14, color="#000"),)
    st.plotly_chart(participacion, use_container_width=True)

def display_peak_load(df):
    peak = tools.get_peak_load(df)
    render_custom_metric(st, "Consumo Máximo", f"{peak['valor']} kWh" ,sym="")

def display_prom_load(df):
    prom = tools.get_prom_load(df)
    render_custom_metric(st, "Consumo Promedio", f"{prom['valor']} kWh" ,sym="")

def display_submedidores(submedidores, nombres_submedidores, icons, metrics, db, pron, fecha_ini, fecha_fin):
    cols = st.columns(len(submedidores))
    for i, label in enumerate(submedidores):
        nombre = nombres_submedidores.get(label, label)
        with cols[i]:
            with st.container(border=False, key=f'nilm-subm-{i}'):
                ca,cb = st.columns([1, 3], vertical_alignment='center')
                with ca:
                    mostrar_imagen(icons[label], 200)
                with cb:
                    porc = (pron.iloc[-1,i]/sum(pron.iloc[-1,:]))*100
                    color = 'green' if float(metrics[label]['energia']) < 0 else 'red' if float(metrics[label]['energia']) > 0 else '#6c757d'
                    render_custom_metric(cb, nombre, f"{metrics[label]['energia']} kWh", f"{porc:.1f}%", color)
                key_exp =f"exp_{i}"
                with st.container(key=key_exp):
                    with st.expander("Ver Detalle", expanded=False):
                        df = db.loc[db["unique_id"] == nombre,['ds','value']]
                        graficar_consumo(df, pron[nombre], True, fecha_ini, fecha_fin)

def get_icons():
    return {
        "General": "images/MedidorGen.png",
        "AC": "images/MedidorAA.png",
        "SSFV": "images/MedidorPV.png",
        "Otros": "images/MedidorOtros.png"
    }

def render_custom_metric(col, label, value, delta=None,color='#6c757d',sym=""):
    html = f"""<div class="custom-metric" style="margin-bottom: 0;"><div class="label">{label}</div><div class="value">{value}</div>"""
    if delta:
        delta = f"{sym+delta}"
        html += f"""<div class="delta" style="color:{color};">{delta}</div>"""
    html += "</div>"
    col.markdown(html, unsafe_allow_html=True)


def render_NILM_tabs(submedidores, nombres_submedidores, icons, metrics, db_pow, Y_hat_df2, fecha_ini, fecha_fin, config_perc, config_hist):
    with st.container(key="styled_tabs"):
        tab1, tab2 = st.tabs(["General", "Submedición"])
        with tab1:
            display_general(icons, metrics, db_pow, Y_hat_df2, fecha_ini, fecha_fin, config_perc, config_hist)
        with tab2:
            if submedidores:
                st.markdown("<br>",unsafe_allow_html=True)
                display_submedidores(submedidores, nombres_submedidores, icons, metrics, db_pow, Y_hat_df2, fecha_ini, fecha_fin)
            else:
                st.warning("No hay submedición disponible")
            
