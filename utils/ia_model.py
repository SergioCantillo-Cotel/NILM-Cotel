from keras.models import load_model
import pandas as pd
import numpy as np

def get_IA_model():
    IA_model = load_model('IA/NILM_Model_best.keras')
    return IA_model

def datos_Exog(db, datos):
    fut = db[db["unique_id"] == 'General'][['ds', 'value']].rename(columns={'value': 'Energia_kWh_General'})
    fut['ds'] = pd.to_datetime(fut['ds'])
    fut['DOW'] = fut['ds'].dt.dayofweek + 1
    fut['Hour'] = fut['ds'].dt.hour
    fut = fut.merge(datos.drop(columns='PRECTOTCORR', errors='ignore'), on='ds', how='left')
    return fut[['ds','Energia_kWh_General','DOW','Hour','T2M','RH2M']].sort_values(['ds'])

def reconcile(exog, pron):
    r = np.copy(pron)
    lectura = exog['Energia_kWh_General'].values

    for i in range(len(r)):
        dow = exog['DOW'][i]
        hour = exog['Hour'][i]

        # Condiciones horarias
        wknd = dow in (6, 7)
        work = 8 <= hour <= 16

        # Aplicar reglas personalizadas
        if not wknd and work and dow <= 3:
            r[i, 1] += 0.5
            r[i, 2] += 0.25
        elif not wknd and work and dow > 3:
            r[i, 1] += 0.65
            r[i, 2] += 0.1
        elif wknd and not work:
            r[i, 2] -= 0.05
        elif wknd and work:
            r[i, 2] += 0.05

        # Forzar no-negatividad antes de reconciliar
        r[i] = np.clip(r[i], 0, None)

    return r
