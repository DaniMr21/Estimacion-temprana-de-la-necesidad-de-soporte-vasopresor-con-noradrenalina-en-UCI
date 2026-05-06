"""
Investigación de dos hallazgos contraintuitivos — v4

1. LACTATO: no significativo estadísticamente pese a ser el marcador
   de hipoperfusión por excelencia. Hipótesis: la lactacidemia en
   pacientes sépticos podría estar "diluyendo" su señal predictiva
   al crear una distribución bimodal o confundir con otros mecanismos.

2. GPT_MAX: efecto inverso (OR<1, protector). Ya vimos el análisis
   de cuartiles global. Aquí investigamos si el efecto varía entre
   sépticos y no sépticos.
"""

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from scipy.stats import mannwhitneyu, kruskal

RUTA_CSV = r'C:\Users\danie\OneDrive\Escritorio\DATA\definitivo_v4.csv'
ETIQUETA = 'etiqueta_norad_6_24'

df = pd.read_csv(RUTA_CSV)
df = df.dropna(subset=['pf_max'])

# Filtrar a 1ª estancia por paciente (independencia estadística)
df = (df.sort_values(['subject_id', 'contador_estancia_uci'])
        .drop_duplicates('subject_id', keep='first')
        .reset_index(drop=True))

print(f"Dataset (1ª estancia/paciente): N={len(df)}")
print(f"Positivos: {df[ETIQUETA].sum()} ({100*df[ETIQUETA].mean():.2f}%)")
print(f"Sépticos : {(df['tiene_sepsis']==1).sum()} "
      f"({100*(df['tiene_sepsis']==1).mean():.1f}%)")
print(f"No sépticos: {(df['tiene_sepsis']==0).sum()} "
      f"({100*(df['tiene_sepsis']==0).mean():.1f}%)")


# ═══════════════════════════════════════════════════════════════════════════════
# 1. INVESTIGACIÓN DEL LACTATO
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*65)
print("1. LACTATO — POR QUÉ NO ES SIGNIFICATIVO")
print("="*65)

# 1a. Distribución global
pos = df.loc[df[ETIQUETA]==1, 'lactato_max'].dropna()
neg = df.loc[df[ETIQUETA]==0, 'lactato_max'].dropna()
stat, p = mannwhitneyu(pos, neg, alternative='two-sided')

print(f"\n[1a] Mann-Whitney global:")
print(f"  Grupo positivo (norad 6-24h): mediana={pos.median():.2f}, "
      f"IQR=[{pos.quantile(0.25):.2f}, {pos.quantile(0.75):.2f}]")
print(f"  Grupo negativo              : mediana={neg.median():.2f}, "
      f"IQR=[{neg.quantile(0.25):.2f}, {neg.quantile(0.75):.2f}]")
print(f"  p-valor = {p:.4f} {'← significativo' if p<0.05 else '← NO significativo'}")

# 1b. Estratificado por sepsis
print(f"\n[1b] Estratificado por sepsis:")
for sep, nombre in [(1,'SÉPTICOS'), (0,'NO SÉPTICOS')]:
    sub = df[df['tiene_sepsis'] == sep]
    pos_s = sub.loc[sub[ETIQUETA]==1, 'lactato_max'].dropna()
    neg_s = sub.loc[sub[ETIQUETA]==0, 'lactato_max'].dropna()
    if len(pos_s) >= 2 and len(neg_s) >= 2:
        _, p_s = mannwhitneyu(pos_s, neg_s, alternative='two-sided')
        print(f"\n  {nombre} (N={len(sub)}, pos={sub[ETIQUETA].sum()}):")
        print(f"    Positivos: mediana={pos_s.median():.2f}, "
              f"IQR=[{pos_s.quantile(0.25):.2f}, {pos_s.quantile(0.75):.2f}]")
        print(f"    Negativos: mediana={neg_s.median():.2f}, "
              f"IQR=[{neg_s.quantile(0.25):.2f}, {neg_s.quantile(0.75):.2f}]")
        print(f"    p-valor = {p_s:.4f} "
              f"{'← significativo' if p_s<0.05 else '← NO significativo'}")

# 1c. Prevalencia de la etiqueta por cuartiles de lactato
print(f"\n[1c] Prevalencia etiqueta por cuartil de lactato:")
df['lactato_cuartil'] = pd.qcut(df['lactato_max'], q=4,
                                 labels=['Q1\n(bajo)','Q2','Q3','Q4\n(alto)'])
tabla_lac = df.groupby('lactato_cuartil', observed=True)[ETIQUETA].agg(['mean','count'])
tabla_lac.columns = ['prevalencia','n']
tabla_lac['prevalencia_pct'] = (tabla_lac['prevalencia']*100).round(2)
print(tabla_lac[['n','prevalencia_pct']].to_string())

# 1d. Lo mismo estratificado por sepsis
print(f"\n[1d] Prevalencia por cuartil de lactato, estratificado por sepsis:")
for sep, nombre in [(1,'SÉPTICOS'), (0,'NO SÉPTICOS')]:
    sub = df[df['tiene_sepsis'] == sep].copy()
    sub['lactato_cuartil'] = pd.qcut(sub['lactato_max'], q=4,
                                      labels=['Q1','Q2','Q3','Q4'])
    t = sub.groupby('lactato_cuartil', observed=True)[ETIQUETA].agg(['mean','count'])
    t['pct'] = (t['mean']*100).round(2)
    print(f"\n  {nombre}:")
    print(t[['count','pct']].to_string())

# 1e. Distribución del lactato en sépticos vs no sépticos
print(f"\n[1e] Distribución del lactato: sépticos vs no sépticos")
sep_lac  = df.loc[df['tiene_sepsis']==1, 'lactato_max'].dropna()
nsep_lac = df.loc[df['tiene_sepsis']==0, 'lactato_max'].dropna()
_, p_lac_sep = mannwhitneyu(sep_lac, nsep_lac, alternative='two-sided')
print(f"  Sépticos    : mediana={sep_lac.median():.2f}, "
      f"IQR=[{sep_lac.quantile(0.25):.2f}, {sep_lac.quantile(0.75):.2f}]")
print(f"  No sépticos : mediana={nsep_lac.median():.2f}, "
      f"IQR=[{nsep_lac.quantile(0.25):.2f}, {nsep_lac.quantile(0.75):.2f}]")
print(f"  p-valor sépticos vs no sépticos = {p_lac_sep:.4f}")
print(f"\n  → Si los sépticos tienen lactato más alto en general,")
print(f"    su señal predictiva se 'aplana' porque tanto los que")
print(f"    reciben norad como los que no tienen lactato elevado.")

# 1f. Lactato en norad_0_6 (los que ya tenían norad antes de la ventana)
print(f"\n[1f] Lactato en pacientes con norad ANTES de la ventana (0-6h):")
cols_norad = [c for c in df.columns if 'norad' in c.lower()]
print(f"  Columnas disponibles: {cols_norad}")


# ═══════════════════════════════════════════════════════════════════════════════
# 2. INVESTIGACIÓN DE GPT_MAX
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*65)
print("2. GPT_MAX — EFECTO INVERSO: INVESTIGACIÓN DETALLADA")
print("="*65)

# 2a. Cuartiles de GPT por subgrupo de sepsis
print(f"\n[2a] Prevalencia etiqueta por cuartil de GPT, global y por sepsis:")
df['gpt_cuartil'] = pd.qcut(df['gpt_max'], q=4, labels=['Q1','Q2','Q3','Q4'])

print(f"\n  GLOBAL:")
t_global = df.groupby('gpt_cuartil', observed=True)[ETIQUETA].agg(['mean','count'])
t_global['pct'] = (t_global['mean']*100).round(2)
print(t_global[['count','pct']].to_string())

for sep, nombre in [(1,'SÉPTICOS'), (0,'NO SÉPTICOS')]:
    sub = df[df['tiene_sepsis'] == sep].copy()
    sub['gpt_cuartil'] = pd.qcut(sub['gpt_max'], q=4, labels=['Q1','Q2','Q3','Q4'])
    t = sub.groupby('gpt_cuartil', observed=True)[ETIQUETA].agg(['mean','count'])
    t['pct'] = (t['mean']*100).round(2)
    print(f"\n  {nombre}:")
    print(t[['count','pct']].to_string())

# 2b. Horas hasta norad por cuartil de GPT
print(f"\n[2b] Horas hasta norad por cuartil de GPT "
      f"(solo positivos con norad en ventana):")
positivos = df[df[ETIQUETA] == 1].copy()
positivos['gpt_cuartil'] = pd.qcut(positivos['gpt_max'], q=4,
                                    labels=['Q1','Q2','Q3','Q4'])
t_horas = positivos.groupby('gpt_cuartil',
                             observed=True)['horas_hasta_norad'].describe()
print(t_horas[['count','mean','50%','25%','75%']].round(2).to_string())

# 2c. Correlación GPT con otras variables hepáticas
print(f"\n[2c] Correlación Spearman entre GPT y otras variables:")
vars_correlacion = ['gpt_max', 'bilirrubina_media', 'sofa_max',
                    'lactato_max', 'map_min', 'ph_min']
vars_disponibles = [v for v in vars_correlacion if v in df.columns]
corr = df[vars_disponibles].corr(method='spearman')['gpt_max'].drop('gpt_max')
print(corr.round(3).to_string())

# 2d. GPT en sépticos vs no sépticos
print(f"\n[2d] GPT en sépticos vs no sépticos:")
sep_gpt  = df.loc[df['tiene_sepsis']==1, 'gpt_max'].dropna()
nsep_gpt = df.loc[df['tiene_sepsis']==0, 'gpt_max'].dropna()
_, p_gpt = mannwhitneyu(sep_gpt, nsep_gpt, alternative='two-sided')
print(f"  Sépticos    : mediana={sep_gpt.median():.2f}, "
      f"IQR=[{sep_gpt.quantile(0.25):.2f}, {sep_gpt.quantile(0.75):.2f}]")
print(f"  No sépticos : mediana={nsep_gpt.median():.2f}, "
      f"IQR=[{nsep_gpt.quantile(0.25):.2f}, {nsep_gpt.quantile(0.75):.2f}]")
print(f"  p-valor = {p_gpt:.4f}")

# 2e. Prevalencia norad_6_24 en pacientes con GPT muy alta (Q4)
#     pero diferenciando si ya tenían norad antes o no
print(f"\n[2e] Resumen del efecto inverso de GPT_max:")
q4_mask = df['gpt_cuartil'] == 'Q4'
q1_mask = df['gpt_cuartil'] == 'Q1'
print(f"  Prevalencia etiqueta en Q1 (GPT baja): "
      f"{df.loc[q1_mask, ETIQUETA].mean()*100:.2f}%")
print(f"  Prevalencia etiqueta en Q4 (GPT alta): "
      f"{df.loc[q4_mask, ETIQUETA].mean()*100:.2f}%")
print(f"\n  Sépticos Q4:")
mask_sep_q4 = (df['tiene_sepsis']==1) & q4_mask
print(f"    N={mask_sep_q4.sum()}, "
      f"prev={df.loc[mask_sep_q4, ETIQUETA].mean()*100:.2f}%")
print(f"\n  No sépticos Q4:")
mask_nsep_q4 = (df['tiene_sepsis']==0) & q4_mask
print(f"    N={mask_nsep_q4.sum()}, "
      f"prev={df.loc[mask_nsep_q4, ETIQUETA].mean()*100:.2f}%")

print("\n" + "="*65)
print("FIN DEL ANÁLISIS")
print("="*65)
