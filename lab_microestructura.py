import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from pathlib import Path

# Configuración de página independiente
st.set_page_config(
    page_title="Laboratorio P2P Bolivia - Microestructura",
    page_icon="🔬",
    layout="wide"
)

@st.cache_data
def cargar_datos_laboratorio():
    posibles_rutas = [Path("data_clean"), Path("../data_clean")]
    ruta_base = next((r for r in posibles_rutas if r.exists()), Path("data_clean"))
    
    granger = pd.read_csv(ruta_base / "resultados_granger_stackelberg_bertrand.csv")
    adf = pd.read_csv(ruta_base / "resultados_estacionariedad_adf.csv")
    red_pares = pd.read_csv(ruta_base / "red_bertrand_pares_epoca2.csv")
    red_nodos = pd.read_csv(ruta_base / "red_bertrand_nodos_epoca2.csv")
    irf = pd.read_csv(ruta_base / "irf_var_epoca2.csv")
    fevd_buy = pd.read_csv(ruta_base / "fevd_multivariado_epoca2_buy.csv", index_col=0)
    fevd_sell = pd.read_csv(ruta_base / "fevd_multivariado_epoca2_sell.csv", index_col=0)
    micro = pd.read_csv(ruta_base / "base_consolidada_microestructura_paper.csv")
    
    return granger, adf, red_pares, red_nodos, irf, fevd_buy, fevd_sell, micro

def main():
    st.title("🔬 Laboratorio de Competencia y Microestructura del Mercado P2P – Bolivia")
    st.markdown(
        "**Investigación Empírica sobre Formación de Precios, Liderazgo Predictivo (Stackelberg) "
        "y Competencia Estratégica (Bertrand) en el Mercado USDT/BOB de Binance P2P.**"
    )
    st.divider()

    try:
        granger, adf, red_pares, red_nodos, irf, fevd_buy, fevd_sell, micro = cargar_datos_laboratorio()
    except Exception as e:
        st.error(f"Error al leer la carpeta data_clean. Verifica que los CSVs estén en el repositorio. Detalle: {e}")
        return

    # Controles superiores
    col_ctrl1, col_ctrl2 = st.columns(2)
    with col_ctrl1:
        epoca_sel = st.selectbox(
            "Periodo Operativo (Segmentación Metodológica)",
            options=["Epoca2", "Epoca1"],
            format_func=lambda x: "Época 2: Alta Frecuencia (May 2026 - Oct 2026)" if x == "Epoca2" else "Época 1: Baja Frecuencia (Sep 2025 - Abr 2026)"
        )
    with col_ctrl2:
        tipo_sel = st.selectbox(
            "Modalidad del Libro de Órdenes (Binance P2P)",
            options=["SELL", "BUY"],
            format_func=lambda x: f"{x} ({'Oferta de Venta / Compra Inmediata Taker' if x == 'SELL' else 'Oferta de Compra / Venta Inmediata Taker'})"
        )

    bloque_actual = f"{epoca_sel}_{tipo_sel}"
    df_bloque = granger[granger['Bloque'] == bloque_actual].sort_values('P_Stackelberg').reset_index(drop=True)
    adf_bloque = adf[adf['Bloque'] == bloque_actual]

    # Resumen rápido de Estacionariedad (ADF)
    with st.expander("📊 Diagnóstico Econométrico de Estacionariedad (Test Dickey-Fuller Aumentado - ADF)", expanded=False):
        c_adf1, c_adf2, c_adf3 = st.columns(3)
        c_adf1.metric("Vendedores Evaluados", f"{len(adf_bloque)}")
        c_adf2.metric("Estacionarias en Niveles I(0)", f"{int(adf_bloque['Estacionaria_Niveles'].sum())}")
        c_adf3.metric("Estacionarias en 1ras Diferencias ΔVWAP", f"{int(adf_bloque['Estacionaria_Diff'].sum())} (100%)")
        st.caption("La totalidad de las series integradas I(1) alcanza estacionariedad estricta en primeras diferencias (ΔP_t), evitando regresiones espurias.")

    # 1. SECCIÓN STACKELBERG
    st.markdown("### 🏆 Identificación del Líder de Mercado (Modelo de Stackelberg)")
    st.caption("Causalidad de Granger evaluada sobre variaciones diarias (ΔVWAP) respecto al Precio de Mercado Excluyente (Leave-One-Out).")

    tabla_stack = df_bloque[['Vendedor', 'Dias_Cambio_Precio', 'Lag_Stackelberg', 'P_Stackelberg', 'Es_Lider_Stackelberg', 'Beta_PassThrough', 'R2_Predictivo']].head(10).copy()
    tabla_stack.columns = ['Vendedor', 'Días Activos (ΔP≠0)', 'Mejor_Lag (Días)', 'P_Valor', 'Es_Lider', 'Sensibilidad (β)', 'Capacidad Predictiva (R²)']
    
    st.dataframe(
        tabla_stack.style.format({
            'P_Valor': '{:.7f}',
            'Sensibilidad (β)': '{:.3f}',
            'Capacidad Predictiva (R²)': '{:.1%}'
        }),
        use_container_width=True,
        hide_index=True
    )

    lideres_robustos = df_bloque[df_bloque['Dias_Cambio_Precio'] >= 30]
    lider_top = lideres_robustos.iloc[0] if not lideres_robustos.empty else df_bloque.iloc[0]

    st.success(f"**Líder Principal Identificado:** `{lider_top['Vendedor']}` (Ajustó precios en {int(lider_top['Dias_Cambio_Precio'])} días del periodo)")
    
    k1, k2, k3 = st.columns(3)
    k1.metric("Tiempo de Reacción", f"{int(lider_top['Lag_Stackelberg'])} Día(s)")
    k2.metric("Sensibilidad / Pass-Through (β)", f"{lider_top['Beta_PassThrough']:.3f}")
    k3.metric("Capacidad Predictiva (R²)", f"{lider_top['R2_Predictivo']*100:.1f}%")

    st.divider()

    # 2. SECCIÓN BERTRAND
    st.markdown("### 🛒 Dinámica de Mercado: Competencia en Precios (Modelo de Bertrand)")
    
    n_lideres_sig = int(df_bloque['Es_Lider_Stackelberg'].sum())
    n_reaccion_sig = int(df_bloque['Reacciona_al_Mercado'].sum())

    if epoca_sel == "Epoca2":
        nodos_actual = red_nodos[red_nodos['Bloque'] == bloque_actual].copy()
        st.markdown("#### 📋 Ranking de Causalidad Cruzada y Centralidad de Red (Top 10 Vendedores)")
        
        tabla_nodos = nodos_actual[['Vendedor', 'Influencia_Saliente_OutDegree', 'Sensibilidad_Entrante_InDegree', 'Indice_Liderazgo_Neto', 'Lag_Medio_Emision', 'P_Valor_Medio_Emision']].copy()
        tabla_nodos.columns = ['Vendedor', 'Out-Degree (Influye a N)', 'In-Degree (Reacciona a N)', 'Liderazgo Neto', 'Lag Medio (Días)', 'P-Valor Medio']
        
        st.dataframe(
            tabla_nodos.style.format({
                'Lag Medio (Días)': '{:.2f}',
                'P-Valor Medio': '{:.6f}'
            }),
            use_container_width=True,
            hide_index=True
        )
        
        top3_nombres = ", ".join(nodos_actual['Vendedor'].head(3).tolist())
        st.info(
            f"**Diagnóstico de Régimen (Modelo de Bertrand / Liderazgo Compartido):** Se detectaron **{n_lideres_sig} vendedores** con influencia predictiva significativa ($p < 0.05$) y **{n_reaccion_sig} vendedores** que reaccionan a los ajustes del mercado. "
            f"Esto confirma un **oligopolio competitivo estilo Bertrand**, donde la formación del tipo de cambio rota entre actores centrales ({top3_nombres})."
        )
    else:
        tabla_bert_e1 = df_bloque[['Vendedor', 'Dias_Cambio_Precio', 'Lag_Bertrand_Reaccion', 'P_Bertrand_Reaccion', 'Reacciona_al_Mercado']].sort_values('P_Bertrand_Reaccion').head(10).copy()
        tabla_bert_e1.columns = ['Vendedor', 'Días Activos (ΔP≠0)', 'Lag Reacción (Días)', 'P_Valor Reacción', 'Reacciona al Mercado']
        st.dataframe(tabla_bert_e1.style.format({'P_Valor Reacción': '{:.6f}'}), use_container_width=True, hide_index=True)
        st.info(f"**Diagnóstico Época 1:** Se identificaron **{n_lideres_sig} líderes predictivos** y **{n_reaccion_sig} seguidores reactivos** ($p < 0.05$).")

    # 3. PESTAÑAS DE MICROESTRUCTURA Y MODELO VAR
    if epoca_sel == "Epoca2":
        st.divider()
        tab1, tab2, tab3 = st.tabs([
            "📈 Impulso-Respuesta (VAR Pass-Through)", 
            "🕸️ Mapa de Calor y Varianza (FEVD)", 
            "🏛️ Determinantes de Microestructura"
        ])
        
        with tab1:
            st.markdown("#### Función de Impulso-Respuesta Acumulada (Choque de +0.10 BOB en el Líder)")
            top5_lista = lideres_robustos.head(5)['Vendedor'].tolist()
            sub_irf = irf[
                (irf['Bloque'] == bloque_actual) & 
                (irf['Vendedor'].isin(top5_lista))
            ].copy()

            fig_irf = px.line(
                sub_irf,
                x="Dia_Horizonte",
                y="Respuesta_Acumulada_BOB",
                color="Vendedor",
                markers=True,
                title=f"Respuesta Acumulada del Mercado Excluyente ({bloque_actual})",
                labels={
                    "Dia_Horizonte": "Días después del choque (Horizonte t + h)",
                    "Respuesta_Acumulada_BOB": "Respuesta Acumulada (BOB)",
                    "Vendedor": "Vendedor Líder"
                },
                hover_data={
                    "Dia_Horizonte": True,
                    "Respuesta_Acumulada_BOB": ":.4f",
                    "Respuesta_Diaria_BOB": ":.4f",
                    "Vendedor": True
                }
            )

            # Línea de referencia del choque inicial (+0.10 BOB) y línea cero
            fig_irf.add_hline(
                y=0.10,
                line_dash="dash",
                line_color="gray",
                annotation_text="Choque Inicial (+0.10 BOB)",
                annotation_position="top left"
            )
            fig_irf.add_hline(y=0.0, line_width=1, line_color="black")

            fig_irf.update_layout(
                xaxis=dict(dtick=1),
                hovermode="x unified",
                legend_title_text="Vendedor Líder",
                height=450
            )

            st.plotly_chart(fig_irf, use_container_width=True)

        with tab2:
            col_hm, col_fevd = st.columns(2)
            with col_hm:
                st.markdown("##### Red Bilateral de Granger (P-Valores)")
                sub_pares = red_pares[red_pares['Bloque'] == bloque_actual]
                matriz_hm = sub_pares.pivot(index='Vendedor_Emisor', columns='Vendedor_Receptor', values='P_Valor')
                fig_hm, ax_hm = plt.subplots(figsize=(6, 5))
                sns.heatmap(matriz_hm, annot=True, fmt=".2f", cmap="YlGnBu_r" if tipo_sel == "BUY" else "YlOrRd_r", vmin=0, vmax=0.10, ax=ax_hm)
                ax_hm.set_xlabel("Receptor (Seguidor)")
                ax_hm.set_ylabel("Emisor (Líder)")
                st.pyplot(fig_hm)
            with col_fevd:
                st.markdown("##### Descomposición de la Varianza (FEVD a 7 Días)")
                df_fevd_mostrar = fevd_buy if tipo_sel == "BUY" else fevd_sell
                st.dataframe(df_fevd_mostrar.style.format("{:.2f}%"), use_container_width=True)

        with tab3:
            st.markdown("#### Atributos Microeconómicos de los Vendedores Dominantes")
            sub_micro = micro[micro['Bloque'] == bloque_actual][
                ['Vendedor', 'Poder_Predictivo_LogP', 'Frecuencia_Ajuste_Pct', 'Anuncios_Por_Snapshot', 'Disponibilidad_Media_USDT', 'Limite_Min_Medio_BOB']
            ].sort_values('Poder_Predictivo_LogP', ascending=False).copy()
            sub_micro.columns = ['Vendedor', 'Poder Predictivo (-log10 P)', 'Frecuencia Repricing (%)', 'Anuncios / Snapshot', 'Liquidez Media (USDT)', 'Límite Mínimo (BOB)']
            st.dataframe(sub_micro.head(15).style.format({
                'Poder Predictivo (-log10 P)': '{:.2f}',
                'Frecuencia Repricing (%)': '{:.1f}%',
                'Anuncios / Snapshot': '{:.2f}',
                'Liquidez Media (USDT)': '{:,.0f}',
                'Límite Mínimo (BOB)': '{:,.0f}'
            }), use_container_width=True, hide_index=True)

if __name__ == "__main__":
    main()