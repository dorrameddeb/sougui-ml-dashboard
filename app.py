# ============================================================
# PAGE : PRÉVISIONS (DYNAMIQUES AVEC TES DONNÉES)
# ============================================================
elif "Prévisions" in page:
    hero("Prévisions des Ventes B2C", "Modèles ARIMA · Prophet — Horizon 3 mois (basé sur vos données réelles)")

    # ========== 1. CHARGEMENT DES DONNÉES RÉELLES ==========
    try:
        # Charger les ventes
        fact_vente = pd.read_csv('Fact_Vente (1).csv')
        dim_date = pd.read_csv('Dim_Date (1).csv')
        
        # Nettoyage
        fact_vente['montant_ht'] = pd.to_numeric(fact_vente['montant_ht'], errors='coerce')
        fact_vente = fact_vente[fact_vente['montant_ht'] > 0]
        
        # Fusion avec les dates
        dim_date['date_key'] = dim_date['date_key'].astype(int)
        fact_vente = fact_vente.merge(dim_date[['date_key', 'date_complete']], on='date_key', how='left')
        fact_vente['date'] = pd.to_datetime(fact_vente['date_complete'])
        
        # Filtrer B2C
        dim_client = pd.read_csv('Dim_Client (1).csv')
        dim_client.columns = dim_client.columns.str.lower()
        fact_vente = fact_vente.merge(dim_client[['client_key', 'type_client']], on='client_key', how='left')
        fact_vente_b2c = fact_vente[fact_vente['type_client'] == 'B2C']
        
        # Agrégation mensuelle
        monthly_sales = fact_vente_b2c.groupby(fact_vente_b2c['date'].dt.to_period('M'))['montant_ht'].sum().reset_index()
        monthly_sales['date'] = monthly_sales['date'].dt.to_timestamp()
        monthly_sales.columns = ['ds', 'y']
        
        historique_dates = monthly_sales['ds'].tolist()
        historique_values = monthly_sales['y'].tolist()
        
        # ========== 2. MODÈLE PROPHET ==========
        from prophet import Prophet
        
        model = Prophet(yearly_seasonality=True, weekly_seasonality=False)
        model.add_seasonality(name='monthly', period=30.5, fourier_order=5)
        model.fit(monthly_sales)
        
        # Prévisions 3 mois
        future = model.make_future_dataframe(periods=3, freq='M')
        forecast = model.predict(future)
        
        # Extraire les prévisions
        previsions_dates = forecast['ds'].tail(3).tolist()
        previsions_values = forecast['yhat'].tail(3).tolist()
        previsions_lower = forecast['yhat_lower'].tail(3).tolist()
        previsions_upper = forecast['yhat_upper'].tail(3).tolist()
        
        # Métriques
        from sklearn.metrics import mean_absolute_error, mean_squared_error
        historical_pred = model.predict(monthly_sales[['ds']])
        mae = mean_absolute_error(monthly_sales['y'], historical_pred['yhat'])
        rmse = np.sqrt(mean_squared_error(monthly_sales['y'], historical_pred['yhat']))
        mape = np.mean(np.abs((monthly_sales['y'] - historical_pred['yhat']) / monthly_sales['y'])) * 100
        
        st.success(f"✅ Données chargées : {len(monthly_sales)} mois de ventes B2C")
        
    except Exception as e:
        st.warning(f"⚠️ Données non disponibles : {e}")
        st.info("📌 Utilisation de données simulées pour la démonstration")
        
        # Fallback : données simulées
        historique_dates = pd.date_range('2024-01-01', '2025-03-01', freq='MS')
        historique_values = [2100, 2300, 2500, 2800, 3000, 3500, 3800, 3600, 3300, 2900, 2600, 2400, 2500, 2700, 2900]
        previsions_dates = pd.date_range('2025-04-01', '2025-06-01', freq='MS')
        previsions_values = [3100, 3300, 3500]
        previsions_lower = [2900, 3100, 3300]
        previsions_upper = [3300, 3500, 3700]
        mae, rmse, mape = 98.7, 145.2, 12.3

    # ========== 3. AFFICHAGE DES MÉTRIQUES ==========
    col1, col2, col3 = st.columns(3)
    col1.metric("MAE", f"{mae:.1f} TND")
    col2.metric("RMSE", f"{rmse:.1f} TND")
    col3.metric("MAPE", f"{mape:.1f} %")

    divider()
    section("Séries temporelles & Prévisions — 3 mois")

    # ========== 4. GRAPHIQUE DYNAMIQUE ==========
    fig, ax = plt.subplots(figsize=(13, 5))
    fig.patch.set_facecolor(OFF_W)
    
    # Tracer l'historique
    ax.plot(historique_dates, historique_values, 'o-', color=NAVY, lw=2.5, ms=5, label='Historique réel')
    
    # Tracer les prévisions
    ax.plot(previsions_dates, previsions_values, 's--', color=GOLD, lw=2.5, ms=8, label='Prévisions Prophet')
    
    # Intervalle de confiance
    ax.fill_between(previsions_dates, previsions_lower, previsions_upper,
                    alpha=0.18, color=GOLD, label='Intervalle 95%')
    
    # Ligne de séparation
    if len(historique_dates) > 0:
        ax.axvline(historique_dates[-1], color=GRAY_M, linestyle=':', lw=1.5)
        ax.text(historique_dates[-1] + pd.Timedelta(days=5), max(historique_values)*0.96, 'Horizon →', color=GRAY_D, fontsize=9)
    
    ax.set_xlabel('Date')
    ax.set_ylabel('Ventes (TND)')
    ax.set_title('Prévisions des ventes B2C — Modèle Prophet (basé sur vos données)', color=NAVY)
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    st.pyplot(fig)

    # ========== 5. DÉCOMPOSITION SAISONNIÈRE ==========
    divider()
    section("Décomposition saisonnière des ventes")
    
    try:
        from statsmodels.tsa.seasonal import seasonal_decompose
        if len(monthly_sales) >= 6:
            # Série avec index date
            ts = monthly_sales.set_index('ds')['y']
            decomposition = seasonal_decompose(ts, model='additive', period=3)
            
            fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
            fig.patch.set_facecolor(OFF_W)
            
            axes[0].plot(ts, color=NAVY)
            axes[0].set_title('Série originale', color=NAVY)
            
            axes[1].plot(decomposition.trend, color=GOLD)
            axes[1].set_title('Tendance', color=NAVY)
            
            axes[2].plot(decomposition.seasonal, color="#3A6BC9")
            axes[2].set_title('Saisonnalité', color=NAVY)
            
            axes[3].plot(decomposition.resid, color=GRAY_D)
            axes[3].set_title('Résidus', color=NAVY)
            
            plt.tight_layout()
            st.pyplot(fig)
        else:
            st.info("📌 Décomposition non disponible : besoin d'au moins 6 mois de données")
    except Exception as e:
        st.info(f"📌 Décomposition non disponible : {str(e)[:50]}")

    # ========== 6. ESTIMATION STOCK ==========
    divider()
    section("Estimation des besoins d'approvisionnement")
    
    avg_monthly = np.mean(historique_values[-6:]) if len(historique_values) >= 6 else np.mean(historique_values)
    std_monthly = np.std(historique_values[-6:]) if len(historique_values) >= 6 else np.std(historique_values)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Vente mensuelle moyenne (6 derniers mois)", f"{avg_monthly:.0f} TND")
    with col2:
        st.metric("Stock de sécurité (1.5σ)", f"{std_monthly * 1.5:.0f} TND", "1.5× écart-type")
    with col3:
        st.metric("Seuil réapprovisionnement", f"{avg_monthly + std_monthly:.0f} TND")

    rec("📌 Recommandation",
        f"Maintenir un stock de sécurité de {std_monthly * 1.5:.0f} TND. "
        f"Le seuil de réapprovisionnement est à {avg_monthly + std_monthly:.0f} TND. "
        f"Anticiper les commandes fournisseurs dès que le stock descend sous ce seuil.")
