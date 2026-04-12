# ========== PAGE PREVISIONS ==========
elif page == "Previsions":
    st.header("📅 Prévisions des ventes B2C")
    
    # Correction : 'ME' au lieu de 'M'
    dates = pd.date_range('2024-01-01', '2025-03-31', freq='ME')
    historique = [2100, 2300, 2500, 2800, 3000, 3500, 3800, 3600, 3300, 2900, 2600, 2400, 2500, 2700, 2900]
    previsions = [3100, 3300, 3500]
    
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(dates[:len(historique)], historique, 'b-o', label='Historique', linewidth=2, markersize=6)
    ax.plot(dates[len(historique):], previsions, 'r-s', label='Prévisions', linewidth=2, markersize=8)
    ax.fill_between(dates[len(historique):], [p-200 for p in previsions], [p+200 for p in previsions], alpha=0.2, color='red')
    ax.set_xlabel('Date')
    ax.set_ylabel('Ventes (TND)')
    ax.set_title('Prévisions des ventes B2C - 3 mois')
    ax.legend()
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)
    
    st.markdown("---")
    st.subheader("Estimation des besoins d'approvisionnement")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Vente mensuelle moyenne", "2 750 TND", "+320")
    with col2:
        st.metric("Stock de sécurité", "1 840 TND", "1.5× écart-type")
    with col3:
        st.metric("Seuil réapprovisionnement", "3 590 TND", "")
