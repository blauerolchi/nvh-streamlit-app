import streamlit as st
import time

# 1. Test: Kann die Seite überhaupt geladen werden?
st.set_page_config(page_title="Verbindungstest", page_icon="🟢")
st.title("🟢 Streamlit Verbindungstest")
st.write("Wenn du diesen Text siehst, funktioniert das grundlegende Server-Routing!")

# 2. Test: Funktionieren die WebSockets? (Das, was dein Browser vorher blockiert hat)
st.divider()
st.write("Teste jetzt die aktive Datenverbindung:")

if st.button("Verbindung prüfen"):
    with st.spinner("Sende Ping an Server..."):
        time.sleep(1) # Simuliert eine kurze Server-Bedenkzeit
        st.success("Erfolg! Die WebSocket-Verbindung ist stabil und wird NICHT blockiert.")
        st.balloons()
