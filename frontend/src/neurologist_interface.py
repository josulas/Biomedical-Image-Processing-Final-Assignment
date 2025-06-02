"""Neurologist interface for patient analysis."""

import streamlit as st

class NeurologistInterface:
    """Interface for neurologist users."""
    def __init__(self, api_client):
        self.api_client = api_client
    def render(self, page: str):
        """Render neurologist interface based on current page."""
        if page == "neurologist_patients":
            self.show_patients()
        elif page == "neurologist_analysis":
            self.show_analysis_tools()
    def show_patients(self):
        """Show assigned patients."""
        st.subheader("👥 My Patients")
        st.info("Patient list interface - to be implemented")
    def show_analysis_tools(self):
        """Show image analysis tools."""
        st.subheader("🔬 Image Analysis")
        st.info("Analysis tools interface - to be implemented")
