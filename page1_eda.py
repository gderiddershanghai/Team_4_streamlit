import streamlit as st
import streamlit.components.v1 as components

st.set_page_config(layout="wide")

st.title("Tableau Dashboard")

# Embed the Tableau Dashboard with centered alignment
tableau_dashboard_code = '''
<div style="text-align:center;">
    <iframe src="https://public.tableau.com/views/MGT6203_TEAM4_2024Spring/Dashboard1?:embed=y&:display_count=yes&:showVizHome=no"
        style="margin-left: auto; margin-right: auto; display: block; width: 100%; height: 1000px; border: none;">
    </iframe>
</div>
'''

components.html(tableau_dashboard_code, height=1000)
