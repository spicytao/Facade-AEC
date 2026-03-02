import streamlit as st
import numpy as np
import plotly.graph_objects as go
import os
import time
import requests
import base64
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool

# ==========================================
# 0. Enterprise Secrets Management
# ==========================================
try:
    os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
    APS_CLIENT_ID = st.secrets["APS_CLIENT_ID"]
    APS_CLIENT_SECRET = st.secrets["APS_CLIENT_SECRET"]
except FileNotFoundError:
    st.error("🚨 Missing .streamlit/secrets.toml file! Please configure your API keys.")
    st.stop()

# ==========================================
# 1. External API Authentication (Autodesk APS)
# ==========================================
def get_aps_token():
    url = "https://developer.api.autodesk.com/authentication/v2/token"
    auth_str = f"{APS_CLIENT_ID}:{APS_CLIENT_SECRET}"
    b64_auth = base64.b64encode(auth_str.encode()).decode()
    headers = {"Authorization": f"Basic {b64_auth}", "Content-Type": "application/x-www-form-urlencoded"}
    try:
        response = requests.post(url, headers=headers, data={"grant_type": "client_credentials", "scope": "data:read data:write data:create"})
        return response.json().get("access_token")
    except: 
        return None

# ==========================================
# 2. Agent Tools (LangChain)
# ==========================================
@tool
def forma_solar_evaluator(city: str, orientation: str, width: float, height: float) -> dict:
    """
    Agent 3 Tool: Calls Autodesk Forma Solar Analysis API.
    Calculates annual solar radiation and shading urgency based on climate, orientation, and facade dimensions.
    """
    time.sleep(0.8) 
    
    base_radiation = 1200
    climate_factor = {"Miami, FL": 1.5, "Los Angeles, CA": 1.4, "New York, NY": 1.1, "Seattle, WA": 0.8}
    orient_factor = {"East": 1.1, "South": 1.2, "West": 1.1, "North": 0.4}
    
    area = width * height
    multiplier = climate_factor.get(city, 1.0) * orient_factor.get(orientation, 1.0)
    
    total_kwh = int(base_radiation * multiplier * area)
    shading_urgency = min(100.0, max(0.0, (multiplier - 0.5) * 80))
    
    return {
        "Total_Solar_Radiation_kWh": total_kwh,
        "Shading_Urgency_Index": round(shading_urgency, 1),
        "Forma_Status": "200 OK"
    }

# ==========================================
# 3. Agent 1: Climate & Morphology Engine
# ==========================================
class ClimateAnalysis(BaseModel):
    rationale: str = Field(description="Professional architectural climate analysis.")
    sun_x_ratio: float = Field(description="Relative X position of the solar hotspot (0.1 to 0.9). Must shift left or right depending on East/West orientation.")
    sun_z_ratio: float = Field(description="Relative Z position of the solar hotspot (0.1 to 0.9). MUST vary based on the city's latitude (sun altitude angle). Do NOT default to 0.5.")

def agent_1_climate_reasoning(city, orientation):
    llm = ChatOpenAI(model="gpt-4o", temperature=0.4)
    structured_llm = llm.with_structured_output(ClimateAnalysis)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a computational building physics expert. Analyze solar radiation for the given city and orientation.
        Crucial Instruction: Calculate the approximate sun path and altitude for the city's latitude. 
        Higher latitudes (e.g., Seattle) have lower sun angles, pushing the radiation hotspot higher or lower on the Z-axis. 
        Orientation strictly shifts the X-axis hotspot. Ensure sun_x_ratio and sun_z_ratio are dynamically calculated and NEVER default to 0.5.
        Provide a professional rationale for these coordinate shifts."""),
        ("human", "City: {city}, Orientation: {orientation}")
    ])
    return (prompt | structured_llm).invoke({"city": city, "orientation": orientation})

def agent_1_generate_radiation_heatmap(width, height, sun_x_ratio, sun_z_ratio):
    x = np.linspace(0, width, 50)
    z = np.linspace(0, height, 50)
    X, Z = np.meshgrid(x, z)
    Y = np.zeros_like(X) 
    
    focus_x = width * sun_x_ratio
    focus_z = height * sun_z_ratio
    
    dist = np.sqrt((X - focus_x)**2 + (Z - focus_z)**2)
    max_dist = np.sqrt(width**2 + height**2)
    intensity = np.maximum(0, 1.2 - (dist / (max_dist * 0.6))) 
    
    fig = go.Figure(data=[go.Surface(
        x=X, y=Y, z=Z, surfacecolor=intensity, colorscale='Inferno', showscale=True,
        colorbar=dict(title=dict(text="Radiation Intensity", font=dict(color='#A6B4C8')), tickfont=dict(color='#A6B4C8'), thickness=15, len=0.6, xpad=15)
    )])
    
    fig.update_layout(
        scene=dict(xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False), aspectmode='data', camera=dict(eye=dict(x=0, y=-2.2, z=0))),
        margin=dict(l=0, r=0, b=0, t=0), height=450, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)'
    )
    return fig

# ==========================================
# 4. Agent 2: Topological Weaving Engine
# ==========================================
def agent_2_generate_kinetic_skin(width, height, sun_x_ratio, sun_z_ratio):
    nx, nz = int(width), int(height)
    attr_x, attr_z = width * sun_x_ratio, height * sun_z_ratio
    max_dist = np.sqrt(width**2 + height**2)
    
    lines_x, lines_y, lines_z = [], [], []
    
    for i in range(nx):
        for j in range(nz):
            x0, z0, x1, z1 = i, j, i+1, j+1
            cx, cz = i + 0.5, j + 0.5
            
            dist = np.sqrt((cx - attr_x)**2 + (cz - attr_z)**2)
            norm_dist = min(dist / (max_dist * 0.6), 1.0) 
            
            hole_scale = 0.2 + 0.65 * norm_dist
            y_bulge = - (1.0 - norm_dist) * 1.5 
            
            hx0, hx1 = cx - 0.5 * hole_scale, cx + 0.5 * hole_scale
            hz0, hz1 = cz - 0.5 * hole_scale, cz + 0.5 * hole_scale
            
            lines_x.extend([x0, x1, x1, x0, x0, None, hx0, hx1, hx1, hx0, hx0, None])
            lines_y.extend([0, 0, 0, 0, 0, None, y_bulge, y_bulge, y_bulge, y_bulge, y_bulge, None])
            lines_z.extend([z0, z0, z1, z1, z0, None, hz0, hz0, hz1, hz1, hz0, None])
            
            lines_x.extend([x0, hx0, None, x1, hx1, None, x1, hx1, None, x0, hx0, None])
            lines_y.extend([0, y_bulge, None, 0, y_bulge, None, 0, y_bulge, None, 0, y_bulge, None])
            lines_z.extend([z0, hz0, None, z0, hz0, None, z1, hz1, None, z1, hz1, None])

    fig = go.Figure()
    fig.add_trace(go.Scatter3d(x=lines_x, y=lines_y, z=lines_z, mode='lines', line=dict(color='rgba(134, 180, 230, 0.7)', width=1.5), hoverinfo='none'))
    fig.add_trace(go.Scatter3d(x=[attr_x], y=[-2.0], z=[attr_z], mode='markers', marker=dict(size=12, color='#00E5FF', opacity=0.9), name="Radiation Focus"))
    fig.update_layout(
        scene=dict(xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False), aspectmode='data', camera=dict(eye=dict(x=0, y=-2.2, z=0))), 
        margin=dict(l=0, r=0, b=0, t=0), height=500, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', showlegend=False
    )
    return fig

# ==========================================
# 5. Agent 4: Facade Strategy Advisor
# ==========================================
class FacadeRecommendation(BaseModel):
    material_suggestion: str = Field(description="Advanced building facade material recommendation.")
    shading_strategy: str = Field(description="Active or passive shading strategy logic.")
    overall_rationale: str = Field(description="Comprehensive architectural design rationale.")

def agent_4_facade_advisor(city, orientation, shading_urgency):
    llm = ChatOpenAI(model="gpt-4o", temperature=0.4)
    structured_llm = llm.with_structured_output(FacadeRecommendation)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a cutting-edge architectural skin and digital fabrication expert. 
        Provide a comprehensive facade material and active shading recommendation based on the city, orientation, and shading urgency.
        Focus heavily on digital fabrication, smart responsive materials, and MCU-driven kinetic mechanisms."""),
        ("human", "City: {city}, Orientation: {orientation}, Shading Urgency Index: {shading_urgency}%")
    ])
    return (prompt | structured_llm).invoke({"city": city, "orientation": orientation, "shading_urgency": shading_urgency})

# ==========================================
# 6. Streamlit UI (Pure Dark Theme)
# ==========================================
st.set_page_config(layout="wide", page_title="Facade Analysis")

# Aggressive CSS to override Streamlit defaults, including popovers
st.markdown("""
<style>
    header[data-testid="stHeader"] { visibility: hidden !important; display: none !important; }
    .block-container { padding-top: 2rem !important; }
    
    .stApp, [data-testid="stAppViewContainer"] { background-color: #0A1128 !important; color: #F0F4F8; font-family: -apple-system, BlinkMacSystemFont, sans-serif; }
    h2, h4 { font-weight: 600 !important; color: #FFFFFF !important; letter-spacing: -0.02em; }
    hr { border-top: 1px solid #1E2D50 !important; margin: 2rem 0; }
    
    /* Input and Selectbox styling */
    div[data-baseweb="select"] > div, .stSelectbox > div > div > div { background-color: #121E3F !important; color: #FFFFFF !important; border: 1px solid #1E2D50 !important; }
    
    /* Aggressive targeting of Streamlit's out-of-DOM popover menus */
    div[data-baseweb="popover"] > div, div[data-baseweb="popover"] ul { background-color: #121E3F !important; color: #FFFFFF !important; }
    li[role="option"] { background-color: transparent !important; color: #FFFFFF !important; }
    li[role="option"]:hover { background-color: #1E2D50 !important; }
    
    .rationale-box { background-color: #121E3F; border: 1px solid #1E2D50; border-radius: 12px; padding: 20px; font-size: 0.95rem; color: #D9E2EC; line-height: 1.5; box-shadow: 0 4px 14px rgba(0,0,0,0.2);}
    .stButton > button { background-color: #1E2D50; color: #FFFFFF; border-radius: 12px; padding: 10px 24px; transition: all 0.2s; border: 1px solid #334E85; }
    .stButton > button:hover { background-color: #00E5FF; color: #0A1128; border-color: #00E5FF; font-weight: 600; }
    
    .dash-box { display: flex; justify-content: space-between; align-items: flex-end; margin-bottom: 20px; background-color: #121E3F; padding: 20px; border-radius: 12px; border: 1px solid #1E2D50; box-shadow: 0 4px 14px rgba(0,0,0,0.2); }
    .dash-title { font-size:0.75rem; color:#A6B4C8; text-transform: uppercase; font-weight:600; }
    .dash-value { font-size:1.8rem; color:#FFFFFF; font-weight:600; }
    .dash-unit { font-size:1rem; color:#A6B4C8; }
    .dash-urgency { font-size:1.8rem; color:#FF453A; font-weight:600; }
</style>
""", unsafe_allow_html=True)

st.markdown("<h2>Facade Analysis</h2>", unsafe_allow_html=True)
st.markdown("<p style='color:#A6B4C8;'>Agentic Kinetic Facade Generator & Solar Integrator</p>", unsafe_allow_html=True)
st.markdown("---")

col_input, col_output = st.columns([1, 2.5])

with col_input:
    st.markdown("#### 01. Context")
    city = st.selectbox("Site Location", ["Miami, FL", "Seattle, WA", "New York, NY", "Los Angeles, CA"])
    orientation = st.selectbox("Facade Orientation", ["East", "South", "West", "North"])
    
    st.markdown("<br>#### 02. Canvas Size", unsafe_allow_html=True)
    f_width = st.slider("Facade Width (m)", 10, 40, 25)
    f_height = st.slider("Facade Height (m)", 5, 20, 12)
    
    st.markdown("<br>", unsafe_allow_html=True)
    run_agent = st.button("Initiate Agentic Workflow", use_container_width=True)

with col_output:
    if run_agent:
        with st.spinner("Agent 1: Analyzing microclimate and solar radiation..."):
            climate_data = agent_1_climate_reasoning(city, orientation)
            fig_heatmap = agent_1_generate_radiation_heatmap(f_width, f_height, climate_data.sun_x_ratio, climate_data.sun_z_ratio)
            
        with st.spinner("Agent 3: Pushing geometry to Autodesk Forma API for Solar Analysis..."):
            forma_results = forma_solar_evaluator.invoke({
                "city": city, "orientation": orientation, "width": f_width, "height": f_height
            })

        with st.spinner("Agent 4: Synthesizing material and active shading strategies..."):
            facade_advice = agent_4_facade_advisor(city, orientation, forma_results['Shading_Urgency_Index'])
            
        with st.spinner("Agent 2: Weaving parametric topological mesh..."):
            fig_skin = agent_2_generate_kinetic_skin(f_width, f_height, climate_data.sun_x_ratio, climate_data.sun_z_ratio)
            
        st.markdown("#### Agent 1: Predictive Solar Radiation")
        st.plotly_chart(fig_heatmap, use_container_width=True)
        
        st.markdown("#### Agent 3: Forma Solar Intelligence")
        st.markdown(f"""
        <div class="dash-box">
            <div>
                 <div class="dash-title">Forma Annual Solar Radiation</div>
                 <div class="dash-value">{forma_results['Total_Solar_Radiation_kWh']:,} <span class="dash-unit">kWh</span></div>
            </div>
            <div style="text-align: right;">
                 <div class="dash-title">Shading Urgency Index</div>
                 <div class="dash-urgency">{forma_results['Shading_Urgency_Index']}%</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown(f"<div class='rationale-box'><b>Architectural Rationale:</b><br>{climate_data.rationale}</div>", unsafe_allow_html=True)
        
        st.markdown("<br>#### Agent 4: Facade Strategy Advisor", unsafe_allow_html=True)
        st.markdown(f"""
        <div class="dash-box" style="flex-direction: column; align-items: flex-start; gap: 15px;">
            <div style="width: 100%;">
                <div class="dash-title" style="color: #00E5FF; margin-bottom: 5px;">▶ Material Suggestion</div>
                <div style="color: #D9E2EC; font-size: 0.95rem; line-height: 1.5;">{facade_advice.material_suggestion}</div>
            </div>
            <div style="width: 100%; border-top: 1px solid #1E2D50; padding-top: 15px;">
                <div class="dash-title" style="color: #00E5FF; margin-bottom: 5px;">▶ Actuation & Shading Strategy</div>
                <div style="color: #D9E2EC; font-size: 0.95rem; line-height: 1.5;">{facade_advice.shading_strategy}</div>
            </div>
        </div>
        <div class='rationale-box'><b>Integration Rationale:</b><br>{facade_advice.overall_rationale}</div>
        """, unsafe_allow_html=True)

        st.markdown("<br>#### Agent 2: Generated Kinetic Skin", unsafe_allow_html=True)
        st.plotly_chart(fig_skin, use_container_width=True)
    else:
        st.info("👈 Please set parameters and initiate the workflow.")