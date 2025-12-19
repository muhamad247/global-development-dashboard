import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Page config
st.set_page_config(page_title="Global Development Dashboard", layout="wide")

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv('world_data_2023_cleaned.csv')
    # Load cluster assignments if available
    try:
        clusters = pd.read_csv('cluster_assignments_comparison_named.csv')
        df = df.merge(clusters[['Country', 'Cluster_K2', 'Cluster_K4']], on='Country', how='left')
    except:
        # If cluster file doesn't exist, create dummy clusters
        df['Cluster_K2'] = 'Unknown'
        df['Cluster_K4'] = 'Unknown'
    return df

df = load_data()

# Initialize session state for brushing & linking
if 'selected_countries' not in st.session_state:
    st.session_state.selected_countries = []

# Initialize filter reset flag
if 'filter_reset_counter' not in st.session_state:
    st.session_state.filter_reset_counter = 0

# Dashboard title
st.title("🌍 Global Development Patterns Dashboard")
st.markdown("Interactive exploration of country development indicators and clustering analysis")

# Sidebar filters
st.sidebar.header("🔍 Filters")

# Cluster filter
cluster_options = ['All'] + sorted(df['Cluster_K2'].dropna().unique().tolist())
selected_cluster = st.sidebar.selectbox(
    "Development Status", 
    cluster_options,
    key=f"cluster_select_{st.session_state.filter_reset_counter}"
)

# GDP range filter
gdp_min = float(df['GDP'].min())
gdp_max = float(df['GDP'].max())
gdp_range = st.sidebar.slider(
    "GDP Range (Billions USD)", 
    gdp_min/1e9, 
    gdp_max/1e9,
    (gdp_min/1e9, gdp_max/1e9),
    format="$%.1fB",
    key=f"gdp_slider_{st.session_state.filter_reset_counter}"
)

# Life expectancy filter
life_exp_range = st.sidebar.slider(
    "Life Expectancy (years)",
    float(df['Life_Expectancy'].min()),
    float(df['Life_Expectancy'].max()),
    (float(df['Life_Expectancy'].min()), float(df['Life_Expectancy'].max())),
    key=f"life_exp_slider_{st.session_state.filter_reset_counter}"
)

# Add button to clear selection and reset filters
if st.sidebar.button("🔄 Clear Selection & Reset Filters", use_container_width=True):
    st.session_state.selected_countries = []
    st.session_state.filter_reset_counter += 1
    st.rerun()

# Apply filters
df_filtered = df.copy()
if selected_cluster != 'All':
    df_filtered = df_filtered[df_filtered['Cluster_K2'] == selected_cluster]
df_filtered = df_filtered[
    (df_filtered['GDP'] >= gdp_range[0] * 1e9) & 
    (df_filtered['GDP'] <= gdp_range[1] * 1e9)
]
df_filtered = df_filtered[
    (df_filtered['Life_Expectancy'] >= life_exp_range[0]) & 
    (df_filtered['Life_Expectancy'] <= life_exp_range[1])
]

# Apply brushing filter if selection exists
df_display = df_filtered.copy()
if st.session_state.selected_countries:
    df_display = df_display[df_display['Country'].isin(st.session_state.selected_countries)]
    st.sidebar.success(f"🎯 {len(st.session_state.selected_countries)} countries selected")
    
    # Debug info - show selected countries
    with st.sidebar.expander("🔍 View selected countries"):
        for country in sorted(st.session_state.selected_countries):
            st.write(f"• {country}")

# Key metrics at top
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Countries", len(df_display))
with col2:
    avg_life = df_display['Life_Expectancy'].mean()
    st.metric("Avg Life Expectancy", f"{avg_life:.1f} years")
with col3:
    avg_infant = df_display['Infant_Mortality'].mean()
    st.metric("Avg Infant Mortality", f"{avg_infant:.1f}/1000")
with col4:
    avg_physicians = df_display['Physicians_per_1000'].mean()
    st.metric("Avg Physicians", f"{avg_physicians:.2f}/1000")

st.markdown("---")

# ============================================================================
# VISUALIZATION 1: Regional Distribution Map (LINKED)
# ============================================================================
st.subheader("🗺️ Global Distribution of Development Status")
st.caption("⬇️ Select countries in the scatter plots below to filter this map")

fig_map = px.choropleth(
    df_display,  # Use filtered data based on selection
    locations='Country',
    locationmode='country names',
    color='Life_Expectancy',
    hover_name='Country',
    hover_data={
        'GDP': ':,.0f',
        'Life_Expectancy': ':.1f',
        'Infant_Mortality': ':.1f',
        'Physicians_per_1000': ':.2f'
    },
    color_continuous_scale='RdYlGn',
    labels={'Life_Expectancy': 'Life Expectancy (years)'}
)

fig_map.update_layout(
    height=400,
    geo=dict(
        showframe=False,
        showcoastlines=True,
        projection_type='natural earth'
    )
)

st.plotly_chart(fig_map, use_container_width=True, key="map")

st.markdown("---")

# Main visualizations in 2x2 grid
col_left, col_right = st.columns(2)

# ============================================================================
# VISUALIZATION 2: Interactive Scatter - GDP vs Life Expectancy
# ============================================================================
with col_left:
    st.subheader("📊 Economic Development vs Health Outcomes")
    st.caption("🖱️ Drag to select countries")
    
    fig1 = px.scatter(
        df_filtered,
        x='GDP',
        y='Life_Expectancy',
        color='Cluster_K2',
        size='Population',
        hover_name='Country',
        hover_data={
            'GDP': ':,.0f',
            'Life_Expectancy': ':.1f',
            'Population': ':,.0f',
            'Cluster_K2': True
        },
        color_discrete_map={'Developed': '#e74c3c', 'Developing': '#3498db'},
        log_x=True,
        labels={
            'GDP': 'GDP (USD, log scale)',
            'Life_Expectancy': 'Life Expectancy (years)',
            'Cluster_K2': 'Development Status'
        }
    )
    
    
    # Add highlighted points as a separate trace if countries are selected
    if st.session_state.selected_countries:
        df_selected = df_filtered[df_filtered['Country'].isin(st.session_state.selected_countries)]
        fig1.add_trace(go.Scatter(
            x=df_selected['GDP'],
            y=df_selected['Life_Expectancy'],
            mode='markers',
            marker=dict(
                size=df_selected['Population'] / 1e6,  # Scale population for size
                color='#FFD700',  # Gold color
                sizemode='area',
                sizeref=2.*max(df_filtered['Population'])/(40.**2),
                sizemin=4,
                line=dict(width=2, color='#FF8C00')  # Orange border
            ),
            hovertext=df_selected['Country'],
            hovertemplate='<b>%{hovertext}</b><br>GDP: %{x:,.0f}<br>Life Expectancy: %{y:.1f}<extra></extra>',
            showlegend=False
        ))
    
    fig1.update_layout(
        height=400,
        hovermode='closest',
        clickmode='event+select',
        dragmode='select'
    )
    
    event1 = st.plotly_chart(fig1, use_container_width=True, key="scatter1", on_select="rerun")
    
    # Update selected countries from this plot
    if event1 and 'selection' in event1:
        points = event1['selection'].get('points', [])
        if points:
            # Extract country names from the hover data
            selected_countries = []
            for p in points:
                # Try multiple ways to get the country name
                if 'hovertext' in p:
                    selected_countries.append(p['hovertext'])
                elif 'pointIndex' in p:
                    idx = p['pointIndex']
                    if idx < len(df_filtered):
                        selected_countries.append(df_filtered.iloc[idx]['Country'])
                elif 'point_index' in p:
                    idx = p['point_index']
                    if idx < len(df_filtered):
                        selected_countries.append(df_filtered.iloc[idx]['Country'])
            
            if selected_countries:
                st.session_state.selected_countries = selected_countries
                st.rerun()

# ============================================================================
# VISUALIZATION 3: Healthcare Access vs Outcomes
# ============================================================================
with col_right:
    st.subheader("🏥 Healthcare System Performance")
    st.caption("🖱️ Drag to select countries")
    
    fig2 = px.scatter(
        df_filtered,
        x='Physicians_per_1000',
        y='Infant_Mortality',
        color='Cluster_K2',
        size='Population',
        hover_name='Country',
        hover_data={
            'Physicians_per_1000': ':.2f',
            'Infant_Mortality': ':.1f',
            'Population': ':,.0f',
            'Cluster_K2': True
        },
        color_discrete_map={'Developed': '#e74c3c', 'Developing': '#3498db'},
        labels={
            'Physicians_per_1000': 'Physicians per 1,000 people',
            'Infant_Mortality': 'Infant Mortality (per 1,000 births)',
            'Cluster_K2': 'Development Status'
        }
    )
    
    
    # Add highlighted points as a separate trace if countries are selected
    if st.session_state.selected_countries:
        df_selected = df_filtered[df_filtered['Country'].isin(st.session_state.selected_countries)]
        fig2.add_trace(go.Scatter(
            x=df_selected['Physicians_per_1000'],
            y=df_selected['Infant_Mortality'],
            mode='markers',
            marker=dict(
                size=df_selected['Population'] / 1e6,  # Scale population for size
                color='#FFD700',  # Gold color
                sizemode='area',
                sizeref=2.*max(df_filtered['Population'])/(40.**2),
                sizemin=4,
                line=dict(width=2, color='#FF8C00')  # Orange border
            ),
            hovertext=df_selected['Country'],
            hovertemplate='<b>%{hovertext}</b><br>Physicians: %{x:.2f}<br>Infant Mortality: %{y:.1f}<extra></extra>',
            showlegend=False
        ))
    
    fig2.update_layout(
        height=400,
        hovermode='closest',
        clickmode='event+select',
        dragmode='select'
    )
    
    event2 = st.plotly_chart(fig2, use_container_width=True, key="scatter2", on_select="rerun")
    
    # Update selected countries from this plot
    if event2 and 'selection' in event2:
        points = event2['selection'].get('points', [])
        if points:
            selected_countries = []
            for p in points:
                if 'hovertext' in p:
                    selected_countries.append(p['hovertext'])
                elif 'pointIndex' in p:
                    idx = p['pointIndex']
                    if idx < len(df_filtered):
                        selected_countries.append(df_filtered.iloc[idx]['Country'])
                elif 'point_index' in p:
                    idx = p['point_index']
                    if idx < len(df_filtered):
                        selected_countries.append(df_filtered.iloc[idx]['Country'])
            
            if selected_countries:
                st.session_state.selected_countries = selected_countries
                st.rerun()

# ============================================================================
# VISUALIZATION 4: GDP vs CO2 Emissions
# ============================================================================
st.subheader("🏭 Economic Activity vs Environmental Impact")
st.caption("🖱️ Drag to select countries")

# Filter for countries with CO2 data
df_co2 = df_filtered[['Country', 'GDP', 'CO2_Emissions', 'Population', 'Cluster_K2']].dropna()

if len(df_co2) > 0:
    fig_co2 = px.scatter(
        df_co2,
        x='GDP',
        y='CO2_Emissions',
        size='Population',
        color='Cluster_K2',
        hover_name='Country',
        hover_data={
            'GDP': ':,.0f',
            'CO2_Emissions': ':,.0f',
            'Population': ':,.0f',
            'Cluster_K2': True
        },
        color_discrete_map={'Developed': '#e74c3c', 'Developing': '#3498db'},
        log_x=True,
        log_y=True,
        labels={
            'GDP': 'GDP (USD, log scale)',
            'CO2_Emissions': 'CO2 Emissions (tonnes, log scale)',
            'Cluster_K2': 'Status'
        }
    )
    
    
    # Add highlighted points as a separate trace if countries are selected
    if st.session_state.selected_countries:
        df_co2_selected = df_co2[df_co2['Country'].isin(st.session_state.selected_countries)]
        fig_co2.add_trace(go.Scatter(
            x=df_co2_selected['GDP'],
            y=df_co2_selected['CO2_Emissions'],
            mode='markers',
            marker=dict(
                size=df_co2_selected['Population'] / 1e6,  # Scale population for size
                color='#FFD700',  # Gold color
                sizemode='area',
                sizeref=2.*max(df_co2['Population'])/(40.**2),
                sizemin=4,
                line=dict(width=2, color='#FF8C00')  # Orange border
            ),
            hovertext=df_co2_selected['Country'],
            hovertemplate='<b>%{hovertext}</b><br>GDP: %{x:,.0f}<br>CO2: %{y:,.0f}<extra></extra>',
            showlegend=False
        ))
    
    # Add annotations for major countries (only if they're in selection or no selection)
    major_countries = ['United States', 'China', 'India', 'Germany', 'Japan']
    for country in major_countries:
        if country in df_co2['Country'].values:
            # Only show annotation if country is selected or no selection exists
            if not st.session_state.selected_countries or country in st.session_state.selected_countries:
                country_data = df_co2[df_co2['Country'] == country].iloc[0]
                fig_co2.add_annotation(
                    x=np.log10(country_data['GDP']),
                    y=np.log10(country_data['CO2_Emissions']),
                    text=country,
                    showarrow=True,
                    arrowhead=2,
                    arrowsize=1,
                    arrowwidth=1,
                    arrowcolor='#555',
                    ax=20,
                    ay=-30,
                    font=dict(size=9, color='#333')
                )
    
    fig_co2.update_layout(
        height=400, 
        hovermode='closest',
        clickmode='event+select',
        dragmode='select'
    )
    
    event_co2 = st.plotly_chart(fig_co2, use_container_width=True, key="scatter3", on_select="rerun")
    
    # Update selected countries from this plot
    if event_co2 and 'selection' in event_co2:
        points = event_co2['selection'].get('points', [])
        if points:
            selected_countries = []
            for p in points:
                if 'hovertext' in p:
                    selected_countries.append(p['hovertext'])
                elif 'pointIndex' in p:
                    idx = p['pointIndex']
                    if idx < len(df_co2):
                        selected_countries.append(df_co2.iloc[idx]['Country'])
                elif 'point_index' in p:
                    idx = p['point_index']
                    if idx < len(df_co2):
                        selected_countries.append(df_co2.iloc[idx]['Country'])
            
            if selected_countries:
                st.session_state.selected_countries = selected_countries
                st.rerun()
else:
    st.warning("No CO2 emissions data available for selected countries")
    
# ============================================================================
# VISUALIZATION 4 & 5: Side-by-side comparisons (LINKED)
# ============================================================================
col_left2, col_right2 = st.columns(2)

with col_left2:
    st.subheader("📈 Top 10 Countries by Life Expectancy")
    
    top_life = df_display.nlargest(10, 'Life_Expectancy')[['Country', 'Life_Expectancy', 'Cluster_K2']]
    
    fig4 = px.bar(
        top_life,
        x='Life_Expectancy',
        y='Country',
        color='Cluster_K2',
        orientation='h',
        color_discrete_map={'Developed': '#e74c3c', 'Developing': '#3498db'},
        labels={
            'Life_Expectancy': 'Life Expectancy (years)',
            'Cluster_K2': 'Development Status'
        }
    )
    
    fig4.update_layout(
        height=400,
        showlegend=True,
        yaxis={'categoryorder': 'total ascending'}
    )
    
    st.plotly_chart(fig4, use_container_width=True, key="bar1")

with col_right2:
    st.subheader("📉 Top 10 Countries by Infant Mortality")
    
    # Top 10 means highest infant mortality (worst performers)
    bottom_infant = df_display.nlargest(10, 'Infant_Mortality')[['Country', 'Infant_Mortality', 'Cluster_K2']]
    
    fig5 = px.bar(
        bottom_infant,
        x='Infant_Mortality',
        y='Country',
        color='Cluster_K2',
        orientation='h',
        color_discrete_map={'Developed': '#e74c3c', 'Developing': '#3498db'},
        labels={
            'Infant_Mortality': 'Infant Mortality (per 1,000 births)',
            'Cluster_K2': 'Development Status'
        }
    )
    
    fig5.update_layout(
        height=400,
        showlegend=True,
        yaxis={'categoryorder': 'total ascending'}
    )
    
    st.plotly_chart(fig5, use_container_width=True, key="bar2")

# ============================================================================
# VISUALIZATION 7: Cluster Comparison (LINKED)
# ============================================================================
st.subheader("📊 Comparing Developed vs Developing Countries")
st.caption("Average values for selected countries (or all if none selected)")

cluster_summary = df_display.groupby('Cluster_K2').agg({
    'GDP': 'mean',
    'Life_Expectancy': 'mean',
    'Infant_Mortality': 'mean',
    'Physicians_per_1000': 'mean'
}).reset_index()

# Create 4 separate bar charts for clarity
fig6 = make_subplots(
    rows=1, cols=4,
    subplot_titles=('GDP (Billions)', 'Life Expectancy (years)', 
                   'Physicians per 1,000', 'Infant Mortality per 1,000')
)

colors = {'Developed': '#e74c3c', 'Developing': '#3498db'}

for idx, cluster in enumerate(cluster_summary['Cluster_K2']):
    cluster_data = cluster_summary[cluster_summary['Cluster_K2'] == cluster]
    color = colors.get(cluster, '#95a5a6')
    
    # GDP
    fig6.add_trace(
        go.Bar(name=cluster, x=[cluster], y=cluster_data['GDP']/1e9, 
               marker_color=color, showlegend=(idx==0)),
        row=1, col=1
    )
    
    # Life Expectancy
    fig6.add_trace(
        go.Bar(name=cluster, x=[cluster], y=cluster_data['Life_Expectancy'],
               marker_color=color, showlegend=False),
        row=1, col=2
    )
    
    # Physicians
    fig6.add_trace(
        go.Bar(name=cluster, x=[cluster], y=cluster_data['Physicians_per_1000'],
               marker_color=color, showlegend=False),
        row=1, col=3
    )
    
    # Infant Mortality
    fig6.add_trace(
        go.Bar(name=cluster, x=[cluster], y=cluster_data['Infant_Mortality'],
               marker_color=color, showlegend=False),
        row=1, col=4
    )

fig6.update_layout(height=350, showlegend=True)
fig6.update_xaxes(showticklabels=False)

st.plotly_chart(fig6, use_container_width=True)

# ============================================================================
# Data Table with Search (LINKED)
# ============================================================================
st.subheader("🔍 Search & Browse Countries")

# Search box
search_term = st.text_input("Search for a country:", "")

display_cols = ['Country', 'GDP', 'Life_Expectancy', 'Infant_Mortality', 
                'Physicians_per_1000', 'Population', 'Cluster_K2']

df_table = df_display[display_cols].copy()

# Filter by search
if search_term:
    df_table = df_table[df_table['Country'].str.contains(search_term, case=False, na=False)]

# Format nicely
df_table['GDP'] = df_table['GDP'].apply(lambda x: f"${x/1e9:.2f}B" if pd.notna(x) else "N/A")
df_table['Population'] = df_table['Population'].apply(lambda x: f"{int(x):,}" if pd.notna(x) else "N/A")
df_table = df_table.rename(columns={
    'Life_Expectancy': 'Life Exp.',
    'Infant_Mortality': 'Infant Mort.',
    'Physicians_per_1000': 'Physicians/1000',
    'Cluster_K2': 'Status'
})

st.dataframe(df_table, use_container_width=True, height=350)

# Summary
col1, col2 = st.columns(2)
with col1:
    st.info(f"**Showing {len(df_table)} of {len(df)} total countries**")
with col2:
    if len(df_table) > 0:
        developed_count = (df_table['Status'] == 'Developed').sum()
        developing_count = (df_table['Status'] == 'Developing').sum()
        st.info(f"**Developed: {developed_count} | Developing: {developing_count}**")