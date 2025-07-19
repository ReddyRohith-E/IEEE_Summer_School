
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import os

# Set page config
st.set_page_config(
    page_title="Power System XAI Dashboard",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Title
st.title("⚡ Spatiotemporal Explainable AI for Power System Contingency Analysis")
st.markdown("### Interactive Dashboard for Model Performance and Explainability Analysis")

# Sidebar
st.sidebar.header("📊 Dashboard Controls")

# Load data function
@st.cache_data
def load_data():
    # This would load your actual data files
    # For demo purposes, we create sample data
    model_results = pd.DataFrame({
        'Model': ['LSTM', 'GRU', 'GCN', 'GCN_LSTM', 'GCN_GRU', 'GCN_GRU_LSTM'],
        'Accuracy': [0.85, 0.87, 0.82, 0.89, 0.88, 0.90],
        'Precision': [0.84, 0.86, 0.81, 0.88, 0.87, 0.89],
        'Recall': [0.86, 0.88, 0.83, 0.90, 0.89, 0.91],
        'F1': [0.85, 0.87, 0.82, 0.89, 0.88, 0.90],
        'ROC_AUC': [0.84, 0.86, 0.81, 0.88, 0.87, 0.89],
        'NDCG@5': [0.82, 0.84, 0.79, 0.86, 0.85, 0.87]
    })

    xai_results = pd.DataFrame({
        'Method': ['SHAP', 'LIME', 'Integrated Gradients', 'Gradient Attention'],
        'Fidelity': [0.842, 0.756, 0.891, 0.623],
        'Sparsity': [15.2, 22.8, 8.5, 5.3],
        'Consistency': [0.124, 0.189, 0.098, 0.267]
    })

    return model_results, xai_results

# Load data
model_results, xai_results = load_data()

# Sidebar selections
selected_models = st.sidebar.multiselect(
    "Select Models to Compare",
    model_results['Model'].tolist(),
    default=model_results['Model'].tolist()[:3]
)

selected_metrics = st.sidebar.multiselect(
    "Select Metrics to Display",
    ['Accuracy', 'Precision', 'Recall', 'F1', 'ROC_AUC', 'NDCG@5'],
    default=['Accuracy', 'F1', 'NDCG@5']
)

# Main content tabs
tab1, tab2, tab3, tab4 = st.tabs(["📊 Model Performance", "🔍 XAI Analysis", "📈 Detailed Metrics", "📋 Summary Report"])

with tab1:
    st.header("Model Performance Comparison")

    # Filter data based on selections
    filtered_models = model_results[model_results['Model'].isin(selected_models)]

    col1, col2 = st.columns(2)

    with col1:
        # Bar chart
        fig_bar = px.bar(
            filtered_models, 
            x='Model', 
            y=selected_metrics,
            title="Model Performance Metrics",
            barmode='group'
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    with col2:
        # Radar chart for best model
        best_model = filtered_models.loc[filtered_models['F1'].idxmax()]

        fig_radar = go.Figure()

        metrics_radar = ['Accuracy', 'Precision', 'Recall', 'F1', 'ROC_AUC']
        values_radar = [best_model[metric] for metric in metrics_radar]
        values_radar.append(values_radar[0])  # Close the polygon
        metrics_radar.append(metrics_radar[0])

        fig_radar.add_trace(go.Scatterpolar(
            r=values_radar,
            theta=metrics_radar,
            fill='toself',
            name=f'Best Model: {best_model["Model"]}'
        ))

        fig_radar.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )
            ),
            title="Best Model Performance Radar"
        )

        st.plotly_chart(fig_radar, use_container_width=True)

    # Model ranking table
    st.subheader("Model Ranking")
    ranking = filtered_models.sort_values('F1', ascending=False)
    st.dataframe(ranking, use_container_width=True)

with tab2:
    st.header("Explainable AI Analysis")

    col1, col2 = st.columns(2)

    with col1:
        # XAI method comparison
        fig_xai = px.bar(
            xai_results,
            x='Method',
            y=['Fidelity', 'Sparsity', 'Consistency'],
            title="XAI Methods Comparison",
            barmode='group'
        )
        st.plotly_chart(fig_xai, use_container_width=True)

    with col2:
        # Fidelity vs Sparsity scatter
        fig_scatter = px.scatter(
            xai_results,
            x='Sparsity',
            y='Fidelity',
            text='Method',
            title="Fidelity vs Sparsity Trade-off",
            size=[20]*len(xai_results)
        )
        fig_scatter.update_traces(textposition="top center")
        st.plotly_chart(fig_scatter, use_container_width=True)

    # XAI recommendations
    st.subheader("XAI Method Recommendations")

    best_fidelity = xai_results.loc[xai_results['Fidelity'].idxmax()]
    best_sparsity = xai_results.loc[xai_results['Sparsity'].idxmax()]

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Highest Fidelity",
            best_fidelity['Method'],
            f"{best_fidelity['Fidelity']:.3f}"
        )

    with col2:
        st.metric(
            "Most Sparse",
            best_sparsity['Method'],
            f"{best_sparsity['Sparsity']:.1f}%"
        )

    with col3:
        overall_best = xai_results.loc[
            (xai_results['Fidelity'] * 0.6 + (100 - xai_results['Sparsity']) * 0.4 / 100).idxmax()
        ]
        st.metric(
            "Overall Best",
            overall_best['Method'],
            "Recommended"
        )

with tab3:
    st.header("Detailed Performance Metrics")

    # Detailed model comparison
    st.subheader("Classification Performance")
    st.dataframe(model_results.round(4), use_container_width=True)

    st.subheader("XAI Benchmarking Results")
    st.dataframe(xai_results.round(4), use_container_width=True)

    # Download buttons
    col1, col2 = st.columns(2)

    with col1:
        csv_models = model_results.to_csv(index=False)
        st.download_button(
            label="📥 Download Model Results (CSV)",
            data=csv_models,
            file_name="model_performance_results.csv",
            mime="text/csv"
        )

    with col2:
        csv_xai = xai_results.to_csv(index=False)
        st.download_button(
            label="📥 Download XAI Results (CSV)",
            data=csv_xai,
            file_name="xai_benchmarking_results.csv",
            mime="text/csv"
        )

with tab4:
    st.header("Executive Summary Report")

    # Key findings
    best_model = model_results.loc[model_results['F1'].idxmax()]
    best_xai = xai_results.loc[xai_results['Fidelity'].idxmax()]

    st.subheader("🎯 Key Findings")

    col1, col2 = st.columns(2)

    with col1:
        st.info(f"""
        **Best Performing Model**: {best_model['Model']}
        - Accuracy: {best_model['Accuracy']:.3f}
        - F1-Score: {best_model['F1']:.3f}
        - NDCG@5: {best_model['NDCG@5']:.3f}
        """)

    with col2:
        st.success(f"""
        **Recommended XAI Method**: {best_xai['Method']}
        - Fidelity: {best_xai['Fidelity']:.3f}
        - Sparsity: {best_xai['Sparsity']:.1f}%
        - Consistency: {best_xai['Consistency']:.3f}
        """)

    st.subheader("📊 Summary Statistics")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Models Evaluated",
            len(model_results),
            "Deep Learning"
        )

    with col2:
        st.metric(
            "XAI Methods",
            len(xai_results),
            "Benchmarked"
        )

    with col3:
        st.metric(
            "Avg Accuracy",
            f"{model_results['Accuracy'].mean():.3f}",
            f"{(model_results['Accuracy'].std()*100):.1f}% std"
        )

    with col4:
        st.metric(
            "Avg XAI Fidelity",
            f"{xai_results['Fidelity'].mean():.3f}",
            f"{(xai_results['Fidelity'].std()*100):.1f}% std"
        )

    st.subheader("💡 Recommendations")

    st.markdown("""
    **For Power System Operations:**
    1. Deploy the **{best_model}** model for contingency classification
    2. Use **{best_xai}** for generating explanations
    3. Implement real-time monitoring with explanation capabilities
    4. Regular model retraining with updated operational data

    **For Regulatory Compliance:**
    1. Document model decisions with XAI explanations
    2. Maintain audit trails of prediction reasoning
    3. Provide transparent explanations to stakeholders
    """.format(
        best_model=best_model['Model'],
        best_xai=best_xai['Method']
    ))

# Footer
st.markdown("---")
st.markdown(
    "🔬 **Spatiotemporal Explainable AI for Power System Contingency Analysis** | "
    "Built with Streamlit 🚀"
)
