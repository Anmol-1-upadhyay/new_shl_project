import streamlit as st
import requests

st.set_page_config(page_title="SHL Recommender", layout="wide")

st.title("SHL Assessment Recommendation System")

query = st.text_area(
    "Enter job description or query:", 
    height=150,
    placeholder="Example: 'Need Java developer assessment under 40 minutes with collaboration skills'"
)

if st.button("Get Recommendations"):
    if query:
        with st.spinner("Analyzing query and finding best assessments..."):
            try:
                response = requests.post(
                    "https://new-shl-project.onrender.com/recommend",
                    json={"query": query},
                    timeout=30
                )
                
                if response.status_code == 200:
                    results = response.json().get("recommended_assessments", [])
                    
                    if not results:
                        st.warning("No matching assessments found. Try broadening your search criteria.")
                    else:
                        st.subheader(f"Top {len(results)} Recommendations")
                        
                        # Display in 2 columns
                        cols = st.columns(2)
                        for idx, assessment in enumerate(results, 1):
                            col = cols[(idx-1)%2]
                            with col:
                                with st.expander(f"{assessment['name']}", expanded=True):
                                    st.markdown(f"**Assessment URL**: [{assessment['url']}]({assessment['url']})")
                                    st.write(f"⏳ Duration: {assessment['duration']} minutes")
                                    st.write(f"🌐 Remote Testing: {assessment['remote_support']}")
                                    st.write(f"🔄 Adaptive Support: {assessment['adaptive_support']}")
                                    st.write(f"📊 Test Types: {', '.join(assessment['test_type'])}")
                                    st.divider()
                else:
                    st.error(f"API Error: {response.text}")
                    
            except Exception as e:
                st.error(f"Connection error: {str(e)}")
    else:
        st.warning("Please enter a query")




