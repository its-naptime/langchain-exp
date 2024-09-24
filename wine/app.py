import streamlit as st
from langchain_agent import handle_query
import asyncio
from dotenv import load_dotenv
import os

load_dotenv()

st.set_page_config(page_title="Wine Recommendation App", page_icon="🍷")

async def main():
    st.title("🍷 Wine Recommendation App")

    user_query = st.text_input("Describe your wine preferences:", "I like fruity red wines with a hint of oak")

    if st.button("Get Recommendations"):
        with st.spinner("Analyzing your preferences..."):
            try:
                _, recommendations = await handle_query(user_query)
                
                st.subheader("Recommended Wines:")
                for rec in recommendations.recommendations:
                    with st.expander(f"{rec.name} (Score: {rec.score:.2f})"):
                        st.write(f"**Reason:** {rec.reason}")
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")

    st.sidebar.header("About")
    st.sidebar.info("This app uses AI to recommend wines based on your preferences. Wine Dataset: https://www.kaggle.com/datasets/priyamchoksi/global-wine-ratings-dataset")

if __name__ == "__main__":
    asyncio.run(main())