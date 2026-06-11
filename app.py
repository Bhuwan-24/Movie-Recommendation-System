import streamlit as st
from recommend import similarity_calculate

st.set_page_config(
    page_title="Product Recommendation System",
    layout="wide"
)

st.title("🛒 E-Commerce Product Recommendation System")

st.write(
    "Search for a product and get similar recommendations."
)

query = st.text_input(
    "Product Search",
    placeholder="Nike shoes"
)

category = st.text_input(
    "Category (Optional)"
)

top_n = st.slider(
    "Number of Recommendations",
    5,
    20,
    10
)

if st.button("Recommend"):

    if query.strip():

        result = similarity_calculate(
            title=query,
            category=category,
            top_similar=top_n
        )

        st.subheader("Recommended Products")

        st.dataframe(
            result,
            use_container_width=True
        )

    else:
        st.warning(
            "Please enter a product query."
        )