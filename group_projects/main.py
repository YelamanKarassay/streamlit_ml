import streamlit as st
import group1
import group2
import group3
import group4
import group5
import group6
import group7

# Set the title for the main app
st.title("Multi-Application Dashboard")

# Create a sidebar for navigation
st.sidebar.title("Navigation")
# Define the different app options in a radio button
display_app = st.sidebar.radio("Select an App", ["Group 1", "Group 2", "Group 3", "Group 4", "Group 5", "Group 6", "Group 7"])

# Load the appropriate app based on user selection
if display_app == "Group 1":
    st.header("Group 1")
    group1.main()  # Each group file has a main() function
elif display_app == "Group 2":
    st.header("Group 2")
    group2.main()
elif display_app == "Group 3":
    st.header("Group 3")
    group3.main()
elif display_app == "Group 4":
    st.header("Group 4")
    group4.main()
elif display_app == "Group 5":
    st.header("Group 5")
    group5.main()
elif display_app == "Group 6":
    st.header("Group 6")
    group6.main()
elif display_app == "Group 7":
    st.header("Group 7")
    group7.main()

# Optionally, you can add a footer or extra information here
st.sidebar.info("This is a multi-page Streamlit app with navigation.")
