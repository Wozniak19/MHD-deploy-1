import streamlit as st
import matplotlib.pyplot as plt

def main():
    st.title("Mental Health Disorder Probabilities")

    # Example predicted probabilities from your model
    predicted_probabilities = {
        "Depression": 0.15,
        "Schizophrenia": 0.10,
        "Acute and Transient Psychotic Disorder": 0.05,
        "Delusional Disorder": 0.05,
        "BiPolar Disorder": 0.20,
        "Anxiety": 0.15,
        "Generalized Anxiety": 0.05,
        "Panic Disorder": 0.05,
        "Specific Phobia": 0.05,
        "Social Anxiety": 0.05,
        "OCD": 0.05,
        "PTSD": 0.05,
        "Gambling Disorder": 0.05,
        "Substance Abuse": 0.05
    }

    # Convert probabilities to percentages
    labels = list(predicted_probabilities.keys())
    sizes = [prob * 100 for prob in predicted_probabilities.values()]  # Convert to percentages

    # Create the pie chart
    fig, ax = plt.subplots()
    wedges, texts, autotexts = ax.pie(
        sizes,
        labels=labels,
        autopct='%1.1f%%',  # Format of the percentage
        startangle=140,
        colors=plt.get_cmap('tab10').colors
    )

    # Add a legend
    ax.legend(wedges, labels,
              title="Disorders",
              loc="center left",
              bbox_to_anchor=(1, 0, 0.5, 1))

    # Add a title
    plt.title('Probability Distribution of Mental Health Disorders')

    # Display the pie chart in Streamlit
    st.pyplot(fig)

if __name__ == "__main__":
    main()
