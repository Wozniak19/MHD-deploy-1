import streamlit as st
from collections import defaultdict
from decimal import *
import joblib
import numpy as np
import matplotlib.pyplot as plt

# Load the trained model
model = joblib.load('gbm3_model.pkl')

def main():
    st.title("Mental Health Diagnostic Questionnaire 🧠")

    query = {
        "Depression ": [
            "1. Have you been feeling consistently sad, down, or hopeless?",
            "2. Have you lost interest or pleasure in activities you used to enjoy?",
            "3. Do you have trouble thinking, concentrating, or making decisions?",
            "4. Have you noticed changes in your behavior, such as withdrawing from social activities or neglecting responsibilities?",
            "5. Have you experienced changes in your sleep patterns or appetite, such as sleeping too much or too little, or significant weight loss or gain?",
            "6. Do you often feel worthless or have low self-esteem?",
            "7. Have you had thoughts of suicide or harming yourself?",
            "8. Do you often feel fatigued or have low energy, even when you haven't exerted yourself?",
            "9. Do you find yourself feeling irritable or easily annoyed?",
            "10. Do you have trouble sleeping, or do you find yourself sleeping too much?",
            "11. Do you struggle with making decisions, even about simple things?",
            "12. Have others noticed that you talk or move more slowly than usual?",
            "13. Have you experienced a decrease in your interest in sex?",
        ],
        "Schizophrenia ": [
            "14. Do you hold beliefs that others consider false or unusual, and that you are unwilling to give up?",
            "15. Do you see or hear things that others do not perceive?",
            "16. Do you find it difficult to organize your thoughts or speak in a way that others can easily understand?",
            "17. Do you experience difficulties in more than one area of your mental functioning, such as thinking, perception, or emotions?",
            "18. Have you noticed a decline in your ability to function at work, in social settings, or in daily activities?",
            "19. Do you find yourself lacking motivation, struggling to express emotions, or feeling less pleasure in activities you once enjoyed?",
            "20. Is there a history of schizophrenia or similar mental health conditions in your family?",
            "21. Have you noticed subtle motor or sensory changes, such as awkwardness or difficulties with coordination?",
            "22. Are your symptoms present even when you are not using drugs or experiencing a medical condition that could explain them?",
        ],
        "Acute and Transient Psychotic Disorder": [
            "23. Have you experienced sudden hallucinations that peaked within two weeks and lasted less than three months?",
            "24. Have you had sudden delusions that peaked within two weeks and lasted less than three months?",
            "25. Do you find yourself speaking in a way that is difficult for others to follow or understand?",
            "26. Have you experienced confusion, disorientation, or sudden changes in awareness or cognition?",
            "27. Have you noticed any periods of extreme stillness, unusual movements, or unresponsiveness?",
            "28. Have your symptoms been manageable without the need for hospitalization?",
        ],
        "Delusional Disorder ": [
            "29. Have you experienced delusions that have lasted for at least three months?",
            "30. Apart from delusions, have you not experienced other symptoms commonly associated with schizophrenia, such as hallucinations or disorganized speech?",
            "31. Have you not experienced significant mood episodes like depression or mania?",
            "32. Do your delusions persist over time without significant change?",
            "33. Have you experienced one or more delusions lasting at least one month?",
            "34. Have you noticed any perceptual disturbances, such as seeing or hearing things differently?",
            "35. Have your delusions caused only minimal impact on your daily functioning or ability to work and socialize?",
        ],
        "Bipolar Disorder": [
            "36. Have you experienced thoughts racing through your mind, making it difficult to focus on one thing at a time?",
            "37. Have you found yourself excessively involved in activities that could lead to painful or harmful consequences, such as reckless spending or risky behaviors?",
            "38. Have you experienced mood disturbances such as being excessively talkative, feeling grandiose, getting easily distracted, sleeping less, or feeling irritable? (Note: At least three should be present, or four if irritability is one of them.)",
            "39. Have you experienced a heightened mood and increased activity level that lasted for over a week?",
            "40. Has your mood ever become so disruptive that it required hospitalization?",
            "41. Have you engaged in risky behaviors that persisted for more than a week?",
            "42. Have your symptoms occurred without being caused by the effects of substance use, such as drugs or alcohol?",
            "43. Have you experienced any psychotic symptoms, such as delusions or hallucinations, during your mood episodes?",
            "44. Have you noticed a clear change in your functioning during an episode that seems uncharacteristic of your usual self?",
            "45. Have others noticed a disturbance in your mood and a change in your functioning?",
            "46. Have you experienced a depressive episode, including symptoms like a depressed mood, loss of pleasure, significant weight changes, sleep disturbances, psychomotor agitation, fatigue, feelings of worthlessness, difficulty concentrating, or recurrent thoughts of death?",
        ],
        "Generalized Anxiety Disorder(GAD)": [
            "47. Have you experienced excessive worry or restlessness that is difficult to control?",
            "48. Have you experienced three or more of the following symptoms associated with your anxiety: fatigue, difficulty concentrating, irritability, muscle tension, or sleep disturbances?",
            "49. Have you had persistent anxiety or worry about various aspects of your life that has lasted for over six months?",
            "50. Have you found it difficult to control your worry or stop yourself from worrying?",
            "51. Has your anxiety caused significant distress or impairment in your social, occupational, or other important areas of functioning?",
            "52. Have your symptoms occurred without being caused by the effects of substances or medical conditions?",
            "53. Do your symptoms of anxiety occur most of the day and are not limited to specific situations or objects?",
            ],
        "Panic Disorder": [
            "54. Have you experienced recurrent panic attacks involving abrupt, intense fear, with symptoms like palpitations, sweating, trembling, shortness of breath, chest pain, dizziness, chills, hot flushes, or fear of imminent death, with four or more symptoms present?",
            "55. Have you experienced discrete episodes of intense fear or apprehension lasting between 5 to 30 minutes?",
            "56. Are your panic attacks not caused by the effects of substances or medical conditions like hyperthyroidism?",
            "57. Do you worry or have fears about further episodes of panic attacks occurring?",
            ],
        "Specific Phobia": [
            "58. Do you experience marked and excessive fear triggered by specific objects or situations, such as the sight of blood, heights, or closed spaces?",
            "59. Does the sight or thought of the specific object or situation almost always cause immediate fear?",
            "60. Do you actively avoid the phobic object or situation, or endure it with intense fear or anxiety?",
            "61. Is your fear or anxiety out of proportion to the actual danger posed by the specific object or situation in your sociocultural context?",
            "62. Does your fear, anxiety, or avoidance cause significant distress or impairment in your social, work, or other important areas of functioning?",
            "63. Is your fear out of proportion to the actual danger posed by the object or situation?",
            "64. Do you experience avoidance or intense anxiety when faced with the object or situation?",
        ],
        "Social Anxiety": [
            "65. Do you find yourself markedly avoiding social situations or objects that trigger fear or anxiety?",
            "66. Is your fear or anxiety in social situations out of proportion to the actual threat or situation?",
            "67. Does your fear or anxiety cause significant impairment in your social, occupational, or other important areas of functioning?",
            "68. Do social situations almost always provoke fear or anxiety for you?",
            "69. Do you avoid social situations or endure them with intense fear or anxiety?",
            "70. Do you fear that you will act in a way or show anxiety symptoms that will be negatively evaluated by others?",
            "71. Do you have a persistent fear of being negatively evaluated by others in social situations?",
            "72. Do you experience intense and persistent fear or anxiety in social situations, including conversations?",
            "73. Do your symptoms or the avoidance of social situations cause significant emotional distress?",
            "74. Do you recognize that your symptoms or the avoidance of social situations are excessive or unreasonable?",
            "75. Do your symptoms predominate in social situations or when thinking about them, such as blushing, fear of vomiting, or urgency/fear of micturition or defecation?",
        ],
        "OCD": [
            "76. Do your obsessions or compulsions consume a significant amount of time each day?",
            "77. Are your obsessions or compulsions not influenced by drugs or medical conditions?",
            "78. Do you experience uncontrollable, obsessive thoughts?",
            "79. Have these symptoms been present for at least 3 weeks?",
            ],
        "PTSD": [
            "80. Have you experienced a traumatic event that you find difficult to forget or cope with?",
            "81. Do you avoid reminders or situations related to the traumatic event?",
            "82. Have you noticed changes in your thinking or emotions since the traumatic event?",
            "83. Do you experience increased arousal symptoms, such as being easily startled or having difficulty sleeping?",
            "84. Do you have frequent flashbacks, dreams, or nightmares about the traumatic event?",
            "85. Has the trauma significantly impacted your personal, family, social, educational, or work life?",
            "86. Do your PTSD symptoms cause significant impairment in your daily functioning?",
            ],
        "Gambling Disorder": [
            "87. Do you engage in gambling activities?",
            "88. Do you experience persistent and recurring problems related to gambling, such as needing to gamble with increasing amounts, feeling irritable when trying to stop, making unsuccessful attempts to cut back, or having gambling-related issues in your life?",
            "89. Do you prioritize gambling over other activities or responsibilities?",
            "90. Do you have difficulty controlling or limiting your gambling behavior?",
            "91. Do you continue to gamble despite experiencing financial, relationship, legal, or mental health problems as a result?",
            ],
        "Substance Abuse": [
            "92. Do you use illicit drugs or alcohol?",
            "93. Do you use medication that was not prescribed to you?",
            "94. Have you increased the dosage of substances you use?",
            "95. Do you have an uncontrollable desire to use drugs or a strong addiction to them?",
            "96. Have you neglected social activities due to substance use?",
            "97. Do you continue using substances despite knowing they cause harm?",
            "98. Do you experience withdrawal symptoms when you stop using substances?",
            "99. Has your substance use persisted for 12 months or more?",
            ]
    }
    getcontext().prec = 2
    responses = []
    for heading in query:
        questions = query[heading]
        st.header(heading)
        for i, question in enumerate(questions):
            st.write(question)
            response = st.radio("", ["Yes", "No"], index=1, key=f"{question}{i}")
            responses.append(1 if response == "Yes" else 0)
    
    # Add age as a dropdown with predefined ranges
    age_ranges = ["Under 18", "18-24", "25-34", "35-44", "45-54", "55-64", "65 and over"]
    age = st.selectbox("What is your age range?", age_ranges)
    age_index = age_ranges.index(age)  # Convert age range to a numerical index

    # Add sex/gender as a dropdown
    sex_options = ["Male", "Female"]
    sex = st.selectbox("What is your sex/gender?", sex_options)
    sex_index = sex_options.index(sex)  # Convert sex to a numerical index

    # Append age and sex to responses
    responses.extend([age_index, sex_index])

    if st.button("Submit"):
        # Convert responses to numpy array for prediction
        responses = np.array(responses).reshape(1, -1)
        prediction = model.predict(responses)
        probabilities = model.predict_proba(responses)

       
        st.write("Prediction:")
        output = prediction[0]
        diagnosis_labels = defaultdict(str)
        i = 0
        for labels in query:
            diagnosis_labels[labels] = "Present" if output[i] else "Not Present" 
            i+=1

        # diagnosis_labels = {
        #     "Depression": output[0],
        #     "Schizophrenia": output[1],
        #     "Acute_and_transient_psychotic_disorder": output[2],
        #     "Delusional_Disorder": output[3],
        #     "BiPolar1": output[4],
        #     "BiPolar2": output[5],
        #     "Anxiety": output[6],
        #     "Generalized_Anxiety": output[7],
        #     "Panic_Disorder": output[8],
        #     "Specific_Phobia": output[9],
        #     "Social_Anxiety": output[10],
        #     "OCD": output[11],
        #     "PTSD": output[12],
        #     "Gambling": output[13],
        #     "substance_abuse": output[14]
        # }
        decimals = probabilities
        
        labels = []
        values = []
        i = 0
        for diagnose in diagnosis_labels:
            if diagnosis_labels[diagnose] == "Present":
                labels.append(diagnose)
                values.append(round(decimals[i][0][1]*100,2))
            i += 1
        prob_labels = defaultdict(float)
        for i in range(len(labels)):
            prob_labels[labels[i]] = values[i]
        if len(values) == 0:
            st.header('NO MHD PRESENT ❌')
        else:
            st.header('MHDs DETECTED ✅')
            st.write(diagnosis_labels)
            st.write(prob_labels)  
            
            st.title("Pie Chart Example")

            # Create a pie chart

            fig, ax = plt.subplots()
            wedges, texts, autotexts = ax.pie(
                values,
                labels=labels,
                autopct='%1.1f%%',  # Format of the percentage
                startangle=140,
                colors=plt.get_cmap('tab10').colors
            )

            # Add a legend
            ax.legend(wedges, labels,
                    title="Categories",
                    loc="center left",
                    bbox_to_anchor=(1, 0, 0.5, 1))

            # Add a title
            plt.title('Distribution of Categories')

            # Display the pie chart in Streamlit
            st.pyplot(fig)

if __name__ == "__main__":
    main()
