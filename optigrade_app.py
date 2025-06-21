import streamlit as st
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import google.generativeai as genai
<<<<<<< HEAD
=======
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Get API key from environment variables
api_key = os.getenv("API_KEY")

>>>>>>> edbfa74055a629e0ecea538b2b990cb192e6b67f

# Initialize session state for navigation and data
if 'page' not in st.session_state:
    st.session_state.page = 'Screen 1'
if 'prev_data' not in st.session_state:
    st.session_state.prev_data = []
if 'curr_data' not in st.session_state:
    st.session_state.curr_data = []
if 'user_id' not in st.session_state:
    st.session_state.user_id = 1
if 'api_key' not in st.session_state:
    st.session_state.api_key = ''

# Streamlit app
st.title("OptiGrade")

# Sidebar for API key and user ID
with st.sidebar:
    st.header("Settings")
    st.session_state.user_id = st.number_input("Student ID (1-10)", min_value=1, max_value=10, value=1)

# Navigation
if st.session_state.page == 'Screen 1':
    st.header("Screen 1: Previous Semester Courses")
    st.write("Enter details for 5 courses from the previous semester.")

    # Form for 5 previous semester courses
    with st.form("prev_form"):
        prev_courses = []
        for i in range(5):
            st.subheader(f"Course {i+1}")
            course_id = st.text_input(f"Course ID (e.g., MATH101)", key=f"prev_course_id_{i}")
            grade = st.number_input(f"Grade (0-100)", min_value=0.0, max_value=100.0, step=0.1, key=f"prev_grade_{i}")
            study_hours = st.number_input(f"Study Hours/Week", min_value=0.0, step=0.1, key=f"prev_hours_{i}")
            course_units = st.number_input(f"Course Units (1-6)", min_value=1, max_value=6, step=1, key=f"prev_units_{i}")
            course_difficulty = st.number_input(f"Difficulty (1-5)", min_value=1, max_value=5, step=1, key=f"prev_diff_{i}")
            attendance = st.number_input(f"Attendance (%)", min_value=0.0, max_value=100.0, step=0.1, key=f"prev_att_{i}")
            prev_courses.append({
                'user_id': st.session_state.user_id,
                'semester': 'Previous',
                'course_id': course_id,
                'grade': grade,
                'study_hours': study_hours,
                'course_units': course_units,
                'course_difficulty': course_difficulty,
                'attendance': attendance
            })

        semester_gpa = st.number_input("Semester GPA (0-5)", min_value=0.0, max_value=5.0, step=0.1)
        current_cgpa = st.number_input("Current CGPA (0-5)", min_value=0.0, max_value=5.0, step=0.1)
        learning_style = st.selectbox("Learning Style", ["Visual", "Auditory", "Kinesthetic"])

        submitted = st.form_submit_button("Save and Proceed to Current Semester")
        if submitted:
            for course in prev_courses:
                course['semester_gpa'] = semester_gpa
                course['current_cgpa'] = current_cgpa
                course['learning_style'] = learning_style
            st.session_state.prev_data = prev_courses
            st.session_state.page = 'Screen 2'
            st.rerun()

elif st.session_state.page == 'Screen 2':
    st.header("Screen 2: Current Semester Courses")
    st.write("Enter details for 5 courses for the current semester.")

    # Form for 5 current semester courses
    with st.form("curr_form"):
        curr_courses = []
        for i in range(5):
            st.subheader(f"Course {i+1}")
            course_id = st.text_input(f"Course ID (e.g., STAT101)", key=f"curr_course_id_{i}")
            course_units = st.number_input(f"Course Units (1-6)", min_value=1, max_value=6, step=1, key=f"curr_units_{i}")
            course_difficulty = st.number_input(f"Difficulty (1-5)", min_value=1, max_value=5, step=1, key=f"curr_diff_{i}")
            curr_courses.append({
                'user_id': st.session_state.user_id,
                'semester': 'Current',
                'course_id': course_id,
                'course_units': course_units,
                'course_difficulty': course_difficulty,
                'learning_style': st.session_state.prev_data[0]['learning_style']
            })

        submitted = st.form_submit_button("Generate Results")
        if submitted:
            st.session_state.curr_data = curr_courses
            st.session_state.page = 'Results'
            st.rerun()

elif st.session_state.page == 'Results':
    st.header(f"Results for Student {st.session_state.user_id}")

    # Convert input data to DataFrames
    prev_data = pd.DataFrame(st.session_state.prev_data)
    curr_data = pd.DataFrame(st.session_state.curr_data)

    # Train model
    features = ['grade', 'study_hours', 'course_units', 'course_difficulty', 'attendance', 'current_cgpa']
    X = prev_data[features]
    y = prev_data['semester_gpa']
    model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=50, random_state=42)
    model.fit(X, y)

    # Predict current semester GPA
    user_prev = prev_data
    user_curr = curr_data.copy()
    avg_grade = user_prev['grade'].mean()
    user_curr['estimated_grade'] = avg_grade - (user_curr['course_difficulty'] - user_prev['course_difficulty'].mean()) * 5
    user_curr['study_hours'] = user_curr['course_units'] * 3
    user_curr['attendance'] = user_prev['attendance'].mean()
    user_curr['current_cgpa'] = user_prev['current_cgpa'].iloc[0]
    X_curr = user_curr[features]
    predicted_gpa = model.predict(X_curr).mean()
    st.subheader(f"Predicted Semester GPA: {predicted_gpa:.2f}")

    # Analyze previous GPA causes
    avg_grade = user_prev['grade'].mean()
    avg_hours = user_prev['study_hours'].mean()
    avg_attendance = user_prev['attendance'].mean()
    weak_course = user_prev[user_prev['grade'] == user_prev['grade'].min()]['course_id'].iloc[0]
    weak_grade = user_prev['grade'].min()
    high_difficulty = user_prev[user_prev['course_difficulty'] >= 3]['course_id'].tolist()

    st.subheader(f"Previous GPA Analysis (Semester GPA: {user_prev['semester_gpa'].iloc[0]:.2f})")
    st.write(f"- Average Grade: {avg_grade:.1f}")
    st.write(f"- Average Study Hours: {avg_hours:.1f}/week")
    st.write(f"- Average Attendance: {avg_attendance:.1f}%")
    st.write(f"- Weakest Course: {weak_course} (Grade: {weak_grade:.1f})")
    if high_difficulty:
        st.write(f"- High-Difficulty Courses: {', '.join(high_difficulty)}")
    if avg_hours < 8:
        st.write("- Possible Cause: Low study hours may have reduced performance.")
    if avg_attendance < 85:
        st.write("- Possible Cause: Low attendance may have impacted GPA.")
    if weak_grade < 70:
        st.write("- Possible Cause: Poor performance in specific courses affected GPA.")

    # Optimize study hours
    total_hours_per_week = 40
    user_curr['study_weight'] = user_curr['course_units'] * user_curr['course_difficulty']
    user_curr['allocated_hours'] = (user_curr['study_weight'] / user_curr['study_weight'].sum()) * total_hours_per_week

    st.subheader("Study Plan")
    for _, row in user_curr[['course_id', 'course_units', 'course_difficulty', 'allocated_hours']].iterrows():
        st.write(f"{row['course_id']}: {row['allocated_hours']:.1f} hours/week (Units: {row['course_units']}, Difficulty: {row['course_difficulty']})")

    # Generate recommendations
    st.subheader("Personalized Recommendations")
    if api_key:
        try:
            genai.configure(api_key=api_key)
            weak_course = user_prev[user_prev['grade'] == user_prev['grade'].min()]['course_id'].iloc[0]
            weak_grade = user_prev['grade'].min()
            learning_style = user_curr['learning_style'].iloc[0]
            avg_hours = user_prev['study_hours'].mean()
            avg_attendance = user_prev['attendance'].mean()

            prompt = f"""
            Student {st.session_state.user_id} has a current CGPA of {user_prev['current_cgpa'].iloc[0]:.2f}, semester GPA of {user_prev['semester_gpa'].iloc[0]:.2f} last semester, and was weak in {weak_course} (grade: {weak_grade:.1f}).
            Past study habits: {avg_hours:.1f} hours/week, {avg_attendance:.1f}% attendance.
            Possible GPA causes: {', '.join(['Low study hours' if avg_hours < 8 else '', 'Low attendance' if avg_attendance < 85 else '', f'Poor performance in {weak_course}' if weak_grade < 70 else ''])}.
            They are taking these courses this semester:
            {', '.join(f"{row['course_id']} ({row['course_units']} units, difficulty: {row['course_difficulty']})" for _, row in user_curr.iterrows())}.
            Predicted semester GPA: {predicted_gpa:.2f}. Learning style: {learning_style}.
            Suggest a 1-week study plan, learning resources, and strategies to address past weaknesses (e.g., low grades in {weak_course}).
            Prioritize higher-unit and difficult courses, and tailor to {learning_style} learning style.
            """
            response = genai.GenerativeModel('gemini-1.5-flash').generate_content(prompt)
            st.write(response.text)
        except Exception as e:
            st.error(f"Error generating recommendations: {str(e)}")
    else:
        st.warning("Please set the API_KEY environment variable.")

    # Navigation back
    if st.button("Back to Screen 1"):
        st.session_state.page = 'Screen 1'
        st.rerun()
