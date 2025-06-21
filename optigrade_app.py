import streamlit as st
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import google.generativeai as genai
import os
from dotenv import load_dotenv
import streamlit.components.v1 as components

# Load environment variables
load_dotenv()
api_key = os.getenv("API_KEY")

# Initialize session state
if "onboarded" not in st.session_state:
    st.session_state.onboarded = False
if "show_prototype" not in st.session_state:
    st.session_state.show_prototype = False

# ------------------ LOGO ------------------
components.html("""
    <div style="display: flex; justify-content: center; padding: 20px;">
        <svg width="150" height="150" viewBox="0 0 150 150" fill="none" xmlns="http://www.w3.org/2000/svg">
            <rect width="150" height="150" rx="36.875" fill="url(#paint0_linear_826_2220)"/>
            <!-- Inner logo path here -->
            <path d="M64.8293 43.5172...Z" fill="white"/>
            <defs>
                <linearGradient id="paint0_linear_826_2220" x1="-13.75" y1="101.25" x2="149.672" y2="34.2008" gradientUnits="userSpaceOnUse">
                    <stop stop-color="#1C69B2"/>
                    <stop offset="0.677451" stop-color="#611EE8"/>
                </linearGradient>
            </defs>
        </svg>
    </div>
""", height=180)

# ------------------ ONBOARDING ------------------
if not st.session_state.onboarded:
    st.markdown(
        "<h1 style='text-align: center; font-size: 64px; color: white;'>Welcome to OptiGrade!</h1>",
        unsafe_allow_html=True
    )
    st.markdown(
        "<p style='text-align: center; color: white; font-size: 18px;'>Smart academic insights to help you study better and succeed with confidence.</p>",
        unsafe_allow_html=True
    )
    st.markdown("<div style='height: 40px;'></div>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns([3, 4, 3])
    with col2:
        if st.button("📱 View Mobile Prototype", key="view_mobile_button"):
            st.session_state.show_prototype = True
            st.session_state.onboarded = True
            st.rerun()

# ------------------ PROTOTYPE DISPLAY ------------------
if st.session_state.show_prototype:
    st.markdown("### 📱 OptiGrade Mobile Preview")
    components.html("""
        <iframe 
            style="border: none;" 
            width="100%" 
            height="600" 
            src="https://www.figma.com/embed?embed_host=streamlit&url=https://www.figma.com/proto/B2L8DOx0u3xuSWPhKpJpO5/OptiGrade-Mobile-App---EduTech?node-id=810-1023&t=LkykJEw02nadKihK-1&starting-point-node-id=802%3A966&content-scaling=fixed"
            allowfullscreen>
        </iframe>
    """, height=620)

# ------------------ LAUNCH OPTIGRADE TOOLS ------------------
if st.session_state.onboarded:
    st.title("📚 OptiGrade Dashboard")

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "ℹ️ About", 
        "🚀 Features", 
        "🧠 CGPA Predictor", 
        "📘 Study Habit Tracker", 
        "📂 Course Manager"
    ])

    #this is whwre the new one pasted started from that from line 73 to be removed incase

    #This is the normal one before pasting oin line 74
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "ℹ️ About", 
        "🚀 Features", 
        "🧠 CGPA Predictor", 
        "📘 Study Habit Tracker", 
        "📂 Course Manager"
    ])

    with tab1:
        st.subheader("Welcome to OptiGrade")
        st.write("OptiGrade is your intelligent academic assistant, built to help students predict academic performance, monitor study habits, and plan smarter for success.")

    with tab2:
        st.subheader("Key Features")
        st.markdown("""
        - 📊 **Predict Your CGPA** from past grades and course insights  
        - 📘 **Track Study Habits** visually over time  
        - 🗂️ **Manage Courses** in real-time  
        - 🎯 **Optimize Study Hours** based on difficulty and credit load  
        - 🔐 **Privacy-First**: All data is yours, no external storage  
        """)

    with tab3:
        st.subheader("Predict Your CGPA")
        st.write("Enter past academic data and get predictions based on performance, attendance, and course difficulty.")

    with tab4:
        st.subheader("Track Your Learning Habits")
        st.write("Log study hours, analyze consistency, and stay on track with your academic goals.")

    with tab5:
        st.subheader("Course Manager")
        st.write("Add or manage courses for the semester in a clean, interactive layout.")

# You can insert your course input forms, dataframes, and model logic below inside the tab blocks as needed.

# Load environment variables
load_dotenv()

# Get API key from environment variables
api_key = os.getenv("API_KEY")

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
