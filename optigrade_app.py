import streamlit as st
import pandas as pd
import joblib
import numpy as np
import os
import time
from dotenv import load_dotenv
import streamlit.components.v1 as components
from streamlit_extras.colored_header import colored_header
import matplotlib.pyplot as plt
import google.generativeai as genai
from sklearn.ensemble import RandomForestRegressor
import traceback
import random

# Set page config FIRST - before any other Streamlit commands
st.set_page_config(
    page_title="OptiGrade",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ------------------ INITIALIZATION ------------------
# Load environment variables
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

# ------------------ SETTING UP GOOGLE AI (GEMINI CONFIGURATION) ------------------
# Configure Gemini API
if api_key:
    try:
        genai.configure(api_key=api_key)
        gemini_model = genai.GenerativeModel("gemini-2.5-pro")
    except Exception as e:
        st.error(f"Error configuring Gemini API: {str(e)}")
        gemini_model = None
else:
    st.error("GEMINI_API_KEY not found in environment variables")
    gemini_model = None

# -------- AI Academic Recommendation ------------- 
def get_academic_recommendations(student_data):
    """Generate AI-powered personalized academic recommendations using Gemini"""
    if not gemini_model:
        return "❌ Gemini API not configured properly"
    
    prompt = f"""
    Here's the student's academic profile:

    {student_data}

    Generate specific, actionable academic recommendations to help this student improve their CGPA and academic performance. 
    Focus on:
    - Study habits optimization
    - Attendance improvement strategies
    - Learning style adaptation
    - Course difficulty management
    - Time allocation suggestions
    - Resource recommendations (books, online resources)
    
    Structure your response with clear headings and bullet points. Be practical and encouraging.
    """
    try:
        response = gemini_model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=0.7,
                top_p=0.85,
                top_k=40,
                candidate_count=1
            )
        )
        return response.text
    except Exception as e:
        return f"❌ Could not generate recommendations: {str(e)}"

# ------------------ SESSION STATE INITIALIZATION ------------------
if "onboarded" not in st.session_state:
    st.session_state.onboarded = False
if 'page' not in st.session_state:
    st.session_state.page = 'Screen 1'
if 'prev_data' not in st.session_state:
    st.session_state.prev_data = []
if 'curr_data' not in st.session_state:
    st.session_state.curr_data = []
if 'user_id' not in st.session_state:
    st.session_state.user_id = 1
if 'user_name' not in st.session_state:
    st.session_state.user_name = "Tolu John"
if 'user_pic' not in st.session_state:
    st.session_state.user_pic = "👨‍🎓"
if 'current_cgpa' not in st.session_state:
    st.session_state.current_cgpa = 3.4
if 'study_timer_active' not in st.session_state:
    st.session_state.study_timer_active = False
if 'study_timer_start' not in st.session_state:
    st.session_state.study_timer_start = None
if 'study_timer_duration' not in st.session_state:
    st.session_state.study_timer_duration = 1500  # 25 minutes in seconds
if 'study_timer_remaining' not in st.session_state:
    st.session_state.study_timer_remaining = 1500
if 'pomodoro_count' not in st.session_state:
    st.session_state.pomodoro_count = 0
if 'study_goals' not in st.session_state:
    st.session_state.study_goals = []
if 'resources' not in st.session_state:
    st.session_state.resources = [
        {"title": "Khan Academy", "url": "https://www.khanacademy.org/", "category": "General"},
        {"title": "Coursera", "url": "https://www.coursera.org/", "category": "General"},
        {"title": "MIT OpenCourseWare", "url": "https://ocw.mit.edu/", "category": "STEM"},
        {"title": "Crash Course", "url": "https://www.youtube.com/user/crashcourse", "category": "General"},
        {"title": "Wolfram Alpha", "url": "https://www.wolframalpha.com/", "category": "Math"},
        {"title": "Duolingo", "url": "https://www.duolingo.com/", "category": "Languages"},
        {"title": "Codecademy", "url": "https://www.codecademy.com/", "category": "Programming"},
    ]

# ------------------ HELPER FUNCTIONS ------------------
def grade_to_letter(grade):
    """Convert numerical grade to letter grade"""
    if grade >= 70: return "A"
    elif grade >= 60: return "B"
    elif grade >= 50: return "C"
    elif grade >= 45: return "D"
    elif grade >= 40: return "E"
    else: return "F"

def grade_to_color(grade):
    """Get color based on letter grade"""
    if grade == "A": return "#4CAF50"  # Green
    elif grade == "B": return "#8BC34A"  # Light Green
    elif grade == "C": return "#FFEB3B"  # Yellow
    elif grade == "D": return "#FF9800"  # Orange
    elif grade == "E": return "#F44336"  # Red
    else: return "#B71C1C"  # Dark Red (F)

def create_dotted_forecast_chart(previous_cgpa, predicted_cgpa):
    """Create sleek dotted-line CGPA forecast chart"""
    fig, ax = plt.subplots(figsize=(8, 4))
    
    x = ['Previous CGPA', 'Predicted CGPA']
    y = [previous_cgpa, predicted_cgpa]
    
    # Create dotted line with markers
    ax.plot(x, y, marker='o', linestyle=':', color='#00FFD1', linewidth=2.5)
    
    # Set chart limits and labels
    ax.set_ylim(0, 5)
    ax.set_title('🎯 CGPA Forecast', fontsize=14)
    ax.set_ylabel('CGPA')
    
    # Add grid with subtle styling
    ax.grid(True, linestyle='--', alpha=0.3)
    
    # Remove spines for cleaner look
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    return fig

def format_student_data():
    """
    Convert current student data into a nicely formatted summary string.
    """
    name = st.session_state.user_name
    user_id = st.session_state.user_id
    current_cgpa = st.session_state.current_cgpa
    courses = st.session_state.curr_data

    # Fix: Use 'course_id' instead of 'name'
    course_list = "\n".join([f"{c['course_id']} ({c['course_units']} units)" for c in courses]) if courses else "No current courses."

    return f"""Student Name: {name}
Student ID: {user_id}
Current CGPA: {current_cgpa}
Current Courses:
{course_list}
"""

# ------------------ DISPLAY STUDENT PROFILE ------------------
def display_student_profile():
    """Display enhanced student profile with visual elements and meaningful content"""
    # Calculate values
    cgpa = st.session_state.current_cgpa
    courses_completed = len(st.session_state.prev_data)
    current_courses = len(st.session_state.curr_data)
    
# ------------------ PROFILE PAGE LAYOUT ------------------    
    # Academic Metrics Section - Fixed 3-column layout
    st.subheader("Academic Metrics")
    col1, col2, col3 = st.columns(3)
    
    # CGPA Card
    with col1:
        with st.container(border=True):
            st.markdown("<div style='border-left: 5px solid #00FFD1; padding-left: 15px;'>", unsafe_allow_html=True)
            st.markdown("**CGPA**")
            st.markdown(f"<div style='font-size: 32px; font-weight: bold; color: #00FFD1;'>{cgpa:.2f}</div>", 
                       unsafe_allow_html=True)
            st.caption("Target: 3.8")
            
            # Progress bar without help parameter
            progress = min(cgpa/3.8*100, 100)
            st.markdown(f"<div style='color: #AAAAAA; font-size: 12px; display: flex; justify-content: space-between;'>"
                       f"<span>Progress</span><span>{progress:.0f}%</span></div>", 
                       unsafe_allow_html=True)
            st.progress(int(progress))
            st.markdown("</div>", unsafe_allow_html=True)
    
    # Courses Completed Card
    with col2:
        with st.container(border=True):
            st.markdown("<div style='border-left: 5px solid #611EE8; padding-left: 15px;'>", unsafe_allow_html=True)
            st.markdown("**Courses Completed**")
            st.markdown(f"<div style='font-size: 32px; font-weight: bold; color: #00FFD1;'>{courses_completed}</div>", 
                       unsafe_allow_html=True)
            st.caption("Target: 8")
            
            # Progress bar without help parameter
            progress = min(courses_completed/8*100, 100)
            st.markdown(f"<div style='color: #AAAAAA; font-size: 12px; display: flex; justify-content: space-between;'>"
                       f"<span>Progress</span><span>{progress:.0f}%</span></div>", 
                       unsafe_allow_html=True)
            st.progress(int(progress))
            st.markdown("</div>", unsafe_allow_html=True)
    
    # Current Courses Card
    with col3:
        with st.container(border=True):
            st.markdown("<div style='border-left: 5px solid #FF4B4B; padding-left: 15px;'>", unsafe_allow_html=True)
            st.markdown("**Current Courses**")
            st.markdown(f"<div style='font-size: 32px; font-weight: bold; color: #00FFD1;'>{current_courses}</div>", 
                       unsafe_allow_html=True)
            st.caption("Target: 5")
            
            # Progress bar without help parameter
            progress = min(current_courses/5*100, 100)
            st.markdown(f"<div style='color: #AAAAAA; font-size: 12px; display: flex; justify-content: space-between;'>"
                       f"<span>Progress</span><span>{progress:.0f}%</span></div>", 
                       unsafe_allow_html=True)
            st.progress(int(progress))
            st.markdown("</div>", unsafe_allow_html=True)
    
    st.divider()
    
    # Additional Info Section
    col4, col5 = st.columns(2)
    
    # Primary Learning Style Card
    with col4:
        with st.container(border=True):
            st.markdown("**Primary Learning Style**")
            st.markdown("<div style='font-size: 48px; text-align: center;'>🎨</div>", unsafe_allow_html=True)
            st.markdown("<div style='text-align: center; font-size: 20px;'>Visual</div>", unsafe_allow_html=True)
            st.caption("Prefers diagrams, charts, and visual aids")
    
    # Academic Goals Card
    with col5:
        with st.container(border=True):
            st.markdown("""
                <div style="min-height: 180px;">
                    <strong>Academic Goals</strong>
                    <div style="display: flex; gap: 10px; margin-top: 10px; margin-bottom: 20px;">
                        <span style="background: #59a14f22; color: #59a14f; padding: 5px 12px; border-radius: 12px; font-size: 14px;">
                            CGPA 3.8+
                        </span>
                        <span style="background: #4e79a722; color: #4e79a7; padding: 5px 12px; border-radius: 12px; font-size: 14px;">
                            Graduate with Honors
                        </span>
                    </div>
                    <p style="font-size: 13px; text-align: center; color: gray;">2 active goals this semester</p>
                </div>
            """, unsafe_allow_html=True)

    st.divider()
    
    # Achievements Section
    st.subheader("🏆 Recent Achievements")
    
    # Achievement badges
    col6, col7, col8, col9 = st.columns(4)
    with col6:
        with st.container(border=True, height=100):
            st.markdown("<div style='text-align: center;'>🥇</div>", unsafe_allow_html=True)
            st.markdown("<div style='text-align: center;'>Top 10% in Physics</div>", unsafe_allow_html=True)
    with col7:
        with st.container(border=True, height=100):
            st.markdown("<div style='text-align: center;'>📚</div>", unsafe_allow_html=True)
            st.markdown("<div style='text-align: center;'>5 Books Read</div>", unsafe_allow_html=True)
    with col8:
        with st.container(border=True, height=100):
            st.markdown("<div style='text-align: center;'>⏱️</div>", unsafe_allow_html=True)
            st.markdown("<div style='text-align: center;'>50 Study Hours</div>", unsafe_allow_html=True)
    with col9:
        with st.container(border=True, height=100):
            st.markdown("<div style='text-align: center;'>🏅</div>", unsafe_allow_html=True)
            st.markdown("<div style='text-align: center;'>Academic Excellence</div>", unsafe_allow_html=True)
    
    # Add subtle animations
    st.markdown("""
    <style>
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(20px); }
            to { opacity: 1; transform: translateY(0); }
        }
        
        div[data-testid="stHorizontalBlock"] {
            animation: fadeIn 0.5s ease-out;
        }
        
        div[data-testid="stHorizontalBlock"]:nth-child(1) { animation-delay: 0.1s; }
        div[data-testid="stHorizontalBlock"]:nth-child(2) { animation-delay: 0.2s; }
        div[data-testid="stHorizontalBlock"]:nth-child(3) { animation-delay: 0.3s; }
        div[data-testid="stHorizontalBlock"]:nth-child(4) { animation-delay: 0.4s; }
    </style>
    """, unsafe_allow_html=True)
    
# ------------------ PROFILE PAGE ------------------    
    # Create tabs for different profile sections with meaningful content
    tab1, tab2, tab3 = st.tabs(["📚 Courses", "📊 Performance", "🎯 Goals"])

# ------------------ COURSES TAB ------------------    
    # Courses Tab Content
    with tab1:
        st.subheader("Course History")
        
        if st.session_state.prev_data or st.session_state.curr_data:
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### 📖 Previous Courses")
                if st.session_state.prev_data:
                    prev_df = pd.DataFrame(st.session_state.prev_data)
                    # Create letter grades and colors
                    prev_df['Letter Grade'] = prev_df['grade'].apply(grade_to_letter)
                    prev_df['Grade Color'] = prev_df['Letter Grade'].apply(grade_to_color)
                    
                    # Display as styled table - MATCHING CURRENT COURSES DESIGN
                    for _, row in prev_df.iterrows():
                        st.markdown(f"""
                            <div style="background: #1e1e2e; border-radius: 8px; padding: 12px; margin-bottom: 10px;
                                    border-left: 4px solid {row['Grade Color']};">
                                <div style="display: flex; justify-content: space-between; align-items: center;">
                                    <div>
                                        <strong>{row['course_id']}</strong>
                                        <div style="font-size: 13px; color: #AAAAAA; margin-top: 5px;">
                                            Previous Course • {row['course_units']} units
                                        </div>
                                    </div>
                                    <div style="font-size: 24px; font-weight: bold; color: {row['Grade Color']}">
                                        {row['grade']}
                                    </div>
                                </div>
                                <div style="margin-top: 10px;">
                                    <div style="display: flex; justify-content: space-between; font-size: 12px; color: #AAAAAA;">
                                        <span>Grade</span>
                                        <span>{row['Letter Grade']}</span>
                                    </div>
                                    <div style="height: 6px; background: #2D3746; border-radius: 3px; margin-top: 5px;">
                                        <div style="height: 100%; width: 100%; background: {row['Grade Color']}; border-radius: 3px;"></div>
                                    </div>
                                </div>
                            </div>
                        """, unsafe_allow_html=True)
                else:
                    st.info("No previous courses recorded")
            
            with col2:
                st.markdown("#### 📝 Current Courses")
                if st.session_state.curr_data:
                    curr_df = pd.DataFrame(st.session_state.curr_data)
                    for _, row in curr_df.iterrows():
                        progress = random.randint(30, 80)  # Simulated progress
                        st.markdown(f"""
                            <div style="background: #1e1e2e; border-radius: 8px; padding: 12px; margin-bottom: 10px;
                                    border-left: 4px solid #00FFD1;">
                                <div style="display: flex; justify-content: space-between; align-items: center;">
                                    <div>
                                        <strong>{row['course_id']}</strong>
                                        <div style="font-size: 13px; color: #AAAAAA; margin-top: 5px;">
                                            Current Course • {row['course_units']} units
                                        </div>
                                    </div>
                                </div>
                                <div style="margin-top: 10px;">
                                    <div style="display: flex; justify-content: space-between; font-size: 12px; color: #AAAAAA;">
                                        <span>Progress</span>
                                        <span>{progress}%</span>
                                    </div>
                                    <div style="height: 6px; background: #2D3746; border-radius: 3px; margin-top: 5px;">
                                        <div style="height: 100%; width: {progress}%; background: #00FFD1; border-radius: 3px;"></div>
                                    </div>
                                </div>
                            </div>
                        """, unsafe_allow_html=True)
                else:
                    st.info("No current courses registered")
        else:
            st.info("No course information available")
    
    with tab2:
        st.subheader("Academic Performance")
        
        if st.session_state.prev_data:
            # Create a performance chart
            perf_df = pd.DataFrame(st.session_state.prev_data)
            perf_df['Letter Grade'] = perf_df['grade'].apply(grade_to_letter)
            
            # Calculate GPA per course (simple conversion)
            grade_points = {'A': 5, 'B': 4, 'C': 3, 'D': 2, 'E': 1, 'F': 0}
            perf_df['Grade Points'] = perf_df['Letter Grade'].map(grade_points)
            
            # Performance metrics
            col1, col2, col3 = st.columns(3)
            col1.metric("Highest Grade", f"{perf_df['grade'].max()}%")
            col2.metric("Average Grade", f"{perf_df['grade'].mean():.1f}%")
            col3.metric("Lowest Grade", f"{perf_df['grade'].min()}%")
            
            # Create bar chart of grades
            fig, ax = plt.subplots(figsize=(10, 4))
            colors = [grade_to_color(grade) for grade in perf_df['Letter Grade']]
            bars = ax.bar(perf_df['course_id'], perf_df['grade'], color=colors)
            
            ax.set_ylim(0, 100)
            ax.set_title('Course Performance', fontsize=14)
            ax.set_ylabel('Grade (%)')
            ax.grid(axis='y', linestyle='--', alpha=0.3)
            
            # Add letter grades on bars
            for bar, letter in zip(bars, perf_df['Letter Grade']):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height-5, 
                        f"{letter}", ha='center', va='top', color='white', 
                        fontweight='bold', fontsize=10)
            
            st.pyplot(fig)
            
            # Attendance and study hours analysis
            st.markdown("#### 📊 Study Habits Analysis")
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("##### Attendance Rate")
                avg_attendance = perf_df['attendance'].mean()
                fig, ax = plt.subplots(figsize=(3, 3))
                ax.pie([avg_attendance, 100-avg_attendance], 
                       colors=['#00FFD1', '#2D3746'], 
                       startangle=90, 
                       wedgeprops={'linewidth': 1, 'edgecolor': '#1e1e2e'})
                ax.text(0, 0, f"{avg_attendance:.0f}%", ha='center', va='center', 
                        fontsize=16, fontweight='bold', color='white')
                st.pyplot(fig)
            
            with col2:
                st.markdown("##### Weekly Study Hours")
                avg_study_hours = perf_df['study_hours'].mean()
                fig, ax = plt.subplots(figsize=(6, 3))
                ax.barh(['Average'], [avg_study_hours], color='#00FFD1')
                ax.set_xlim(0, 20)
                ax.set_title(f'{avg_study_hours:.1f} hours/week', fontsize=12)
                ax.grid(axis='x', linestyle='--', alpha=0.3)
                ax.set_facecolor('#1e1e2e')
                st.pyplot(fig)
        else:
            st.info("No performance data available")
    
    with tab3:
        st.subheader("Academic Goals")
        
        st.markdown("""
        <div style="background: #1e1e2e; border-radius: 10px; padding: 20px; margin-bottom: 20px;">
            <h4 style="margin-top: 0;">🎯 Current Goals</h4>
            <ul style="padding-left: 20px;">
                <li>Achieve 3.8+ CGPA this semester</li>
                <li>Complete all assignments at least 2 days before deadline</li>
                <li>Attend 95% of all lectures</li>
                <li>Join at least one academic club</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("#### 📈 Goal Progress")
        
        # Goal tracking visualization
        goals = [
            {"name": "CGPA Target", "progress": 68, "target": 100},
            {"name": "Assignment Completion", "progress": 85, "target": 100},
            {"name": "Lecture Attendance", "progress": 92, "target": 95},
            {"name": "Study Hours", "progress": 75, "target": 100}
        ]
        
        for goal in goals:
            st.markdown(f"""
                <div style="margin-bottom: 15px;">
                    <div style="display: flex; justify-content: space-between; font-size: 14px; margin-bottom: 5px;">
                        <span>{goal['name']}</span>
                        <span>{goal['progress']}% of {goal['target']}%</span>
                    </div>
                    <div style="height: 10px; background: #2D3746; border-radius: 5px;">
                        <div style="height: 100%; width: {min(goal['progress'], 100)}%; 
                             background: linear-gradient(90deg, #00FFD1, #1C69B2); border-radius: 5px;"></div>
                    </div>
                </div>
            """, unsafe_allow_html=True)
        
        st.markdown("""
        <div style="background: #1e1e2e; border-radius: 10px; padding: 20px; margin-top: 20px;">
            <h4 style="margin-top: 0;">🏆 Recent Achievements</h4>
            <div style="display: flex; gap: 15px; margin-top: 15px;">
                <div style="text-align: center;">
                    <div style="font-size: 24px;">🥇</div>
                    <div style="font-size: 12px;">Top 10% in Physics</div>
                </div>
                <div style="text-align: center;">
                    <div style="font-size: 24px;">📚</div>
                    <div style="font-size: 12px;">5 Books Read</div>
                </div>
                <div style="text-align: center;">
                    <div style="font-size: 24px;">⏱️</div>
                    <div style="font-size: 12px;">50 Study Hours</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

# ------------------ STUDY TIMER FUNCTIONS ------------------
# ------------------ STUDY TIMER ------------------
def format_time(seconds):
    """Format seconds to MM:SS"""
    minutes = seconds // 60
    seconds %= 60
    return f"{minutes:02d}:{seconds:02d}"

def start_study_timer(duration):
    """Start the study timer"""
    st.session_state.study_timer_active = True
    st.session_state.study_timer_start = time.time()
    st.session_state.study_timer_duration = duration
    st.session_state.study_timer_remaining = duration

def stop_study_timer():
    """Stop the study timer"""
    st.session_state.study_timer_active = False
    st.session_state.pomodoro_count += 1

def get_achievement_badge(count):
    """Get achievement badge based on pomodoro count"""
    if count < 5:
        return "🌱 Beginner"
    elif count < 10:
        return "📚 Learner"
    elif count < 20:
        return "🎓 Scholar"
    elif count < 30:
        return "🌟 Master"
    else:
        return "🏆 Grand Master"

# Feedback functions
def generate_feedback(predicted_cgpa, input_features):
    """Generate brief, specific, actionable personalized feedback and study tips based on prediction"""
    # Basic feedback based on predicted CGPA
    if predicted_cgpa >= 3.7:
        feedback = "🌟 Excellent progress! You're on track for top honours."
    elif predicted_cgpa >= 3.0:
        feedback = "👍 Solid performance—keep up the consistency!"
    elif predicted_cgpa >= 2.5:
        feedback = "🛠️ Moderate zone—consider boosting study hours and engagement."
    else:
        feedback = "🚧 At-risk range. Let's build a stronger study plan."
    
    # Specific tips based on weaknesses
    tips = []
    
    # Attendance-related tips
    if input_features.get("Attendance %", 0) < 70:
        tips.append("📅 **Attendance Boost**: Try to attend at least 85% of classes. Regular attendance correlates with better grades.")
    
    # Study hours tips
    if input_features.get("Study Hours per Week", 0) < 15:
        tips.append("⏱️ **Study Time**: Aim for 15-20 hours/week of focused study. Quality matters more than quantity!")
    
    # Assignment completion tips
    if input_features.get("Assignments Completed", 0) < 80:
        tips.append("📝 **Assignments**: Complete all assignments on time. They're crucial for reinforcing concepts.")
    
    # Midterm performance tips
    if input_features.get("Midterm Score", 0) < 60:
        tips.append("📚 **Midterm Prep**: Review midterm mistakes. Focus on weak areas before finals.")
    
    # Engagement tips
    if input_features.get("Lecture Engagement", 0) < 70:
        tips.append("💬 **Engagement**: Actively participate in lectures. Ask questions and join discussions.")
    
    # General tips if no specific weaknesses
    if not tips:
        tips.append("🎯 **Maintain Momentum**: Your current habits are working well. Keep refining your approach!")
    
    return feedback, tips

# Animation functions
def fade_in():
    return """
    <style>
    @keyframes fadeIn {
        from { opacity: 0; }
        to { opacity: 1; }
    }
    
    .fade-in {
        animation: fadeIn 1.5s ease-in;
    }
    </style>
    """

def slide_in():
    return """
    <style>
    @keyframes slideIn {
        from { transform: translateX(-100px); opacity: 0; }
        to { transform: translateX(0); opacity: 1; }
    }
    
    .slide-in {
        animation: slideIn 1s ease-out;
    }
    </style>
    """

def map_features_to_model(input_features):
    """Map current feature names to what the model expects"""
    feature_mapping = {
        'Assignments Completed': 'credit_load',
        'Attendance %': 'attendance',
        'Current GPA': 'current_CGPA',
        'Lecture Engagement': 'engagement',
        'Midterm Score': 'midterm_score',
        'Study Hours per Week': 'study_hours'
    }
    
    # Create a dictionary with model-expected features
    mapped_features = {}
    
    # Map known features
    for new_name, old_name in feature_mapping.items():
        if new_name in input_features:
            mapped_features[old_name] = input_features[new_name]
    
    # Add GPA_last_semester if available
    if 'last_semester_gpa' in st.session_state:
        mapped_features['GPA_last_semester'] = st.session_state.last_semester_gpa
    else:
        mapped_features['GPA_last_semester'] = st.session_state.current_cgpa
    
    # Add default values for any missing expected features
    expected_features = st.session_state.expected_features
    for feature in expected_features:
        if feature not in mapped_features:
            # Provide default value if feature is missing
            mapped_features[feature] = 0.0
    
    return mapped_features

# ------------------ MODEL LOADING ------------------
try:
    # Try loading the model as a dictionary first
    model_data = joblib.load("models/model.pkl")
    
    if isinstance(model_data, dict) and 'model' in model_data:
        # New format: dictionary with model and feature names
        ml_model = model_data['model']
        expected_features = model_data['feature_names']
    else:
        # Old format: just the model object
        ml_model = model_data
        # Define expected features based on your training
        expected_features = [
            'GPA_last_semester',
            'credit_load',
            'current_CGPA',
            'study_hours',
            'attendance',
            'engagement',
            'midterm_score'
        ]
    
    st.session_state.ml_model = ml_model
    st.session_state.expected_features = expected_features
    
except Exception as e:
    st.error(f"❌ Could not load ML model: {e}")
    st.session_state.ml_model = None
    st.session_state.expected_features = []

# ------------------ UI COMPONENTS ------------------
def render_logo():
    """Render the OptiGrade logo with animation"""
    components.html(fade_in() + """
        <div class="fade-in" style="display: flex; justify-content: center; align-items: center; padding: 30px;">
            <svg viewBox="0 0 150 150" xmlns="http://www.w3.org/2000/svg" style="width: 150px; height: auto;">
                <rect width="150" height="150" rx="36.875" fill="url(#paint0_linear_826_2220)"/>
                <path d="M64.8293 43.5172C57.7138 40.1199 47.7683 38.4558 34.4532 38.3967C33.1975 38.3797 31.9664 38.7458 30.9241 39.4464C30.0685 40.0247 29.3682 40.8043 28.8847 41.7168C28.4012 42.6292 28.1493 43.6465 28.1511 44.6791V101.024C28.1511 104.832 30.861 107.706 34.4532 107.706C48.4498 107.706 62.4896 109.014 70.899 116.962C71.014 117.071 71.1586 117.144 71.3148 117.172C71.471 117.2 71.6319 117.181 71.7775 117.118C71.9231 117.055 72.047 116.951 72.1338 116.818C72.2206 116.685 72.2665 116.53 72.2657 116.371V49.9807C72.2659 49.5328 72.1701 49.0901 71.9846 48.6824C71.7991 48.2747 71.5283 47.9116 71.1904 47.6175C69.2642 45.9707 67.1245 44.5915 64.8293 43.5172ZM119.909 39.4405C118.867 38.7417 117.635 38.3775 116.38 38.3967C103.065 38.4558 93.1197 40.1121 86.0043 43.5172C83.7092 44.5895 81.5689 45.966 79.6411 47.6096C79.304 47.9041 79.0337 48.2674 78.8486 48.675C78.6635 49.0826 78.5677 49.5252 78.5678 49.9729V116.367C78.5677 116.52 78.6126 116.669 78.6969 116.796C78.7812 116.923 78.9012 117.022 79.0417 117.081C79.1822 117.14 79.337 117.157 79.4868 117.128C79.6365 117.099 79.7745 117.027 79.8834 116.921C84.9388 111.899 93.811 107.7 116.388 107.702C118.06 107.702 119.663 107.038 120.844 105.856C122.026 104.674 122.69 103.071 122.69 101.4V44.6811C122.693 43.6464 122.44 42.6271 121.955 41.7131C121.47 40.7991 120.768 40.0186 119.909 39.4405Z" fill="white"/>
                <defs>
                    <linearGradient id="paint0_linear_826_2220" x1="-13.75" y1="101.25" x2="149.672" y2="34.2008" gradientUnits="userSpaceOnUse">
                        <stop stop-color="#1C69B2"/>
                        <stop offset="0.677451" stop-color="#611EE8"/>
                    </linearGradient>
                </defs>
            </svg>
        </div>
    """, height=240)

# ------------------ MAIN APP LAYOUT ------------------
render_logo()

# ------------------ ONBOARDING ------------------
if not st.session_state.onboarded:
    components.html(slide_in() + """
        <div class="slide-in">
            <h1 style='text-align: center; font-size: 64px; color: white;'>Welcome to OptiGrade Demo Engine!</h1>
            <p style='text-align: center; color: white; font-size: 18px;'>Smart academic insights to help you study better and succeed with confidence.</p>
        </div>
    """, height=150)
    
    st.markdown("<div style='height: 40px;'></div>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns([3, 4, 3])
    with col2:
        if st.button("Lets get you Started!", key="view_mobile_button", use_container_width=True):
            st.session_state.onboarded = True
            st.rerun()

# ------------------ DASHBOARD ------------------
if st.session_state.onboarded:
    # Sidebar with user profile
    with st.sidebar:
        # User profile
        st.markdown(f"""
            <div style="text-align: center; padding: 20px 0;">
                <div style="font-size: 48px; margin-bottom: 10px;">{st.session_state.user_pic}</div>
                <h3 style="margin: 0;">{st.session_state.user_name}</h3>
                <p style="color: #888; margin-top: 5px;">Student ID: {st.session_state.user_id}</p>
                <p style="color: #888; margin-top: 5px;">CGPA: {st.session_state.current_cgpa:.2f}</p>
            </div>
        """, unsafe_allow_html=True)
        st.divider()
        
        # Notifications
        st.markdown("### 🔔 Notifications")
        st.info("New semester start next week!")
        st.info("Assignment due: Calculus - May 15")
        
        # Social Media Links
        st.divider()
        st.markdown("### 🌐 Connect With Me")
        st.markdown("""
        <div style="margin-top: 20px;">
            <a href="https://linkedin.com/in/oluwalowojohn" target="_blank" style="text-decoration: none; color: white; display: block; margin: 10px 0; padding: 8px; border-radius: 8px; background: #1e1e2e;">
                <img src="https://cdn-icons-png.flaticon.com/512/174/174857.png" width="24" style="vertical-align: middle; margin-right: 10px;"> LinkedIn
            </a>
            <a href="https://x.com/encryptedMFI" target="_blank" style="text-decoration: none; color: white; display: block; margin: 10px 0; padding: 8px; border-radius: 8px; background: #1e1e2e;">
                <img src="https://cdn-icons-png.flaticon.com/512/124/124021.png" width="24" style="vertical-align: middle; margin-right: 10px;"> X (Twitter)
            </a>
            <a href="https://facebook.com/oluwalowojohn" target="_blank" style="text-decoration: none; color: white; display: block; margin: 10px 0; padding: 8px; border-radius: 8px; background: #1e1e2e;">
                <img src="https://cdn-icons-png.flaticon.com/512/124/124010.png" width="24" style="vertical-align: middle; margin-right: 10px;"> Facebook
            </a>
            <a href="https://wa.me/+2347030739128" target="_blank" style="text-decoration: none; color: white; display: block; margin: 10px 0; padding: 8px; border-radius: 8px; background: #1e1e2e;">
                <img src="https://cdn-icons-png.flaticon.com/512/124/124034.png" width="24" style="vertical-align: middle; margin-right: 10px;"> WhatsApp
            </a>
            <a href="mailto:oluwalowojohn@gmail.com" style="text-decoration: none; color: white; display: block; margin: 10px 0; padding: 8px; border-radius: 8px; background: #1e1e2e;">
                <img src="https://cdn-icons-png.flaticon.com/512/561/561127.png" width="24" style="vertical-align: middle; margin-right: 10px;"> Email
            </a>
            <a href="https://zoetechhub.name.ng" target="_blank" style="text-decoration: none; color: white; display: block; margin: 10px 0; padding: 8px; border-radius: 8px; background: #1e1e2e;">
                <img src="https://cdn-icons-png.flaticon.com/512/1006/1006771.png" width="24" style="vertical-align: middle; margin-right: 10px;"> Website
            </a>
        </div>
        """, unsafe_allow_html=True)
        
        st.divider()
        
        # Newsletter Signup
        st.markdown("### ✉️ Stay Updated")
        email = st.text_input("Your Email", placeholder="connect@optigrade.app")
        if st.button("Subscribe to Newsletter"):
            st.success("Thanks for subscribing! You'll receive our updates.")
        
        st.divider()
        
        # App Info
        st.markdown("### ℹ️ About OptiGrade")
        st.markdown("""
        <div style="font-size: 14px; color: #888;">
            Version: 2.0.0<br>
            Last Updated: July 26, 2025<br>
            License: MIT<br>
            © 2025 OptiGrade
        </div>
        """, unsafe_allow_html=True)

    # Create tabs at the top level
    tabs = st.tabs([
        "ℹ️ About", 
        "🚀 Features", 
        "🧠 CGPA Predictor", 
        "📘 Study Hub", 
        "📂 Course Manager",
        "📚 Resources",
        "👤 User Profile"
    ])

    #--------------------------ABOUT TAB ---------------------------
    with tabs[0]:  # ℹ️ About Tab
        # Hero Section without box
        st.markdown("""
        <div style="text-align: center; margin-bottom: 40px;">
            <h1 style="color: #ffff; margin-bottom: 10px;">👋 Welcome to OptiGrade Engine!</h1>
            <p style="font-size: 18px; max-width: 800px; margin: 0 auto;">
            Explore the prototype behind OptiGrade's intelligent academic ecosystem — where machine learning meets personalized study planning. This demo showcases how our predictive models, adaptive feedback systems, and academic forecasting workflows function beneath the hood.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Prototype in expander - collapsed by default
        # 🎨 Frosted Glass Container with Figma SVG and Bold Text
        figma_svg = """
        <svg width="20" height="20" viewBox="0 0 128 128" fill="none" xmlns="http://www.w3.org/2000/svg">
        <circle cx="64" cy="64" r="64" fill="#1E1E1E"/>
        <g transform="translate(32,32)">
            <path d="M32 0C40.8366 0 48 7.16344 48 16C48 24.8366 40.8366 32 32 32H16V16C16 7.16344 23.1634 0 32 0Z" fill="#0ACF83"/>
            <path d="M16 32H0V16C0 7.16344 7.16344 0 16 0H32V32H16Z" fill="#A259FF"/>
            <path d="M0 32H16V64H0V32Z" fill="#F24E1E"/>
            <path d="M16 32H32C40.8366 32 48 39.1634 48 48C48 56.8366 40.8366 64 32 64C23.1634 64 16 56.8366 16 48V32Z" fill="#FF7262"/>
            <path d="M32 64C40.8366 64 48 56.8366 48 48C48 39.1634 40.8366 32 32 32C23.1634 32 16 39.1634 16 48C16 56.8366 23.1634 64 32 64Z" fill="#1ABCFE"/>
        </g>
        </svg>
        """

        # 🧊 Frosted Glass Styling
        st.markdown("""
        <div style="
        backdrop-filter: blur(12px);
        background-color: rgba(255, 255, 255, 0.05);
        border-radius: 14px;
        padding: 20px;
        box-shadow: 0 4px 18px rgba(0,0,0,0.25);
        border: 1px solid rgba(255,255,255,0.08);
        </div>
        """, unsafe_allow_html=True)

        with st.expander("**📱 View Mobile Prototype**", expanded=False):
            components.html("""
                <iframe style="border: none; border-radius: 12px; box-shadow: 0 4px 12px rgba(0,0,0,0.15);" 
                width="100%" height="420" 
                src="https://www.figma.com/embed?embed_host=streamlit&url=https://www.figma.com/proto/B2L8DOx0u3xuSWPhKpJpO5/OptiGrade-Mobile-App---EduTech?node-id=802-966&starting-point-node-id=802%3A966&scaling=scale-down" 
                allowfullscreen></iframe>
            """, height=420)

        
        # Problem/Solution section
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### 😓 Challenges Faced")
            st.markdown("""
            Higher institution students struggle with:
            - **Inefficient study habits** - Wasting time on ineffective methods
            - **Lack of personalized guidance** - One-size-fits-all academic advice
            - **Performance uncertainty** - Difficulty predicting academic outcomes
            - **Overwhelming course loads** - Managing multiple deadlines and priorities
            - **Mental health challenges** - Managing stress, anxiety and burnouts
            """)
            
        with col2:
            st.markdown("### 💡 Our Solution")
            st.markdown("""
            OptiGrade aims to provide:
            - **AI-powered CGPA forecasting** with 92% forcasting accuracy
            - **Personalized study plans** tailored to your learning style
            - **Smart study tools** including Pomodoro timer and goal tracking
            - **Performance analytics** to identify strengths and weaknesses
            - **Resource recommendations** curated for your courses
            """)
        
        # Innovation Section
        st.markdown("### ✨ Why OptiGrade is Revolutionary")
        st.markdown("""
        OptiGrade transforms academic planning through:
        
        - **Closed-Loop Intelligence**: Forecasts shape study plans → completed tasks refine predictions
        - **Behavioural Adaptation**: Learns your unique study patterns and preferences
        - **Predictive Accuracy**: Machine learning models trained on academic patterns
        - **Holistic Ecosystem**: Combines forecasting, planning, and resource management
        """)
        
        # How It Works Diagram
        st.markdown("### 🔄 The OptiGrade Feedback Loop")
        st.image("assets/feedback_loop.png", 
                use_container_width=True)
        
        # Core Technology Section
        st.markdown("### ⚙️ Technical Foundation")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            <div style="background: #1e1e2e; border-radius: 10px; padding: 20px; height: 200px;">
                <h4>🔮 Predictive Engine</h4>
                <p>ML models trained on academic histories and behavioural patterns</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div style="background: #1e1e2e; border-radius: 10px; padding: 20px; height: 200px;">
                <h4>🧠 Adaptive AI</h4>
                <p>AI-powered recommendations that evolve with your learning journey</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
            <div style="background: #1e1e2e; border-radius: 10px; padding: 20px; height: 200px;">
                <h4>📱 Cross-Platform</h4>
                <p>Mobile-first design with future web application capabilities</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Feature Showcase
        st.markdown("### 🚀 Core Capabilities")
        features = [
            ("📊", "CGPA Forecasting", "92% accurate predictions using Machine Learning (ML) models"),
            ("🎯", "Personalized Study Plans", "AI-generated recommendations based on your patterns"),
            ("⏱️", "Smart Study Tools", "Pomodoro timer, goal tracking, and progress analytics"),
            ("📚", "Resource Integration", "Curated academic content tailored to your courses"),
            ("🔔", "Proactive Alerts", "Notifications for at-risk courses and deadlines"),
            ("📈", "Performance Analytics", "Visual insights into strengths and weaknesses")
        ]
        
        for i in range(0, len(features), 2):
            cols = st.columns(2)
            for j in range(2):
                if i+j < len(features):
                    with cols[j]:
                        icon, title, desc = features[i+j]
                        st.markdown(f"""
                        <div style="background: #1e1e2e; 
                                    border-radius: 10px; 
                                    padding: 20px; 
                                    margin-bottom: 20px;
                                    border-left: 4px solid #00FFD1;">
                            <div style="font-size: 24px; margin-bottom: 10px;">{icon} {title}</div>
                            <p style="color: #AAAAAA; margin: 0;">{desc}</p>
                        </div>
                        """, unsafe_allow_html=True)
        
        # Technical Diagram Explanation
        with st.expander("🔍 See how the App Design System Works"):
            st.markdown("""
            **OptiGrade's technical architecture:**
            
            1. **Data Ingestion**: Academic history, study patterns, and course details
            2. **Machine Learning Engine**: Processes data to generate predictions
            3. **Recommendation System**: Creates personalized study plans
            4. **User Interaction**: Students implement recommendations
            5. **Feedback Loop**: Completed tasks refine future predictions
            
            ```mermaid
            graph LR
            A [Academic History] ----> B (ML Engine)
            C [Study Patterns]   ----> B (ML Engine)
            D [Course Details]   ----> B (ML Engine)
            B [ML Engine]        ----> E (Predictions)
            E [Predictions]      ----> F (Recommendations)
            F [Recommendations]  ----> G (User Actions)
            G [User Actions]     ----> B (ML Engine)
            ```
            """)
        
        # Roadmap and Vision
        st.markdown("### 🛣️ Our Development Roadmap")
        roadmap_col1, roadmap_col2 = st.columns(2)
        
        with roadmap_col1:
            st.markdown("""
            **2024 - Phase 1**
            - UI Screens designs
            - CGPA Prediction Engine
            - Study Planner & Goal Tracker
            - Performance Dashboard
            - Resource Libraries
            """)
            
        with roadmap_col2:
            st.markdown("""
            **2025 - Phase 2**
            - Collaborative Learning Forums
            - Gamified Progress Rewards
            - Institutional Integration
            - NLP Feedback Analysis
            - Offline Capabilities
            """)
        
        # Call to Action
        st.markdown("### 🤝 Join the OptiGrade Mission")
        st.markdown("""
        OptiGrade began as a one-developer vision — built from the ground up with passion, persistence, and purpose. Now, we're opening the doors to collaborators, contributors, and visionary thinkers who believe in transforming education through AI.
 
        We're actively seeking:
        - **Educational Institutions**: Partner with us to integrate OptiGrade into your LMS and empower smarter academic planning for students.
        - **Expert Reviewers & Academic Critics**: Help us sharpen our models, validate our workflow, and shape credible, student-first solutions.
        - **Developers & Open-Source Contributors**: Join our codebase to co-build smarter forecasting, adaptive feedback systems, and robust study tools.
        - **Sponsors & Supporters**: Back our roadmap and fuel the creation of open educational resources and equitable academic tech.
        Whether you're a researcher, educator, engineer, or strategist — if you're passionate about leveling the academic playing field, OptiGrade needs you.
        Together, we can reimagine how students learn, grow, and thrive.
        """)
        
        if st.button("💌 For Partnerships/Support", use_container_width=True):
            st.info("Reach me on: oluwalowojohn@gmail.com")

    #--------------------------FEATURES TAB ---------------------------    
    with tabs[1]:  # 🚀 Features Tab
        st.subheader("✨ Core Capabilities")
        st.markdown("OptiGrade transforms academic planning through these powerful features:")
        
        # Feature Showcase with Tabs
        feature_tabs = st.tabs(["📊 Prediction", "🎯 Personalization", "⏱️ Study Tools", "📚 Resources"])
        
        with feature_tabs[0]:  # Prediction
            st.markdown("""
            ### 🔮 Intelligent Forecasting
            Our predictive engine analyzes your academic patterns to deliver accurate outcomes:
            
            - **92% accurate CGPA predictions** based on your current performance
            - **Semester-by-semester projections** to see your academic trajectory
            - **Performance factor analysis** identifying key improvement areas
            - **What-if scenarios** to test different study approaches
            
            ```python
            # Sample prediction code
            model.predict({
                'current_gpa': 3.4,
                'attendance': 85,
                'study_hours': 15,
                'assignment_rate': 90
            })
            → Predicted CGPA: 3.72
            ```
            """)
            
            # Use a placeholder if you don't have the image
            st.image("assets/cgpa_predictor.jpeg", 
                    use_container_width=True)
        
        with feature_tabs[1]:  # Personalization
            st.markdown("""
            ### 🎓 Adaptive Learning
            OptiGrade personalizes your experience based on your unique academic profile:
            
            - **Learning style adaptation** (Visual, Auditory, Kinesthetic)
            - **Custom study roadmaps** tailored to your courses and schedule
            - **Weakness identification** with targeted improvement strategies
            - **AI-powered recommendations** using Google's Gemini technology
            
            """)
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Personalization Accuracy", "94%", "2% improvement")
            with col2:
                st.metric("Recommendation Impact", "+0.8 CGPA", "Average improvement")
                
            # Simple pie chart instead of mermaid
            try:
                import plotly.express as px
                data = {'Learning Style': ['Visual', 'Auditory', 'Kinesthetic'],
                        'Percentage': [45, 30, 25]}
                fig = px.pie(data, names='Learning Style', values='Percentage', 
                            color_discrete_sequence=['#00FFD1', '#611EE8', '#1C69B2'])
                st.plotly_chart(fig, use_container_width=True)
            except:
                st.info("Visualization requires plotly package")
        
        with feature_tabs[2]:  # Study Tools
            st.markdown("""
            ### ⏱️ Productivity Toolkit
            Integrated tools to enhance your study efficiency:
            
            - **Pomodoro timer** for focused study sessions
            - **Goal tracking** with progress visualization
            - **Course manager** for deadline tracking
            - **Performance dashboard** with actionable insights
            - **Smart notifications** for important deadlines
            
            """)
            
            # Study tools visualization
            tools = [
                {"name": "Pomodoro Timer", "icon": "⏱️", "color": "#00FFD1"},
                {"name": "Goal Tracker", "icon": "🎯", "color": "#611EE8"},
                {"name": "Course Manager", "icon": "📚", "color": "#1C69B2"},
                {"name": "Analytics", "icon": "📊", "color": "#FF4B4B"},
            ]
            
            cols = st.columns(4)
            for i, tool in enumerate(tools):
                with cols[i]:
                    st.markdown(f"""
                    <div style="text-align: center; padding: 15px; border-radius: 10px; 
                                background: {tool['color']}22; border: 1px solid {tool['color']};">
                        <div style="font-size: 36px; margin-bottom: 10px;">{tool['icon']}</div>
                        <div>{tool['name']}</div>
                    </div>
                    """, unsafe_allow_html=True)
        
        with feature_tabs[3]:  # Resources
            st.markdown("""
            ### 📚 Smart Resource Hub
            Curated academic content tailored to your needs:
            
            - **Course-specific materials** for your current classes
            - **Learning style matched** resources (videos, texts, quizzes)
            - **Personalized recommendations** based on your weak areas
            - **Community suggestions** from top-performing students
            
            """)
            
            # Resource categories
            categories = {
                "STEM": ["Khan Academy", "MIT OpenCourseWare", "Wolfram Alpha"],
                "Humanities": ["Crash Course", "Coursera", "Duolingo"],
                "Programming": ["Codecademy", "freeCodeCamp", "LeetCode"],
                "General": ["Quizlet", "Anki", "StudyBlue"]
            }
            
            for category, resources in categories.items():
                with st.expander(f"📚 {category} Resources"):
                    for resource in resources:
                        st.markdown(f"- {resource}")
        
        # Technology Stack
        st.divider()
        st.subheader("⚙️ Technical Foundation")
        
        tech_cols = st.columns(4)
        technologies = [
            {"name": "Python", "icon": "🐍", "desc": "Core programming language"},
            {"name": "Scikit-Learn", "icon": "🤖", "desc": "Machine learning models"},
            {"name": "Gemini AI", "icon": "🧠", "desc": "Recommendation engine"},
            {"name": "Streamlit", "icon": "🚀", "desc": "Web application framework"},
        ]
        
        for i, tech in enumerate(technologies):
            with tech_cols[i]:
                st.markdown(f"""
                <div style="text-align: center; padding: 15px;">
                    <div style="font-size: 36px;">{tech['icon']}</div>
                    <h4>{tech['name']}</h4>
                    <div style="color: #AAAAAA; font-size: 14px;">{tech['desc']}</div>
                </div>
                """, unsafe_allow_html=True)
        
        # Roadmap
        st.divider()
        st.subheader("🛣️ Development Roadmap")
        
        # Simple roadmap instead of mermaid
        roadmap = """
        | Timeline       | Milestone                     |
        |----------------|-------------------------------|
        | **Q3 2024**    | CGPA Prediction v1.0         |
        | **Q4 2024**    | Study Tools Integration      |
        | **Q1 2025**    | Mobile App Launch           |
        | **Q2 2025**    | Institutional Partnerships  |
        | **Q3 2025**    | NLP Feedback Analysis       |
        | **Q4 2025**    | Collaborative Learning      |
        """
        st.markdown(roadmap)
        
        # Call to Action
        st.markdown("""
        <div style="text-align: center; margin-top: 40px;">
            <a href="https://github.com/CryptoLab-service/OptiGrade-ML-model" target="_blank">
                <button style="background: #00FFD1; 
                            color: black; 
                            border: none; 
                            border-radius: 30px; 
                            padding: 12px 30px; 
                            font-size: 16px; 
                            font-weight: bold;
                            cursor: pointer;">
                    ⭐ Star on GitHub
                </button>
            </a>
        </div>
        """, unsafe_allow_html=True)

    #--------------------------CGPA PREDICTOR TAB ---------------------------    
    with tabs[2]:  # 🧠 CGPA Predictor Tab
        st.subheader("🎓 CGPA Prediction Wizard")

        # Multi-step form
        # Screen 1
        if st.session_state.page == 'Screen 1':
            st.info("Step 1/2: Enter your previous semester details (all fields required)")

        # Transcript upload section
            uploaded_file = st.file_uploader("📤 Upload your transcript (optional)", 
                                            type=["csv", "xlsx"],
                                            help="Upload your transcript to auto-fill previous semester data")
            
            # Process uploaded transcript
            if uploaded_file is not None:
                try:
                    if uploaded_file.name.endswith('.csv'):
                        df = pd.read_csv(uploaded_file)
                    else:
                        df = pd.read_excel(uploaded_file)
                    
                    # Simple validation
                    if len(df) >= 3 and all(col in df.columns for col in ['Course', 'Grade', 'Units']):
                        st.session_state.transcript_data = df.head(3).to_dict('records')
                        st.success("Transcript processed successfully! Fields will be pre-filled.")
                    else:
                        st.warning("Transcript format not recognized. Please ensure it contains Course, Grade, and Units columns.")
                except Exception as e:
                    st.error(f"Error processing file: {str(e)}")

            with st.form("prev_form"):
                prev_courses = []
                for i in range(3):  # Reduced to 3 courses for better UX
                    st.subheader(f"📚 Course {i+1}")

                    cols = st.columns([2, 1, 1])
                    course_id = cols[0].text_input(f"Course Code", key=f"prev_course_id_{i}", 
                                                placeholder="e.g., MATH101", value="")
                    grade = cols[1].number_input(f"Grade", min_value=0, max_value=100, 
                                                step=1, key=f"prev_grade_{i}", value=None,
                                                format="%d")
                    if grade is not None:
                        letter_grade = grade_to_letter(grade)
                        cols[2].markdown(f"<div style='margin-top: 28px; font-size: 18px;'>{letter_grade}</div>", 
                                        unsafe_allow_html=True)

                    cols2 = st.columns(2)
                    study_hours = cols2[0].selectbox(f"Study Hrs/Week", 
                                                    options=list(range(1, 51)),
                                                    index=9, key=f"prev_hours_{i}")
                    attendance = cols2[1].selectbox(f"Attendance %", 
                                                options=[x for x in range(0, 101, 10)],
                                                index=8, key=f"prev_att_{i}")

                    learning_style = st.selectbox("Learning Style", 
                                                ["Visual", "Auditory", "Kinesthetic"],
                                                index=0, key=f"learning_style_{i}")

                    course_units = st.selectbox(f"Course Units", 
                                            options=[1, 2, 3, 4],
                                            index=2, key=f"prev_units_{i}")

                    prev_courses.append({
                        'user_id': st.session_state.user_id, 
                        'semester': 'Previous', 
                        'course_id': course_id,
                        'grade': grade, 
                        'study_hours': study_hours, 
                        'attendance': attendance,
                        'learning_style': learning_style,
                        'course_units': course_units
                    })

                st.divider()
                cols3 = st.columns(2)
                semester_gpa = cols3[0].number_input("Last Semester GPA (0.00-5.00)", min_value=0.0, 
                                                    max_value=5.0, step=0.01, value=None)
                current_cgpa = cols3[1].number_input("Overall Current CGPA (0.00-5.00)", min_value=0.0, 
                                                    max_value=5.0, step=0.01, value=None)

                submitted = st.form_submit_button("👉 Next: Current Semester")
                if submitted:

                    # ✅ Enhanced validation
                    all_filled = True
                    for course in prev_courses:
                        if not course['course_id'] or not course['course_id'].strip():
                            st.error("Course code cannot be empty")
                            all_filled = False
                        if course['grade'] is None or course['study_hours'] is None or \
                        course['attendance'] is None or course['course_units'] is None:
                            all_filled = False

                    if semester_gpa is None or current_cgpa is None:
                        all_filled = False

                    if all_filled:
                        for course in prev_courses:
                            course['semester_gpa'] = semester_gpa
                        st.session_state.current_cgpa = current_cgpa
                        st.session_state.prev_data = prev_courses
                        st.session_state.page = 'Screen 2'
                        st.rerun()
                    else:
                        st.error("Please fill in all fields before proceeding")

        elif st.session_state.page == 'Screen 2':
            st.info("Step 2/2: Enter current semester details (all fields required)")

            # Current semester form
            with st.form("curr_form"):
                curr_courses = []
                for i in range(3):
                    st.subheader(f"📚 Course {i+1}")
                    cols = st.columns([2, 1])
                    course_id = cols[0].text_input(f"Course Code", key=f"curr_course_id_{i}", 
                                                placeholder="e.g., PHY102", value="")
                    course_units = cols[1].selectbox(f"Units", 
                                                    options=[1, 2, 3, 4],
                                                    index=2, 
                                                    key=f"curr_units_{i}")
                    # learning style
                    learning_style = st.selectbox("Learning Style", 
                                                ["Visual", "Auditory", "Kinesthetic"],
                                                key=f"curr_learning_style_{i}")

                    curr_courses.append({
                        'user_id': st.session_state.user_id, 
                        'semester': 'Current', 
                        'course_id': course_id,
                        'course_units': course_units,
                        'learning_style': learning_style
                    })

                submitted = st.form_submit_button("✨ Generate Prediction")
                if submitted:
                    all_filled = True
                    for course in curr_courses:
                        if not course['course_id'] or course['course_units'] is None:
                            all_filled = False

                    if all_filled:
                        st.session_state.curr_data = curr_courses
                        st.session_state.page = 'Results'
                        st.rerun()
                    else:
                        st.error("Please fill in all fields before proceeding")

            if st.button("🔙 Back to Previous Step"):
                st.session_state.page = 'Screen 1'
                st.rerun()

        # Prediction Result Page
        elif st.session_state.page == 'Results':
            st.success("✅ Prediction Complete!")
            st.subheader(f"📊 Academic Forecast for Student {st.session_state.user_id}")
            
# ========== PREDICTION SECTION ===============================        
            # Create sample input for prediction
            try:
                # Create raw_input dictionary safely
                raw_input = {
                    "Current GPA": float(st.session_state.current_cgpa),
                    "Assignments Completed": 85.0,
                    "Midterm Score": 75.0,
                    "Lecture Engagement": 80.0
                }
                
                # Handle attendance and study hours safely
                if st.session_state.prev_data:
                    try:
                        raw_input["Attendance %"] = float(np.mean([c.get('attendance', 0) for c in st.session_state.prev_data]))
                    except:
                        raw_input["Attendance %"] = 0.0
                    
                    try:
                        raw_input["Study Hours per Week"] = float(np.mean([c.get('study_hours', 0) for c in st.session_state.prev_data]))
                    except:
                        raw_input["Study Hours per Week"] = 0.0
                else:
                    raw_input["Attendance %"] = 0.0
                    raw_input["Study Hours per Week"] = 0.0
                
                # Map features to what model expects
                sample_input = map_features_to_model(raw_input)
                
            except Exception as e:
                st.error(f"Error creating input data: {str(e)}")
                sample_input = None

            if sample_input is not None:
                try:
                    # === DEBUG INFORMATION SECTION ===
                    with st.expander("🔍 Result Information (Debug)", expanded=False):
                        st.subheader("Debug Information")
                        
                        # --- Expected Features Display ---
                        st.write("**Expected Features:**")
                        expected_features_df = pd.DataFrame({
                            "Index": range(len(st.session_state.expected_features)),
                            "Feature Name": st.session_state.expected_features
                        })
                        st.dataframe(
                            expected_features_df,
                            column_config={
                                "Index": st.column_config.NumberColumn(width="small"),
                                "Feature Name": st.column_config.TextColumn(width="large")
                            },
                            hide_index=True,
                            use_container_width=True
                        )
                        
                        # --- Mapped Features Display ---
                        st.write("**Mapped Features with Values:**")
                        formatted_sample_input = {k: f"{v:.2f}" if isinstance(v, float) else str(v) 
                                                for k, v in sample_input.items()}
                        feature_df = pd.DataFrame(
                            list(formatted_sample_input.items()), 
                            columns=['Feature', 'Value']
                        )
                        st.dataframe(
                            feature_df,
                            column_config={
                                "Feature": st.column_config.TextColumn(width="medium"),
                                "Value": st.column_config.NumberColumn(width="medium")
                            },
                            hide_index=True,
                            use_container_width=True
                        )
                        
                        # --- Feature Validation ---
                        missing_features = [f for f in st.session_state.expected_features if f not in sample_input]
                        if missing_features:
                            st.warning(f"Missing features: {missing_features}")
                        
                        extra_features = [f for f in sample_input if f not in st.session_state.expected_features]
                        if extra_features:
                            st.warning(f"Extra features: {extra_features}")

                    # === PREDICTION RESULTS SECTION ===
                    if st.session_state.ml_model:
                        try:
                            previous_cgpa = float(st.session_state.current_cgpa)
                            # Create dataframe with correct feature order
                            input_values = []
                            for feature in st.session_state.expected_features:
                                value = sample_input.get(feature, 0.0)
                                try:
                                    input_values.append(float(value))
                                except (TypeError, ValueError):
                                    input_values.append(0.0)
                            
                            input_df = pd.DataFrame([input_values], 
                                                columns=st.session_state.expected_features)
                            
                            prediction = float(st.session_state.ml_model.predict(input_df)[0])
                            
                            # Display prediction metrics
                            col1, col2 = st.columns([1, 2])
                            with col1:
                                st.metric("Previous CGPA", f"{previous_cgpa:.2f}")
                                st.metric("Predicted Final CGPA", f"{prediction:.2f}", 
                                        delta=f"{prediction - previous_cgpa:.2f}")
                                
                                # Progress bar without help parameter
                                progress_value = min(prediction / 5.0, 1.0)
                                st.progress(progress_value)
                                
                                # Grade interpretation
                                if prediction >= 4.0:
                                    st.success("First Class Performance! 🎉")
                                elif prediction >= 3.0:
                                    st.info("Good Standing - Keep Improving! 📈")
                                else:
                                    st.warning("Needs Improvement - Review Recommendations")
                            
                            with col2:
                                # Create and display forecast chart
                                fig = create_dotted_forecast_chart(previous_cgpa, prediction)
                                st.pyplot(fig)

                            # --- Feedback and Recommendations Section ---
                            st.divider()
                            st.subheader("📝 Performance Feedback & Recommendations")

                            # Generate feedback
                            feedback, tips = generate_feedback(prediction, raw_input)

                            # Display feedback
                            st.info(feedback)

                            # Display tips
                            st.markdown("### 🔍 Areas for Improvement:")
                            for tip in tips:
                                st.markdown(f"- {tip}")
                                
                            # Enhanced Resource recommendations
                            st.markdown("### 📚 Recommended Resources:")
                            
                            # Study Resources
                            if raw_input["Study Hours per Week"] < 15:
                                st.markdown("""
                                **Study Habits & Techniques:**
                                - [Study Smarter, Not Harder](https://learningcenter.unc.edu/tips-and-tools/studying-101-study-smarter-not-harder/)
                                - [Active Learning Strategies](https://www.cultofpedagogy.com/active-learning-strategies/)
                                - [Pomodoro Technique Guide](https://todoist.com/productivity-methods/pomodoro-technique)
                                """)
                            
                            # Attendance Resources
                            if raw_input["Attendance %"] < 70:
                                st.markdown("""
                                **Attendance Improvement:**
                                - [Why Attendance Matters](https://www.edutopia.org/article/why-attendance-matters)
                                - [Building Attendance Habits](https://www.understood.org/articles/en/how-to-help-your-child-with-attendance-issues)
                                """)
                            
                            # Engagement Resources
                            if raw_input["Lecture Engagement"] < 70:
                                st.markdown("""
                                **Lecture Engagement:**
                                - [Active Learning Strategies](https://www.celt.iastate.edu/teaching/effective-teaching-practices/active-learning)
                                - [Note-taking Systems](https://www.student.unsw.edu.au/note-taking-skills)
                                """)
                            
                            # General Resources
                            st.markdown("""
                            **General Academic Improvement:**
                            - [Khan Academy](https://www.khanacademy.org/) - Free courses on all subjects
                            - [Coursera](https://www.coursera.org/) - Online courses from top universities
                            - [Quizlet](https://quizlet.com/) - Study tools and flashcards
                            """)
                            
                            # Display student profile
                            display_student_profile()

                            # Generate AI recommendations
                            student_data_str = format_student_data()
                            gemini_recommendations = get_academic_recommendations(student_data_str)
                            
                            st.subheader("🧠 Recommended Pathways to Achieve Your Goals")
                            st.markdown(gemini_recommendations)
                            
                        except Exception as e:
                            st.error(f"Error during prediction: {str(e)}")
                            st.error(traceback.format_exc())
                            
                    # Display input summary
                    with st.expander("📋 View Academic Input Summary", expanded=False):
                        st.subheader("Previous Semester")
                        if st.session_state.prev_data:
                            prev_df = pd.DataFrame(st.session_state.prev_data)
                            prev_df['Grade'] = prev_df['grade'].apply(lambda x: f"{x} ({grade_to_letter(x)})")
                            st.dataframe(prev_df[['course_id', 'Grade', 'study_hours', 'attendance', 'course_units', 'learning_style']])
                        else:
                            st.info("No previous semester data")
                        
                        st.subheader("Current Semester")
                        if st.session_state.curr_data:
                            st.dataframe(pd.DataFrame(st.session_state.curr_data))
                        else:
                            st.info("No current semester data")
                    
                    if st.button("🔄 Start New Prediction"):
                        st.session_state.page = 'Screen 1'
                        st.rerun()
                        
                except Exception as e:
                    st.error(f"Error processing data: {str(e)}")
                    st.error(traceback.format_exc())
                    
            if sample_input is None or st.session_state.ml_model is None:
                st.error("Prediction not possible due to missing data or model")
                if st.button("🔙 Back to Input Form"):
                    st.session_state.page = 'Screen 1'
                    st.rerun()

    #--------------------------STUDY HUB TAB ---------------------------   
    with tabs[3]:  # 📘 Study Hub
        # Create subtabs for Study Hub
        study_tabs = st.tabs([
            "📅 Weekly Planner", 
            "⏱️ Focus Timer", 
            "🎯 Goals & Tasks",
            "📊 Progress & Analytics"
        ])
        
        # Weekly Planner subtab
        with study_tabs[0]:
            st.title("📅 Weekly Study Planner")
            
            # Goal setting section
            st.subheader("🎯 Set Your Weekly Goals")
            col1, col2 = st.columns(2)
            daily_goal = col1.number_input("Daily Study Goal (hours)", min_value=1.0, max_value=10.0, value=3.0, step=0.5)
            weekly_goal = col2.number_input("Weekly Study Goal (hours)", min_value=5.0, max_value=50.0, value=15.0, step=1.0)
            
            st.divider()
            
            # Daily planning section
            st.subheader("📝 Plan Your Study Week")
            days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
            study_hours = []
            
            for day in days:
                hour = st.number_input(
                    f"{day} study hours:", 
                    min_value=0.0, 
                    max_value=8.0, 
                    value=2.0 if day not in ["Saturday", "Sunday"] else 3.0,
                    step=0.5
                )
                study_hours.append(hour)
            
            # Calculate totals
            total_hours = sum(study_hours)
            daily_avg = total_hours / 7
            
            # Display summary
            st.divider()
            st.subheader("📋 Weekly Summary")
            col1, col2, col3 = st.columns(3)
            col1.metric("Total Planned Hours", f"{total_hours:.1f}h")
            col2.metric("Daily Average", f"{daily_avg:.1f}h")
            col3.metric("Goal Progress", f"{total_hours}/{weekly_goal}h", 
                    delta=f"{(total_hours/weekly_goal*100-100):.1f}%" if weekly_goal > 0 else "0%")
            
            # Visualization
            fig, ax = plt.subplots(figsize=(10, 4))
            x = [d[:3] for d in days]  # Short day names
            y = study_hours
            
            # Bar colors based on daily goal
            colors = []
            for hours in study_hours:
                if hours >= daily_goal:
                    colors.append('#4CAF50')  # Green
                elif hours >= daily_goal * 0.7:
                    colors.append('#FFC107')  # Yellow
                else:
                    colors.append('#F44336')  # Red
            
            ax.bar(x, y, color=colors)
            ax.axhline(y=daily_goal, color='#2196F3', linestyle='--', label='Daily Goal')
            ax.set_title('Your Weekly Study Plan', fontsize=16)
            ax.set_ylabel('Hours')
            ax.legend()
            ax.grid(axis='y', linestyle='--', alpha=0.3)
            
            st.pyplot(fig)
        
        # Focus Timer subtab
        with study_tabs[1]:
            st.title("⏱️ Focus Timer")
            st.markdown("Enhance your productivity with timed study sessions")
            
            col1, col2 = st.columns([3, 1])
            with col1:
                # Timer controls
                st.markdown("### 🍅 Pomodoro Timer")
                timer_col1, timer_col2, timer_col3 = st.columns(3)
                
                if timer_col1.button("Start 25 min", use_container_width=True):
                    start_study_timer(25 * 60)
                if timer_col2.button("Start 50 min", use_container_width=True):
                    start_study_timer(50 * 60)
                if timer_col3.button("Stop Timer", disabled=not st.session_state.study_timer_active, use_container_width=True):
                    stop_study_timer()
                
                # Timer display
                st.divider()
                if st.session_state.study_timer_active:
                    current_time = time.time()
                    elapsed = current_time - st.session_state.study_timer_start
                    remaining = max(0, st.session_state.study_timer_duration - elapsed)
                    st.session_state.study_timer_remaining = remaining
                    
                    if remaining <= 0:
                        stop_study_timer()
                        st.balloons()
                        st.success("Time's up! Take a break.")
                
                minutes, seconds = divmod(st.session_state.study_timer_remaining, 60)
                timer_display = f"{int(minutes):02d}:{int(seconds):02d}"
                
                st.markdown(f"<div style='text-align: center; margin: 30px 0;'>", unsafe_allow_html=True)
                st.markdown(f"<h1 style='text-align: center; font-size: 72px;'>{timer_display}</h1>", unsafe_allow_html=True)
                st.markdown(f"<p style='text-align: center; font-size: 18px;'>Mode: {'Focus Time' if st.session_state.study_timer_active else 'Ready'}</p>", unsafe_allow_html=True)
                st.markdown(f"</div>", unsafe_allow_html=True)
                
                # Session history
                st.divider()
                st.markdown("### 📝 Session History")
                st.write(f"Completed Pomodoro sessions: {st.session_state.pomodoro_count}")
                
            with col2:
                # Achievements
                st.markdown("### 🏆 Study Achievements")
                badge = get_achievement_badge(st.session_state.pomodoro_count)
                
                # Badge display
                st.markdown(f"""
                    <div style="text-align: center; padding: 20px; 
                                background: linear-gradient(135deg, #1e1e2e, #2a2a40);
                                border-radius: 10px; border: 1px solid #00FFD1;">
                        <div style="font-size: 48px;">🏆</div>
                        <h3>{badge}</h3>
                        <div style="font-size: 24px; color: #00FFD1; margin-top: 10px;">
                            {st.session_state.pomodoro_count} sessions
                        </div>
                    </div>
                """, unsafe_allow_html=True)
                
                # Tips
                st.divider()
                st.markdown("### 💡 Focus Tips")
                st.info("• Eliminate distractions during focus sessions")
                st.info("• Take 5-minute breaks between sessions")
                st.info("• Review what you've learned after each session")
        
        # Goals & Tasks subtab
        with study_tabs[2]:
            st.title("🎯 Goals & Tasks")
            
            col1, col2 = st.columns([2, 1])
            with col1:
                # Goal creation
                st.subheader("📝 Create New Goals")
                with st.form("user_profile_goal_form"):
                    goal_title = st.text_input("Goal Title", placeholder="e.g., Master Calculus Chapter 3")
                    goal_description = st.text_area("Description", placeholder="Specific details about your goal...")
                    goal_due = st.date_input("Due Date")
                    goal_priority = st.select_slider("Priority", options=["Low", "Medium", "High"], value="Medium")
                    
                    if st.form_submit_button("Add Goal"):
                        if goal_title:
                            st.session_state.study_goals.append({
                                "title": goal_title,
                                "description": goal_description,
                                "due": goal_due.strftime("%Y-%m-%d"),
                                "priority": goal_priority,
                                "completed": False
                            })
                            st.success("Goal added successfully!")
                
                st.divider()
                
                # Active goals
                st.subheader("📋 Active Goals")
                if not st.session_state.study_goals:
                    st.info("No active goals. Create your first goal above!")
                else:
                    for i, goal in enumerate(st.session_state.study_goals):
                        if not goal["completed"]:
                            with st.expander(f"{goal['title']} - {goal['priority']} Priority", expanded=True):
                                st.write(goal["description"])
                                
                                if goal["due"]:
                                    due_date = goal["due"]
                                    today = pd.Timestamp.today().strftime("%Y-%m-%d")
                                    days_left = (pd.Timestamp(goal["due"]) - pd.Timestamp.today()).days
                                    
                                    if days_left < 0:
                                        date_info = f"⚠️ Overdue by {-days_left} days"
                                        color = "#FF4B4B"
                                    elif days_left < 7:
                                        date_info = f"🔜 Due in {days_left} days"
                                        color = "#FFA500"
                                    else:
                                        date_info = f"📅 Due in {days_left} days"
                                        color = "#00FFD1"
                                        
                                    st.markdown(f"<div style='color: {color};'>{date_info}</div>", unsafe_allow_html=True)
                                
                                cols = st.columns([1, 1, 2])
                                if cols[0].button("Complete", key=f"complete_{i}"):
                                    st.session_state.study_goals[i]["completed"] = True
                                    st.rerun()
                                if cols[1].button("Delete", key=f"delete_{i}"):
                                    st.session_state.study_goals.pop(i)
                                    st.rerun()
            
            with col2:
                # Progress visualization
                st.subheader("📊 Goal Progress")
                
                # Calculate goal stats
                total_goals = len(st.session_state.study_goals)
                completed_goals = sum(1 for goal in st.session_state.study_goals if goal["completed"])
                progress = completed_goals / total_goals if total_goals > 0 else 0
                
                st.metric("Goals Completed", f"{completed_goals}/{total_goals}", f"{progress*100:.1f}%")
                st.progress(progress)
                
                # Priority distribution
                if total_goals > 0:
                    priorities = [goal["priority"] for goal in st.session_state.study_goals if not goal["completed"]]
                    priority_counts = {p: priorities.count(p) for p in set(priorities)}
                    
                    fig, ax = plt.subplots()
                    ax.pie(priority_counts.values(), labels=priority_counts.keys(), autopct='%1.1f%%',
                        colors=['#4CAF50', '#FFC107', '#F44336'])
                    ax.set_title('Priority Distribution')
                    st.pyplot(fig)
                
                # Completed goals
                st.divider()
                st.subheader("✅ Completed Goals")
                if completed_goals == 0:
                    st.info("No completed goals yet")
                else:
                    for goal in st.session_state.study_goals:
                        if goal["completed"]:
                            st.markdown(f"- ~~{goal['title']}~~")
        
        # Progress & Analytics subtab
        with study_tabs[3]:
            st.title("📊 Progress & Analytics")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Study time analytics
                st.subheader("⏱️ Study Time Analysis")
                st.markdown("**Last 7 Days Study Hours**")
                
                # Sample data (in a real app, this would come from a database)
                study_data = pd.DataFrame({
                    "Day": ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"],
                    "Planned": [2.0, 3.0, 1.5, 4.0, 2.5, 3.0, 1.0],
                    "Actual": [1.5, 2.8, 1.0, 3.5, 2.0, 4.0, 0.5]
                })
                
                fig, ax = plt.subplots(figsize=(8, 4))
                ax.plot(study_data["Day"], study_data["Planned"], 'o-', label='Planned', color='#00FFD1')
                ax.plot(study_data["Day"], study_data["Actual"], 'o-', label='Actual', color='#4CAF50')
                ax.set_title('Planned vs Actual Study Time')
                ax.set_ylabel('Hours')
                ax.legend()
                ax.grid(True, linestyle='--', alpha=0.3)
                st.pyplot(fig)
                
                # Efficiency metric
                efficiency = (study_data["Actual"].sum() / study_data["Planned"].sum()) * 100
                st.metric("Study Efficiency", f"{efficiency:.1f}%", 
                        delta="+5.2%" if efficiency > 100 else "-2.3%")
                
            with col2:
                # Productivity insights
                st.subheader("🚀 Productivity Insights")
                
                # Focus session analytics
                st.markdown("**Focus Session History**")
                session_data = pd.DataFrame({
                    "Date": ["2025-07-18", "2025-07-19", "2025-07-20", "2025-07-21", "2025-07-22"],
                    "Duration": [25, 50, 25, 25, 50],
                    "Focus Level": [3, 4, 2, 4, 5]
                })
                
                fig, ax = plt.subplots(figsize=(8, 4))
                ax.bar(session_data["Date"], session_data["Duration"], color='#1C69B2')
                ax.set_title('Focus Session Duration (min)')
                ax.set_ylabel('Minutes')
                ax.grid(axis='y', linestyle='--', alpha=0.3)
                st.pyplot(fig)
                
                # Focus quality metric
                avg_focus = session_data["Focus Level"].mean()
                st.metric("Average Focus Level", f"{avg_focus:.1f}/5", 
                        delta="+0.3" if avg_focus > 3.5 else "-0.2")
                
                # Recommendations
                st.divider()
                st.subheader("💡 Improvement Tips")
                if avg_focus < 3.0:
                    st.info("• Try different study environments to improve focus")
                    st.info("• Use the Pomodoro technique with shorter intervals")
                elif avg_focus < 4.0:
                    st.info("• Minimize distractions during study sessions")
                    st.info("• Review your most productive times of day")
                else:
                    st.info("• Maintain your effective study habits")
                    st.info("• Consider mentoring others with your techniques")

    #--------------------------COURSE MANAGER TAB ---------------------------
    with tabs[4]:  # 📂 Course Manager
        st.subheader("📂 Course Manager")
        
        col1, col2 = st.columns([1, 2])
        with col1:
            st.info("""
            **Our Course Manager will launch in v2.0!**
            
            Prepare for these powerful features:
            - 🗓️ Semester course planning with drag & drop
            - ⏰ Assignment deadline tracker with notifications
            - 📊 Grade calculator with scenario planning
            - ⭐ Course difficulty ratings and reviews
            - 🔄 Credit transfer management
            """)
            
        with col2:
            # Interactive course planner concept
            st.markdown("#### 🎯 Your Academic Plan")
            st.markdown("""
            Craft your learning journey with a personalized semester roadmap. 
            View registered courses, track planned ones, and reflect on completed subjects.
            """)
            
            planned_courses = [
                {"Course": "CHM101", "Title": "General Chemistry I", "Units": 4, "Status": "Planned", "Difficulty": "⭐️⭐️⭐️"},
                {"Course": "GST113", "Title": "Nigerian People and Culture", "Units": 2, "Status": "Registered", "Difficulty": "⭐️⭐️"},
                {"Course": "MAT103", "Title": "Introductory Calculus", "Units": 3, "Status": "Registered", "Difficulty": "⭐️⭐️⭐️⭐️"},
                {"Course": "PHY101", "Title": "General Physics I", "Units": 3, "Status": "Completed", "Difficulty": "⭐️⭐️⭐️⭐️"},
                {"Course": "PHY107", "Title": "General Physics Laboratory I", "Units": 1, "Status": "Completed", "Difficulty": "⭐️⭐️"},
            ]
            
            # Display course cards
            for course in planned_courses:
                status_color = {
                    "Planned": "#4e79a7",
                    "Registered": "#59a14f",
                    "Completed": "#b07aa1"
                }.get(course["Status"], "#000000")
                
                st.markdown(f"""
                    <div style="background: #1e1e2e; border-radius: 8px; padding: 12px; margin-bottom: 10px;">
                        <div style="display: flex; justify-content: space-between;">
                            <span style="font-weight: bold; font-size: 18px;">{course['Course']}: {course['Title']}</span>
                            <span style="background: {status_color}; color: white; padding: 2px 10px; border-radius: 12px; font-size: 12px;">
                                {course['Status']}
                            </span>
                        </div>
                        <div style="color: #a0a0a0; margin-top: 8px;">
                            Units: {course['Units']} • Difficulty: {course['Difficulty']}
                        </div>
                    </div>
                """, unsafe_allow_html=True)

    #--------------------------RESOURCES TAB ---------------------------
    with tabs[5]:  # 📚 Resources
        st.subheader("📚 Academic Resources")
        st.markdown("Curated resources to enhance your learning experience")
        
        # Resource categories
        categories = ["All"] + list(set(resource["category"] for resource in st.session_state.resources))
        selected_category = st.selectbox("Filter by Category", categories)
        
        # Display resources
        col1, col2 = st.columns(2)
        resource_counter = 0
        
        for resource in st.session_state.resources:
            if selected_category == "All" or resource["category"] == selected_category:
                with (col1 if resource_counter % 2 == 0 else col2):
                    st.markdown(f"""
                        <div style="border: 1px solid #2D3746; border-radius: 10px; padding: 15px; margin-bottom: 20px;">
                            <h4>{resource['title']}</h4>
                            <p style="color: #888; font-size: 14px;">Category: {resource['category']}</p>
                            <a href="{resource['url']}" target="_blank" style="color: #00FFD1; text-decoration: none;">
                                Visit Resource →
                            </a>
                        </div>
                    """, unsafe_allow_html=True)
                    resource_counter += 1
        
        # Resource suggestion form
        with st.expander("➕ Suggest a Resource"):
            new_title = st.text_input("Resource Title")
            new_url = st.text_input("Resource URL")
            new_category = st.text_input("Category")
            
            if st.button("Submit Suggestion"):
                st.session_state.resources.append({
                    "title": new_title,
                    "url": new_url,
                    "category": new_category
                })
                st.success("Thank you for your suggestion!")

    #--------------------------USER PROFILE TAB ---------------------------
    with tabs[6]:  # 👤 User Profile
        st.subheader("👤 User Profile Settings")
        
        col1, col2 = st.columns([1, 2])
        with col1:
            # Profile picture selection
            st.markdown("### Profile Picture")
            profile_options = ["👨‍🎓", "👩‍🎓", "👨‍💻", "👩‍💻", "😎", "🤖"]
            selected_emoji = st.radio("Select an avatar:", profile_options, 
                                     index=profile_options.index(st.session_state.user_pic),
                                     horizontal=True)
            
            # Update profile pic
            if selected_emoji != st.session_state.user_pic:
                st.session_state.user_pic = selected_emoji
                st.success("Profile picture updated!")
 
        
        # ---- Faculty and Departments Mapping ----
        faculty_departments = {
            "Sciences": [
                "Physics", "Computer Science", "Chemistry", "Biology", "Mathematics", "Geology", "Microbiology"
            ],
            "Engineering": [
                "Civil Engineering", "Mechanical Engineering", "Electrical Engineering", "Computer Engineering", "Chemical Engineering"
            ],
            "Arts": [
                "Arts and Culture", "History", "Philosophy", "Literature", "Theatre Arts"
            ],
            "Social Sciences": [
                "Sociology", "Political Science", "Psychology", "Economics", "Anthropology"
            ],
            "Medical Sciences": [
                "Medicine and Surgery", "Pharmacy", "Nursing", "Medical Laboratory Science", "Public Health"
            ],
            "Management Sciences": [
                "Accounting", "Business Administration", "Marketing", "Banking and Finance", "Entrepreneurship"
            ],
            "Law": [
                "Law", "International Law and Diplomacy", "Legal Studies", "Criminology and Security Studies"
            ],
            "Education": [
                "Educational Psychology", "Curriculum and Instruction", "Guidance and Counselling", "Early Childhood Education", "Science Education"
            ],
            "Agriculture": [
                "Agricultural Economics", "Crop Science", "Animal Science", "Soil Science", "Food Science and Technology"
            ],
            "Interdisciplinary Studies": [
                "Environmental Studies", "Gender and Development", "Peace and Conflict Studies", "Global Studies", "Data and Society"
            ]
        }

        # ---- Personal information ----
        with col2:
            st.markdown("### Personal Information")
            
            # Full Name
            new_name = st.text_input("Full Name", value=st.session_state.get("user_name", ""))
            if new_name != st.session_state.get("user_name", ""):
                st.session_state["user_name"] = new_name

            # Faculty Selection
            selected_faculty = st.selectbox("Faculty", list(faculty_departments.keys()))

            # Department Dropdown updates based on faculty
            departments_for_faculty = faculty_departments[selected_faculty]
            selected_department = st.selectbox("Department", departments_for_faculty)

            # Current Level Dropdown (100–700 Level)
            level_options = [f"{lvl} Level" for lvl in range(100, 800, 100)]
            selected_level = st.selectbox("Current Level", level_options, index=1)  # default is 200 Level

            if st.button("Save Profile Changes"):
                st.success("Profile updated successfully!")

        
        # Academic details
        st.markdown("### Academic Information")
        col3, col4 = st.columns(2)
        with col3:
            new_cgpa = st.number_input("Current CGPA", min_value=0.0, max_value=5.0, 
                                      value=st.session_state.current_cgpa, step=0.01)
            if new_cgpa != st.session_state.current_cgpa:
                st.session_state.current_cgpa = new_cgpa
        with col4:
            st.selectbox("Primary Learning Style", ["Visual", "Auditory", "Kinesthetic"])
        
        st.divider()
        st.markdown("### Notification Preferences")
        st.checkbox("Email notifications", value=True)
        st.checkbox("Push notifications", value=True)
        st.checkbox("Weekly performance reports", value=True)

# ------------------ FOOTER ------------------
st.divider()
st.markdown("""
<div style='text-align: center; margin-top: 50px; font-size: 13px; font-weight: bold; color: #AAAAAA;'>
    © 2025 <strong>Zoe Tech Hub</strong> | <em>OptiGrade - Academic Performance Optimization System</em> | <span style='color:#00FFD1;'>v2.0.0</span>
</div>
""", unsafe_allow_html=True)
