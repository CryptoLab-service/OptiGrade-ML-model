def format_student_data(session_state):
    name = session_state["user_name"]
    user_id = session_state["user_id"]
    current_cgpa = session_state["current_cgpa"]
    courses = session_state["curr_data"]

    course_list = "\n".join([f"{c['name']} - Grade: {c['grade']}" for c in courses]) if courses else "No current courses."

    return f"""Student Name: {name}
Student ID: {user_id}
Current CGPA: {current_cgpa}
Current Courses:
{course_list}
"""
