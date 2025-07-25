import pytest
import streamlit as st
from optigrade_helpers import format_student_data

def test_format_student_data_basic():
    mock_data = {
        'user_name': 'Jane Doe',
        'user_id': '2023-XYZ',
        'current_cgpa': 3.75,
        'curr_data': [
            {'name': 'Mathematics', 'grade': 'A'},
            {'name': 'Biology', 'grade': 'B+'}
        ]
    }

    result = format_student_data(mock_data)

    assert "Jane Doe" in result
    assert "2023-XYZ" in result
    assert "Mathematics - Grade: A" in result
    assert "Biology - Grade: B+" in result

# 🧪 Fixture to mock session state
@pytest.fixture
def mock_session_state(monkeypatch):
    mock_data = {
        'user_name': 'Jane Doe',
        'user_id': '2023-XYZ',
        'current_cgpa': 3.75,
        'curr_data': [
            {'name': 'Mathematics', 'grade': 'A'},
            {'name': 'Biology', 'grade': 'B+'}
        ]
    }

    for key, value in mock_data.items():
        monkeypatch.setitem(st.session_state, key, value)

    return mock_data

# 🚨 Actual test using fixture data
def test_format_student_data_from_session(mock_session_state):
    result = format_student_data(mock_session_state)

    assert "Jane Doe" in result
    assert "2023-XYZ" in result
    assert "3.75" in result
    assert "Mathematics - Grade: A" in result
    assert "Biology - Grade: B+" in result
    assert len(result) > 50
