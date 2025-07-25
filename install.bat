@echo off
echo 🚀 Setting up OptiGrade environment...

REM Step 1: Create virtual environment
python -m venv optigrade_env
call optigrade_env\Scripts\activate.bat

REM Step 2: Upgrade pip & setuptools
python -m pip install --upgrade pip setuptools

REM Step 3: Install dependencies
pip install -r requirements.txt

REM Step 4: Create .env from example
IF NOT EXIST .env (
    copy .env.example .env
    echo ✅ Created .env file from .env.example
)

REM Step 5: Optional model retrain step
IF EXIST models\train_model.py (
    python models\train_model.py
)

echo 🎉 Setup complete! Run the app using:
echo streamlit run optigrade_app.py
pause
