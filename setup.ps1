# PowerShell Setup Script for OptiGrade

Write-Host "`n🚀 Setting up OptiGrade environment..." -ForegroundColor Cyan

# Step 1: Create virtual environment
python -m venv optigrade_env

# Step 2: Activate environment
$envPath = ".\optigrade_env\Scripts\Activate.ps1"
if (Test-Path $envPath) {
    & $envPath
    Write-Host "✅ Virtual environment activated."
} else {
    Write-Host "❌ Could not find activation script." -ForegroundColor Red
    exit 1
}

# Step 3: Upgrade pip and setuptools
pip install --upgrade pip setuptools

# Step 4: Install dependencies
pip install -r requirements.txt

# Step 5: Create .env file if missing
if (!(Test-Path ".env")) {
    Copy-Item ".env.example" ".env"
    Write-Host "✅ .env file created from example."
}

# Step 6: Train model if training script exists
$trainScript = ".\models\train_model.py"
if (Test-Path $trainScript) {
    python $trainScript
    Write-Host "🧠 Model trained and saved to models/model.pkl"
}

Write-Host "`n🎉 Setup complete! You can now run:"
Write-Host "streamlit run optigrade_app.py" -ForegroundColor Green
