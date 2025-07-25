# 🎓 OptiGrade – Intelligent Academic Companion

[![Streamlit](https://img.shields.io/badge/Framework-Streamlit-red)](https://streamlit.io/)
![AI Powered](https://img.shields.io/badge/Powered%20by-AI-brightgreen?logo=OpenAI&style=for-the-badge&labelColor=black)
![Accuracy](https://img.shields.io/badge/Accuracy-Up%20to%2092%25-blueviolet?style=for-the-badge&logo=Speedtest)
[![Streamlit App](https://img.shields.io/badge/Open-App-green?logo=streamlit)](https://optigrade.streamlit.app/)

[![License](https://img.shields.io/github/license/CryptoLab-service/OptiGrade-ML-model)](LICENSE)
[![GitHub Stars](https://img.shields.io/github/stars/CryptoLab-service/OptiGrade-ML-model?style=social)](https://github.com/CryptoLab-service/OptiGrade-ML-model/stargazers)
[![GitHub Forks](https://img.shields.io/github/forks/CryptoLab-service/OptiGrade-ML-model?style=social)](https://github.com/CryptoLab-service/OptiGrade-ML-model/forks)
[![Demo](https://img.shields.io/badge/Demo-Streamlit-blue)](https://optigrade.streamlit.app/)

> ✨ Empowering students through smart academic forecasting, adaptive feedback, and personalized learning journeys

---

# 🔍 About OptiGrade

**OptiGrade** is an AI-powered academic assistant that analyzes student behavior and academic data to generate personalized study plans, predictive insights, and smart progress tracking. Built for higher institutions, OptiGrade empowers students with clarity, confidence, and tailored guidance through each semester.

- 🚀 Powered by machine learning
- 📊 Deployed via [Streamlit Model Engine](https://optigrade.streamlit.app/)
- 🎨 Designed as a responsive mobile prototype for intuitive student interaction
- 💻 Solo-developed, open-source, and ready to evolve through collaboration

---

# 🧩 Tags

`ai` `edtech` `figma` `mobile-prototype` `uiux` `streamlit` `student-dashboard`  
`machine-learning` `academic-planner` `higher-institution` `nigeria`

---

# 📱 Mobile Prototype Preview

Explore the mobile concept design on Figma below:

> ✨ Live Preview:  
`"https://www.figma.com/embed?embed_host=streamlit&url=https://www.figma.com/proto/B2L8DOx0u3xuSWPhKpJpO5/OptiGrade-Mobile-App---EduTech?node-id=802-966&starting-point-node-id=802%3A966&scaling=scale-down"`

---

# 🚀 Features

- **📊 CGPA Prediction** — Forecast with up to 92% accuracy  
- **🧠 Personalized Planning** — AI study recommendations based on habits and performance  
- **📂 Course Manager** — Organize courses and progress  
- **⏱️ Study Hub** — Pomodoro timer, goal tracking, and achievement system  
- **📚 Resource Library** — Curated content per department  
- **📈 Analytics Dashboard** — Visual insights into patterns and planning

---

# 📍 Project Roadmap

> 🧭 Track progress in [GitHub Projects](https://github.com/CryptoLab-service/OptiGrade-ML-model/projects) and [Issues](https://github.com/CryptoLab-service/OptiGrade-ML-model/issues)

| Milestone                      | Status       | Badge                                                                 |
|-------------------------------|--------------|------------------------------------------------------------------------|
| ✅ MVP Development            | Completed    | ![MVP](https://img.shields.io/badge/MVP-Built-green?style=flat-square) |
| 🚧 Feedback Integration       | In Progress  | ![Feedback](https://img.shields.io/badge/Feedback-In--Progress-orange?style=flat-square) |
| ⏳ LMS Integration            | Upcoming     | ![LMS](https://img.shields.io/badge/LMS%20Integration-Pending-blue?style=flat-square) |
| 🔍 Model Optimization         | Upcoming     | ![Optimize](https://img.shields.io/badge/Model%20Tuning-Pending-blue?style=flat-square) |
| 🪄 Gamified Learning Flow     | Planned      | ![Gamify](https://img.shields.io/badge/Gamification-Planned-blueviolet?style=flat-square) |
| 🌐 Global Deployment          | Future       | ![Global](https://img.shields.io/badge/Deployment-Soon-lightgrey?style=flat-square) |

---

# 📦 Installation

## Clone the repository
```bash
git clone https://github.com/CryptoLab-service/OptiGrade-ML-model.git
cd optigrade
```
## Virtual environment setup
```bash
python -m venv optigrade_env
optigrade-env\Scripts\activate    # Windows
source optigrade_env/bin/activate # macOS/Linux
```
## Dependencies
```bash
pip install -r requirements.txt
```
## Environment config
```bash
cp .env.example .env
# Add your Gemini API key
```

## Launch app
```bash
streamlit run optigrade_app.py
```

---

# 🐳 For Docker Setup

## Prerequisites
- [Install Docker](https://docs.docker.com/get-docker/)

## Build the Docker Image
```bash
docker build -t optigrade-app .


---

# ⚙️ Automated Setup

### 🪟 Windows (.bat)

```bash
install.bat
```

### 💻 PowerShell (.ps1)
```bash
.\setup.ps1
```
These scripts will:
- Set up your virtual environment
- Install required dependencies
- Copy .env config file
- Retrain the prediction model
- Launch the Streamlit app

---

# 🔁 Model Retraining
To retrain CGPA prediction:

## 1. Place your dataset at
```data/training_data.csv```

## 2. Columns expected:
- credit_load
- study_hours
- GPA_last_semester
- current_CGPA
- target_CGPA

## 3. Run training script:
```bash
python models/train_model.py
```

## 4. Generated model file:
```bash
models/model.pkl
```

---

# 🤝 Join the OptiGrade Mission

**OptiGrade** began as a one-developer vision. Now it’s a call for collaboration. Help expand access to intelligent learning tools worldwide.

We welcome:

- 🏫 **Institutions** — Integrate with LMS systems  
- 💻 **Developers** — Build open-source features  
- 🧠 **Experts** — Review, challenge, and strengthen the logic  
- 🎯 **Sponsors** — Fund innovation and empower students globally  

> Together, we build tech that guides students smarter.

---

# 👥 Contributors

<!-- ALL-CONTRIBUTORS-LIST:START -->
| [<img src="https://avatars.githubusercontent.com/u/121734918?v=4" width="75px;"/><br><sub><b>Toluwalope Oluwalowo</b></sub>](https://github.com/CryptoLab-service) |
| :---: |
| 💻 🚀 🧠 🎨 📖 ⚙️ 💡 📋 |
<!-- ALL-CONTRIBUTORS-LIST:END -->

# 💬 Want your avatar on this grid?  
Submit a [Pull Request](https://github.com/CryptoLab-service/OptiGrade-ML-model/pulls) or join [Discussions](https://github.com/CryptoLab-service/OptiGrade-ML-model/discussions)!

## 📄 License

This project is licensed under the **MIT License**.  
Feel free to use, modify, and share with proper attribution.

👉 [View Full License](LICENSE)

## 💬 Feedback & Contact

Always open to feedback, questions, and new ideas.  
Reach out via [GitHub Discussions](https://github.com/CryptoLab-service/OptiGrade-ML-model/discussions) or visit the [Streamlit Demo](https://optigrade.streamlit.app/)

> Built with 💙 for students everywhere.
