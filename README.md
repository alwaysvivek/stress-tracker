# 🧠 StressTracker AI

> **"Your digital wellbeing, monitored in real-time."**

![Status](https://img.shields.io/badge/Status-Production%20Ready-brightgreen)
![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red)

**StressTracker AI** is a privacy-focused, local-first application that analyzes your digital biomarkers (mouse movements, keystroke dynamics) to detect stress and cognitive load in real-time.

---

## ✨ Features

- **🖱️ Kinematic Analysis**: Tracks mouse jitter, path efficiency, and velocity to detect SNS (Sympathetic Nervous System) activation.
- **⌨️ Keystroke Dynamics**: Analyzes flight time variance (ISO 9241-11) as a proxy for cognitive load.
- **🤖 AI Agent**: Integrated Llama-3.2 based clinical psychiatrist persona provides actionable text-based feedback.
- **📊 Real-time Dashboard**: Beautiful "Deep Space" aesthetic UI built with Streamlit.
- **🔒 Privacy First**: All tracking happens locally on your machine. No raw input data leaves your device.

## 🛠️ Tech Stack

- **Frontend**: Streamlit (Python)
- **Backend/Logic**: Python (Pydantic, NumPy, SciPy)
- **Input Tracking**: `pynput` (System-wide hooks)
- **AI/LLM**: Llama-3.2 (via LangChain & Ollama)
- **Architecture**: Monolithic Local App

## 🚀 How to Use

### 1. Prerequisites
- Python 3.9+
- [Ollama](https://ollama.ai) installed (The script will automatically pull the model for you!)

### 2. Installation & Run (All Platforms)

We have a magic script that sets everything up for you (Virtual Environment + Dependencies + App Launch).

```bash
# 1. Clone the repository
git clone https://github.com/yourusername/stresstracker.git
cd stresstracker

# 2. Run the magic script
python start.py
```

That's it! The app will open in your browser.

---

## 🌍 Why "Local Only"?

You might wonder why this isn't a hosted web app.
> [!IMPORTANT]
> **System-Wide Tracking**: To analyze real work stress, the app needs to track mouse/keyboard usage *outside* the browser window (e.g., while you use Excel, Slack, IDEs). Web browsers strictly block this for security. Therefore, this app **must run locally** on your machine to access these operating system hooks via `pynput`.

## 🤝 Contributing

1.  Fork the Project
2.  Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3.  Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4.  Push to the Branch (`git push origin feature/AmazingFeature`)
5.  Open a Pull Request

---
*Built with ❤️ for Digital Health*
