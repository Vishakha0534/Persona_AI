---
title: PersonaAI
emoji: 🤖
colorFrom: blue
colorTo: purple
sdk: docker
app_file: app.py
pinned: false
---

## 🚀 PersonaAI
## 🧠 Overview

PersonaAI is a lightweight Reinforcement Learning (RL) based environment where an agent interacts with a custom-built system and learns decision-making through rewards and penalties.

This project focuses on understanding the core RL loop: State → Action → Reward → Next State.

## 🎯 Objective

To build a simple RL environment where an agent can:

Observe environment state
Take actions
Receive rewards/penalties
Learn optimal behavior over time
## 🌍 Environment Description

The environment is a custom simulation system built in Python.

## ⚙️ Key Features:
Episodic structure (each run is a new episode)
Limited steps per episode
Resource-constrained system
Task-based environment setup (e.g., easy mode)
Reward-based feedback mechanism
## 🔁 RL Cycle:
Environment resets
Agent observes state
Agent takes action
Environment returns next state + reward
Episode continues until termination condition
## 📊 Observation Space

The observation space represents the current state of the environment.

## 🧠 State includes:
Current step count
Available resources (e.g., beds or capacity)
Task type (easy/medium/hard)
Current environment status (patients/tasks lists)

## 🎮 Action Space

The agent can take discrete actions based on the current state.

## ⚙️ Actions:
0 → Skip / Do nothing
1 → Accept / Allocate resource
2 → Reject task

## 🏗️ Project Structure
PersonaAI/
│
├── app.py                 # Main entry point
├── inference.py           # Agent logic
├── env/
│   └── environment.py     # Custom RL environment
├── openenv.yaml           # Configuration file
├── requirements.txt       # Dependencies
└── README.md

## 🚀 Setup Instructions
1. Clone repository
git clone https://github.com/Vishakha0534/Persona_AI.git
cd PersonaAI
2. Create virtual environment
python -m venv venv
venv\Scripts\activate   # Windows
3. Install dependencies
pip install -r requirements.txt
4. Run project
python app.py

## 🧪 Tech Stack
Python 
Reinforcement Learning (RL) basics 
Custom environment design
YAML configuration

## 🔮 Future Improvements
🌐 Frontend dashboard for visualization
🧠 Improved memory-based learning agent
🗺️ Google Maps integration

## 👩‍💻 Author

Vishakha Solanki

## 🏆 Key Highlights
1. Demonstrates RL environment design
2. Implements state-action-reward loop
3. Lightweight and modular architecture
4. Beginner-friendly reinforcement learning project
5. Deployed on Hugging Face Spaces using Docker for reproducible environment execution.