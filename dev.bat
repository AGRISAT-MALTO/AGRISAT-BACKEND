@echo off
"%~dp0.venv\Scripts\python.exe" -m uvicorn app.main:app --reload --port 3001
