#!/bin/bash
source .venv/bin/activate
poetry run uvicorn app.main:app --reload