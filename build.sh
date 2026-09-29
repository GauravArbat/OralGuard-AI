#!/usr/bin/env bash
# Render Build Script for OralGuard AI

set -o errexit

echo ">>> Installing Python dependencies..."
pip install --upgrade pip
pip install -r backend/requirements.txt

echo ">>> Build complete!"
