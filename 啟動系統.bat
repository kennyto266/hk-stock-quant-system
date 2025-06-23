@echo off
chcp 65001 >nul
title 港股量化分析系統

echo.
echo ========================================
echo       🏦 港股量化分析系統
echo ========================================
echo.

cd /d "%~dp0"

echo 🔍 檢查 Python 環境...
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python 未安裝或不在 PATH 中
    echo 💡 請安裝 Python 3.8+ 並添加到系統路徑
    pause
    exit /b 1
)

echo ✅ Python 環境正常
echo.

echo 🚀 啟動港股量化分析系統...
python run_analysis.py

echo.
echo 📝 系統已退出
pause 