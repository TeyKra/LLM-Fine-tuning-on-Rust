@echo off

:: Path to your project directory
set "PROJECT_DIR=E:\AI\projets\LLM project efrei\llm_project_M2"

:: Path to the virtual environment directory
set "VENV_DIR=%PROJECT_DIR%\myenv"

:: Path to the requirements file
set "REQUIREMENTS_FILE=%PROJECT_DIR%\requirements.txt"

:: Check if the virtual environment already exists
IF NOT EXIST "%VENV_DIR%" (
    echo Creating virtual environment...
    python -m venv "%VENV_DIR%"
    
    :: Activate the virtual environment
    call "%VENV_DIR%\Scripts\activate"

    :: Upgrade pip to the latest version
    pip install --upgrade pip

    :: Install required packages from requirements.txt
    pip install -r "%REQUIREMENTS_FILE%"
    pip install tf-keras

    :: Deactivate after installation
    call deactivate
) ELSE (
    echo Virtual environment already exists.
)

:: Pause to keep the window open
echo Script execution finished. Press any key to exit.
pause