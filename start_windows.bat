@echo off
rem Start ComfyUI with the VigenFlow nodes (Windows version of start.sh).
rem
rem   start_windows.bat                 then open http://127.0.0.1:8188
rem   start_windows.bat --port 8190     extra arguments go to ComfyUI's main.py
rem
rem Cloned into ComfyUI\custom_nodes\: starts that ComfyUI. Cloned anywhere else: links this folder
rem (a directory junction) into %COMFYUI_DIR%\custom_nodes\ (default %USERPROFILE%\ComfyUI) and starts
rem that ComfyUI. ComfyUI runs with --cpu (its own PyTorch nodes on the CPU); the VigenFlow nodes
rem always run on the NPU. COMFYUI_PYTHON chooses the Python; by default ComfyUI's .venv, venv or the
rem portable build's python_embeded, else python.
setlocal EnableDelayedExpansion

set "NODE_DIR=%~dp0"
set "NODE_DIR=%NODE_DIR:~0,-1%"
for %%I in ("%NODE_DIR%") do set "NODE_NAME=%%~nxI"
for %%I in ("%NODE_DIR%\..") do set "PARENT=%%~fI"
for %%I in ("%PARENT%") do set "PARENT_NAME=%%~nxI"

if /i "%PARENT_NAME%"=="custom_nodes" if exist "%PARENT%\..\main.py" (
    for %%I in ("%PARENT%\..") do set "COMFY=%%~fI"
    goto found
)
if defined COMFYUI_DIR (set "COMFY=%COMFYUI_DIR%") else (set "COMFY=%USERPROFILE%\ComfyUI")
if not exist "%COMFY%\main.py" (
    echo ComfyUI was not found at %COMFY%. Set COMFYUI_DIR to your ComfyUI folder, 1>&2
    echo or clone this repository into ComfyUI\custom_nodes\. 1>&2
    exit /b 1
)
for %%I in ("%COMFY%") do set "COMFY=%%~fI"
:found

rem Another copy of these nodes in the same ComfyUI would replace them (same node names).
set "LINK=%COMFY%\custom_nodes\%NODE_NAME%"
for /d %%D in ("%COMFY%\custom_nodes\*") do (
    if exist "%%~fD\nodes.py" if /i not "%%~fD"=="%NODE_DIR%" if /i not "%%~fD"=="!LINK!" (
        findstr /c:"class VigenFlowTextToImage" "%%~fD\nodes.py" >nul 2>&1 && (
            echo Another copy of the VigenFlow nodes is installed: %%~fD 1>&2
            echo Remove it ^(or move it out of custom_nodes^) and start again. 1>&2
            exit /b 1
        )
    )
)

rem Cloned elsewhere: link this folder into that ComfyUI.
if /i not "%PARENT%"=="%COMFY%\custom_nodes" (
    if exist "%LINK%\nodes.py" (
        fc /b "%LINK%\nodes.py" "%NODE_DIR%\nodes.py" >nul 2>&1 || (
            echo %LINK% already exists and is not this folder ^(%NODE_DIR%^). 1>&2
            exit /b 1
        )
    ) else if exist "%LINK%" (
        echo %LINK% already exists and is not this folder ^(%NODE_DIR%^). 1>&2
        exit /b 1
    ) else (
        if not exist "%COMFY%\custom_nodes" mkdir "%COMFY%\custom_nodes"
        mklink /J "%LINK%" "%NODE_DIR%" >nul || exit /b 1
        echo Linked %NODE_DIR% into %COMFY%\custom_nodes\
    )
)

cd /d "%COMFY%"
set "PY=python"
if exist ".venv\Scripts\python.exe" (
    set "PY=.venv\Scripts\python.exe"
) else if exist "venv\Scripts\python.exe" (
    set "PY=venv\Scripts\python.exe"
) else if exist "..\python_embeded\python.exe" (
    set "PY=..\python_embeded\python.exe"
)
if defined COMFYUI_PYTHON set "PY=%COMFYUI_PYTHON%"

echo Starting ComfyUI in %COMFY%
if defined VIGENFLOW_DRY_RUN (
    echo %PY% main.py --cpu %*
    exit /b 0
)
rem The NPU progress bars print UTF-8 block characters: show them in the UTF-8 code page,
rem then give the console back its own.
for /f "tokens=2 delims=:." %%c in ('chcp') do set /a "VGF_CODE_PAGE=%%c"
chcp 65001 >nul
set PYTHONIOENCODING=utf-8
"%PY%" main.py --cpu %*
set "VGF_EXIT=%ERRORLEVEL%"
if defined VGF_CODE_PAGE chcp %VGF_CODE_PAGE% >nul
exit /b %VGF_EXIT%
