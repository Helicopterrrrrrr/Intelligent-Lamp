@echo off
setlocal

set "ROOT=%~dp0"
set "PYTHON=%USERPROFILE%\anaconda3\envs\smartled\python.exe"
set "BACKEND_URL=http://127.0.0.1:5000"
set "VISION_TOKEN=change-me"
set "DEVICE_TOKEN=change-me"
set "YOLO_CONFIG_DIR=%ROOT%output"
set "MPLCONFIGDIR=%ROOT%output\matplotlib"

if not exist "%PYTHON%" (
  echo Python not found: %PYTHON%
  echo Please update PYTHON in run_vision.cmd.
  pause
  exit /b 1
)

set "CAMERA_PROFILE=%SMARTLAMP_CAMERA_PROFILE%"
set "CAMERA_SOURCE=%SMARTLAMP_CAMERA_SOURCE%"
set "SOURCE_ID=%SMARTLAMP_SOURCE_ID%"
set "CAPTURE_BACKEND=%SMARTLAMP_CAPTURE_BACKEND%"

if not "%~1"=="" set "CAMERA_PROFILE=%~1"
if "%CAMERA_PROFILE%"=="" set "CAMERA_PROFILE=phone"

if /i "%CAMERA_PROFILE%"=="phone" goto phone_profile
if /i "%CAMERA_PROFILE%"=="board" goto board_profile
goto custom_profile

:phone_profile
if not "%~2"=="" set "CAMERA_SOURCE=%~2"
if "%CAMERA_SOURCE%"=="" set "CAMERA_SOURCE=0"
if "%SOURCE_ID%"=="" set "SOURCE_ID=phone-camera"
if "%CAPTURE_BACKEND%"=="" set "CAPTURE_BACKEND=dshow"
goto run_worker

:board_profile
if not "%~2"=="" set "SMARTLAMP_BOARD_CAMERA_URL=%~2"
if "%CAMERA_SOURCE%"=="" if not "%SMARTLAMP_BOARD_CAMERA_URL%"=="" set "CAMERA_SOURCE=%SMARTLAMP_BOARD_CAMERA_URL%"
if "%SOURCE_ID%"=="" set "SOURCE_ID=esp32-camera"
if "%CAPTURE_BACKEND%"=="" set "CAPTURE_BACKEND=auto"
goto run_worker

:custom_profile
if "%CAMERA_SOURCE%"=="" set "CAMERA_SOURCE=%CAMERA_PROFILE%"
if "%SOURCE_ID%"=="" set "SOURCE_ID=custom-camera"
if "%CAPTURE_BACKEND%"=="" set "CAPTURE_BACKEND=auto"
set "CAMERA_PROFILE=custom"
goto run_worker

:run_worker
echo Starting YOLO vision worker...
echo Profile: %CAMERA_PROFILE%
echo Source: %CAMERA_SOURCE%
if "%CAMERA_SOURCE%"=="" echo Source: ^(from vision\camera_sources.json or SMARTLAMP_BOARD_CAMERA_URL^)
echo Source ID: %SOURCE_ID%
echo Capture backend: %CAPTURE_BACKEND%

cd /d "%ROOT%"
if not exist "%YOLO_CONFIG_DIR%" mkdir "%YOLO_CONFIG_DIR%"
if not exist "%MPLCONFIGDIR%" mkdir "%MPLCONFIGDIR%"
if /i "%SMARTLAMP_DRY_RUN%"=="1" (
  echo Dry run only. Worker command:
  echo "%PYTHON%" vision\run_vision_worker.py --camera-profile "%CAMERA_PROFILE%" --source "%CAMERA_SOURCE%" --source-id "%SOURCE_ID%" --capture-backend "%CAPTURE_BACKEND%" --backend-url "%BACKEND_URL%" --vision-token "%VISION_TOKEN%" --device-token "%DEVICE_TOKEN%" --preview
  exit /b 0
)

"%PYTHON%" vision\run_vision_worker.py --camera-profile "%CAMERA_PROFILE%" --source "%CAMERA_SOURCE%" --source-id "%SOURCE_ID%" --capture-backend "%CAPTURE_BACKEND%" --backend-url "%BACKEND_URL%" --vision-token "%VISION_TOKEN%" --device-token "%DEVICE_TOKEN%" --preview

endlocal
