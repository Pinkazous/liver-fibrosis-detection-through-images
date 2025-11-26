@echo off
setlocal
set PROJECTPATH=%cd%
set PYTHONDIR=%PYTHONPATH%
echo "Python installed at: '%PYTHONDIR%'"
echo "My project path is: '%PROJECTPATH%'"
set MAINPATH=%PROJECTPATH%\app.py
set PATH=%PYTHONDIR%;%PATH%
echo "Run command is: 'python.exe -m streamlit run "%MAINPATH%"'"
python.exe -m streamlit run "%MAINPATH%"
endlocal
pause

