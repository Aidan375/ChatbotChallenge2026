@echo off
REM Runs the build steps from HOW_TO_BUILD.md. Output goes to build_log.txt.
cd /d "%~dp0"
set PY=..\venv\Scripts\python.exe
set PYTHONIOENCODING=utf-8
set PYTHONUNBUFFERED=1
echo === START %date% %time% > build_log.txt
echo === STEP 1 pip install >> build_log.txt
%PY% -m pip install -r requirements.txt >> build_log.txt 2>&1 || goto fail
echo === STEP 1b check_setup (stops here if the AI server refuses) >> build_log.txt
%PY% check_setup.py >> build_log.txt 2>&1 || goto fail
echo === STEP 2 build.index >> build_log.txt
%PY% -m build.index >> build_log.txt 2>&1 || goto fail
echo === STEP 3 build.images budget 150 >> build_log.txt
%PY% -m build.images --budget 150 >> build_log.txt 2>&1 || goto fail
echo === STEP 4 score >> build_log.txt
%PY% score.py >> build_log.txt 2>&1
echo === STEP 5 make_submission >> build_log.txt
%PY% make_submission.py >> build_log.txt 2>&1 || goto fail
echo === DONE %date% %time% >> build_log.txt
exit /b 0
:fail
echo === FAILED %date% %time% >> build_log.txt
exit /b 1
