@echo off
cd /d "%~dp0"
if exist "%USERPROFILE%\anaconda3\Scripts\activate.bat" call "%USERPROFILE%\anaconda3\Scripts\activate.bat"
python -c "import numba" 2>nul || pip install numba
if not exist out mkdir out
set AME_OUT=out
echo [1/2] Validation: AME(6,4), expected TOTAL 1552
python fast_search.py 3 > out\validation_ame64.log 2>&1
findstr /c:"TOTAL" out\validation_ame64.log
echo [2/2] AME(8,4): one window per row-1 class (main search, then independent check)
for %%c in (0 1 2 3 4 5 6 7 8 9) do start "AME84 class %%c" /min cmd /c "python fast_search.py 4 %%c > out\class%%c.log 2>&1 & python subspace_check.py all %%c > out\subspace_class%%c.log 2>&1"
echo Running in the background. Results go to the out folder.
pause
