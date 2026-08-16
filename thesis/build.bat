@echo off
REM ============================================================
REM  Build the thesis PDF.
REM  Double-click this file, or run it from any terminal.
REM
REM  Uses pdflatex/bibtex directly (latexmk needs Perl, which is
REM  not installed). Runs from this file's own folder so the
REM  relative \input{} paths resolve.
REM ============================================================

cd /d "%~dp0"
set MIKTEX_ENABLE_INSTALLER=t

echo.
echo === pass 1 ===
pdflatex -interaction=nonstopmode -file-line-error main.tex

REM pdflatex reads main.toc/.lof/.lot from the previous run before rewriting
REM them. If they came from a structurally different version of the document,
REM pass 1 dies in a cascade of "perhaps a missing \item". Discard and retry.
findstr /c:"perhaps a missing" main.log >nul
if not errorlevel 1 (
  echo   stale .toc/.aux detected - discarding and repeating pass 1 ...
  del /q main.aux main.toc main.lof main.lot main.out 2>nul
  del /q chapters\*.aux frontmatter\*.aux 2>nul
  pdflatex -interaction=nonstopmode -file-line-error main.tex
)

echo.
echo === bibtex ===
bibtex main

echo.
echo === pass 2 ===
pdflatex -interaction=nonstopmode -file-line-error main.tex

echo.
echo === pass 3 ===
pdflatex -interaction=nonstopmode -file-line-error main.tex

echo.
echo ============================================================
if exist main.pdf (
  echo  BUILD OK  -^>  %~dp0main.pdf
) else (
  echo  BUILD FAILED - no PDF produced.
  echo  Scroll up for the first line starting with "!"
)
echo ============================================================
echo.
pause
