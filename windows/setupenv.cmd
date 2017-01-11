REM @echo off

setlocal EnableExtensions EnableDelayedExpansion

REM Remove duplicate entries in PATH to avoid long PATH.
set _PATH_=
for %%a in ("%PATH:;=" "%") do @if not "%%~a" == "" (
if "!_PATH_!" == "" @set "_PATH_=;%%~a;"
set "_T_=!_PATH_:;%%~a;=x!"
if "!_T_!" == "!_PATH_!" @set "_PATH_=!_PATH_!%%~a;"
)
set "_path=%_PATH_:~1,-1%
set _path=%_path%;

if [%MXNET_HOME%] == [] GOTO Define
    echo Deleting original install
    set _root=%MXNET_HOME%
    set _dirs=(%_root%\3rdparty\openblas\bin %_root%\3rdparty\vc %_root%\3rdparty\cudart %_root%\3rdparty\cudnn\bin %_root%\lib %_root%\3rdparty\opencv)
    for %%d in %_dirs% do set _path=!_path:%%d;=!
    echo PATH=%_path%

:Define
    setlocal EnableExtensions EnableDelayedExpansion
    echo Defining new environmental variable...
    set _root=%~dp0
    setx MXNET_HOME %_root%
    set _root=%%MXNET_HOME%%
    set _dirs=(%_root%\3rdparty\openblas\bin %_root%\3rdparty\vc %_root%\3rdparty\cudart %_root%\3rdparty\cudnn\bin %_root%\lib %_root%\3rdparty\opencv)
    for %%d in %_dirs% do set _path=%%d;!_path!
    setx PATH "%_path%"
    echo MXNET_HOME=%~dp0
    echo PATH=%_path%