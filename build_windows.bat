call "C:\Program Files (x86)\Intel\oneAPI\setvars.bat" intel64 vs2022
cd ops\build
cmake.exe -G Ninja -DCMAKE_BUILD_TYPE=Release -DPYTHON_EXECUTABLE="C:/Users/tingqian/AppData/Local/Programs/Python/Python38/python38.exe" ..
cmake.exe --build . --config Release --verbose
pause
