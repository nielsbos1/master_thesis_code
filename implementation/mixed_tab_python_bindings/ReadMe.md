# How to create your own 

- Clone pybind11 repository in this folder
- Run commands below

```
cd build
cmake -DCMAKE_CXX_COMPILER="C:/msys64/mingw64/bin/g++"  -DCMAKE_C_COMPILER="C:/msys64/mingw64/bin/gcc" --debug-trycompile  .. -G "MinGW Makefiles"
cd ..
cmake --build build
```
