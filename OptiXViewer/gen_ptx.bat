nvcc.exe --machine=64 --ptx --gpu-architecture=compute_75 --use_fast_math --relocatable-device-code=true --generate-line-info -Wno-deprecated-gpu-targets devicePrograms.cu -ccbin "C:\Program Files (x86)\Microsoft Visual Studio\2019\Professional\VC\Tools\MSVC\14.29.30133\bin\Hostx64\x64" -o devicePrograms.ptx -I"C:\ProgramData\NVIDIA Corporation\OptiX SDK 7.4.0\include" -I"3rdParty" -I.
