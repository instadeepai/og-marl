## Troubleshooting

In MAMuJoCo you may need to export the following environment variables.

`export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/x86_64-linux-gnu/:$MUJOCOPATH/mujoco210/bin:/usr/lib/nvidia`

`export LD_PRELOAD=$LD_PRELOAD:/usr/lib/x86_64-linux-gnu/libGLEW.so`

Error:

`ImportError: libpython3.8.so.1.0: cannot open shared object file: No such file or directory`

Solution:

`export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:\$LD_LIBRARY_PATH`
