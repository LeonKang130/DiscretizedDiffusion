# DiscretizedDiffusion

Render subsurface diffusion via convolution over vertices on the mesh.

The current version looks like this:

![rendering result](https://github.com/LeonKang130/DiscretizedDiffusion/blob/main/result-scene.png)

The composition of time spent to render one frame is approximately as following:

- Collect vertex influx using ray tracing: 20ms
- Transfer influx data from LC to OpenGL: 200ms
- Compute shader && vertex lighting: 5ms
