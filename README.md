# DiscretizedDiffusion

Render subsurface diffusion via convolution over vertices on the mesh.

The current version looks as follow:

![normalized bssrdf](https://github.com/LeonKang130/DiscretizedDiffusion/blob/main/result-scene-normalized.png)

![dipole](https://github.com/LeonKang130/DiscretizedDiffusion/blob/main/result-scene-dipole.png)

The composition of time spent to render one frame is approximately as following:

- Collect vertex influx using ray tracing: 20ms
- Transfer influx data from LC to OpenGL: 200ms
- Compute shader && vertex lighting: 15ms

To use the repo, pass a `json` file to config the scene. You may refer to `scenes/*.json` for examples.

Update: Both **Normalized BSSRDF** and **Dipole** can be selected for the equation of diffusion. To do so, you can set the `equation` field of the config file to **"dipole"** or **"normalized"**.
