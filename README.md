# DiscretizedDiffusion

Render subsurface diffusion via convolution over vertices on the mesh.

The current version looks like this:

![rendering result](https://github.com/LeonKang130/DiscretizedDiffusion/blob/main/result-scene.png)

## Todo

The current version spends a lot of time on transferring data from *LuisaCompute* framework, which is on the GPU side,
to *numpy* on CPU and then back to *OpenGL* on GPU.
Obviously, this works against the goal of reaching real-time translucency rendering.

- Move the uploading to *OpenGL* prior to the calculation of efflux and turn to compute shader in *OpenGL* for
  calculation of efflux, which can be stored nice and neatly in a buffer ready for further rendering.
