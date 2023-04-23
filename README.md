# DiscretizedDiffusion

Render subsurface scattering via convolution over vertices on the mesh.

![teaser](https://github.com/LeonKang130/DiscretizedDiffusion/blob/main/teaser.png)

The method can be divided into three steps:

- Gather influx around each vertex of the mesh using path tracing(or rasterization if no surface light is involved in the scene).
- Transfer the influx buffer to OpenGL.
- Use compute shader to do a $O(n^2)$ convolution over the mesh, transforming the influx into efflux.
- Use the efflux(color) at each vertex to do fragment shading.

To use the repo, pass a `json` file as the first argument to config the scene. You may refer to `scenes/*.json` for examples of such configuration.
