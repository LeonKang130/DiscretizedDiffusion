from __future__ import annotations

import json
import math
import time
from typing import NamedTuple, List

import moderngl
import numpy as np
import tinyobjloader
from matplotlib import pyplot as plt
from pyrr import Vector3
import sys
import luisa
from camera import Camera
from luisa import RandomSampler
from luisa.mathtypes import *
import OpenGL.GL as gl

accel: luisa.Accel = None
vertex_buffer: luisa.Buffer = None
normal_buffer: luisa.Buffer = None
triangle_buffer: luisa.Buffer = None
surface_light_buffer: luisa.Buffer = None
point_light_buffer: luisa.Buffer = None
direction_light_buffer: luisa.Buffer = None
DeviceDirectionLight = luisa.StructType(direction=float3, emission=float3)
DevicePointLight = luisa.StructType(position=float3, emission=float3)
DeviceSurfaceLight = luisa.StructType(emission=float3)
light_area = 0.0
light_normal = make_float3(0.0, -1.0, 0.0)
amplifier = 6.218


def calculate_parameters(sigma_a: float3, sigma_s: float3, g: float, eta: float):
    sigma_s_prime = sigma_s * (1.0 - g)
    sigma_t_prime = sigma_s_prime + sigma_a
    alpha_prime = sigma_s_prime / sigma_t_prime
    fresnel = -1.440 / (eta * eta) + 0.710 / eta + 0.668 + 0.0636 * eta
    a = (1.0 + fresnel) / (1.0 - fresnel)
    albedo = 0.5 * alpha_prime * (1.0 + make_float3(
        math.exp(-4.0 / 3.0 * a * math.sqrt(3.0 * (1.0 - alpha_prime.x))),
        math.exp(-4.0 / 3.0 * a * math.sqrt(3.0 * (1.0 - alpha_prime.y))),
        math.exp(-4.0 / 3.0 * a * math.sqrt(3.0 * (1.0 - alpha_prime.z)))
    )) * make_float3(
        math.exp(-math.sqrt(3.0 * (1.0 - alpha_prime.x))),
        math.exp(-math.sqrt(3.0 * (1.0 - alpha_prime.y))),
        math.exp(-math.sqrt(3.0 * (1.0 - alpha_prime.z)))
    )
    sigma_tr = make_float3(
        math.sqrt(3.0 * (1.0 - alpha_prime.x)),
        math.sqrt(3.0 * (1.0 - alpha_prime.y)),
        math.sqrt(3.0 * (1.0 - alpha_prime.z))
    ) * sigma_t_prime
    dmfp = 1.0 / ((3.5 + 100 * make_float3(
        (albedo.x - 0.33) ** 4,
        (albedo.y - 0.33) ** 4,
        (albedo.z - 0.33) ** 4
    )) * sigma_tr)
    return dmfp, albedo


dmfp = make_float3(0.1)
albedo = make_float3(1.0)
ctx: moderngl.Context = None


class RenderModel(NamedTuple):
    vertex_buffer: luisa.Buffer
    vertex_buffer_object: moderngl.Buffer
    normal_buffer: luisa.Buffer
    triangle_buffer: luisa.Buffer
    index_buffer_object: moderngl.Buffer

    @staticmethod
    def from_file(filename: str) -> RenderModel:
        reader = tinyobjloader.ObjReader()
        if not reader.ParseFromFile(filename):
            print(f"Warning: Error occurred when parsing file '{filename}'")
            exit(-1)
        attrib = reader.GetAttrib()
        vertex_buffer = luisa.Buffer.empty(len(attrib.vertices) // 3, dtype=float4)
        normal_buffer = luisa.Buffer.empty(vertex_buffer.size, dtype=float4)
        vertex_arr = np.hstack(
            (
                vertex_arr := np.array(attrib.vertices, dtype=np.float32).reshape(-1, 3),
                np.zeros((vertex_arr.shape[0], 1))
            )
        ).astype(np.float32)
        vertex_buffer.copy_from_array(vertex_arr)
        normal_vectors = np.array(attrib.normals, dtype=np.float32).reshape(-1, 3)
        normal_vectors /= np.linalg.norm(normal_vectors, axis=1).reshape((-1, 1))
        normal_arr = np.empty((vertex_arr.shape[0], 3), dtype=np.float32)
        for shape in reader.GetShapes():
            for index in shape.mesh.indices:
                normal_arr[index.vertex_index] = normal_vectors[index.normal_index]
        normal_arr = np.hstack(
            (
                normal_arr,
                np.zeros((normal_arr.shape[0], 1))
            )
        ).astype(np.float32)
        normal_buffer.copy_from_array(normal_arr)
        shapes = reader.GetShapes()
        triangle_arr = np.array([
            index.vertex_index for shape in shapes for index in shape.mesh.indices
        ], dtype=np.int32)
        triangle_buffer = luisa.Buffer.empty(len(triangle_arr), dtype=int)
        triangle_buffer.copy_from_array(triangle_arr)
        return RenderModel(
            vertex_buffer,
            ctx.buffer(vertex_arr),
            normal_buffer,
            triangle_buffer,
            ctx.buffer(triangle_arr)
        )


class SurfaceLight(NamedTuple):
    vertex_buffer: luisa.Buffer
    triangle_buffer: luisa.Buffer
    emission: float3

    @staticmethod
    def from_file(filename: str, emission: float3 = make_float3(0.0)) -> SurfaceLight:
        global light_area, light_normal
        reader = tinyobjloader.ObjReader()
        if not reader.ParseFromFile(filename):
            print(f"Warning: Error occurred when parsing file '{filename}'")
            exit(-1)
        attrib = reader.GetAttrib()
        vertex_buffer = luisa.Buffer.empty(len(attrib.vertices) // 3, dtype=float4)
        vertex_buffer.copy_from_array(np.hstack(
            (
                vertex_arr := np.array(attrib.vertices, dtype=np.float32).reshape(-1, 3),
                np.zeros((vertex_arr.shape[0], 1))
            )
        ).astype(np.float32))
        shapes = reader.GetShapes()
        triangle_arr = np.array([
            index.vertex_index for shape in shapes for index in shape.mesh.indices
        ], dtype=np.int32)
        triangle_buffer = luisa.Buffer.empty(len(triangle_arr), dtype=int)
        triangle_buffer.copy_from_array(triangle_arr)
        light_area = float(np.linalg.norm(np.cross(vertex_arr[triangle_arr[1]] - vertex_arr[triangle_arr[0]], vertex_arr[triangle_arr[2]] - vertex_arr[triangle_arr[0]])))
        light_normal = make_float3(*np.cross(vertex_arr[triangle_arr[1]] - vertex_arr[triangle_arr[0]], vertex_arr[triangle_arr[2]] - vertex_arr[triangle_arr[0]])) / light_area
        print(f"Surface light direction: {light_normal}")
        print(f"Surface light contribution: {light_area * emission}")
        return SurfaceLight(vertex_buffer, triangle_buffer, emission)


class DirectionLight(NamedTuple):
    direction: float3
    emission: float3


class PointLight(NamedTuple):
    position: float3
    emission: float3


class Scene(NamedTuple):
    render_model: RenderModel
    surface_lights: List[SurfaceLight]
    direction_lights: List[DirectionLight]
    point_lights: List[PointLight]


def parse_scene(filename: str) -> Scene:
    global vertex_buffer, normal_buffer, triangle_buffer, dmfp, albedo
    with open(filename, 'r') as file:
        scene_data = json.load(file)
        dmfp, albedo = calculate_parameters(make_float3(*scene_data["sigma_a"]), make_float3(*scene_data["sigma_s"]),
                                            scene_data["g"],
                                            scene_data["eta"])
        render_model = RenderModel.from_file(scene_data["render_model"])
        vertex_buffer = render_model.vertex_buffer
        normal_buffer = render_model.normal_buffer
        triangle_buffer = render_model.triangle_buffer
        surface_lights, direction_lights, point_lights = [], [], []
        for light in scene_data["surface_lights"]:
            surface_lights.append(SurfaceLight.from_file(light["model"], make_float3(*light["emission"])))
        for light in scene_data["direction_lights"]:
            direction_lights.append(DirectionLight(make_float3(*light["direction"]), make_float3(*light["emission"])))
        for light in scene_data["point_lights"]:
            point_lights.append(PointLight(make_float3(*light["position"]), make_float3(*light["emission"])))
        return Scene(render_model, surface_lights, direction_lights, point_lights)


def upload_scene(scene: Scene) -> None:
    global accel, surface_light_buffer, point_light_buffer, direction_light_buffer
    accel = luisa.Accel()
    accel.add(luisa.Mesh(scene.render_model.vertex_buffer, scene.render_model.triangle_buffer))
    surface_light_buffer = luisa.Buffer.empty(max(len(scene.surface_lights), 1), dtype=DeviceSurfaceLight)
    if scene.surface_lights:
        surface_light_buffer.copy_from_list(
            [DeviceSurfaceLight(emission=light.emission) for light in scene.surface_lights])
    for light in scene.surface_lights:
        accel.add(luisa.Mesh(light.vertex_buffer, light.triangle_buffer))
    point_light_buffer = luisa.Buffer.empty(max(len(scene.point_lights), 1), dtype=DevicePointLight)
    if scene.point_lights:
        point_light_buffer.copy_from_list(
            [DevicePointLight(position=light.position, emission=light.emission) for light in scene.point_lights])
    direction_light_buffer = luisa.Buffer.empty(max(len(scene.direction_lights), 1), dtype=DeviceDirectionLight)
    if scene.direction_lights:
        direction_light_buffer.copy_from_list(
            [DeviceDirectionLight(direction=light.direction, emission=light.emission) for light in
             scene.direction_lights])
    accel.update()


Onb = luisa.StructType(tangent=float3, binormal=float3, normal=float3)


@luisa.func
def to_world(self, v):
    return v.x * self.tangent + v.y * self.binormal + v.z * self.normal


Onb.add_method(to_world, "to_world")


@luisa.func
def make_onb(normal):
    binormal = normalize(select(
        make_float3(0.0, -normal.z, normal.y),
        make_float3(-normal.y, normal.x, 0.0),
        abs(normal.x) > abs(normal.z)))
    tangent = normalize(cross(binormal, normal))
    result = Onb()
    result.tangent = tangent
    result.binormal = binormal
    result.normal = normal
    return result


@luisa.func
def cosine_sample_hemisphere(u):
    r = sqrt(u.x)
    phi = 2.0 * 3.1415926 * u.y
    return make_float3(r * cos(phi), r * sin(phi), sqrt(1.0 - u.x))


@luisa.func
def influx_kernel(influx, point_light_count, direction_light_count, spp):
    acc = make_float3(0.0)
    idx = dispatch_id().x
    sampler = RandomSampler(make_int3(idx))
    vertex = vertex_buffer.read(idx).xyz
    normal = normal_buffer.read(idx).xyz
    onb = make_onb(normal)
    for i in range(point_light_count):
        point_light = point_light_buffer.read(i)
        direction = point_light.position - vertex
        probe_ray = make_ray(vertex, normalize(direction), 1e-3, length(direction))
        if not accel.trace_any(probe_ray):
            acc += point_light.emission * max(dot(normal, direction), 0.0)
    for i in range(direction_light_count):
        direction_light = direction_light_buffer.read(i)
        direction = normalize(-direction_light.direction)
        probe_ray = make_ray(vertex, direction, 1e-3, 1e10)
        if not accel.trace_any(probe_ray):
            acc += direction_light.emission * max(dot(normal, direction), 0.0)
    surface_acc = make_float3(0.0)
    for i in range(spp):
        direction = cosine_sample_hemisphere(make_float2(sampler.next(), sampler.next()))
        probe_direction = onb.to_world(direction)
        probe_ray = make_ray(vertex, probe_direction, 1e-3, 1e10)
        hit = accel.trace_closest(probe_ray)
        if hit.miss() or hit.inst == 0:
            continue
        else:
            surface_light = surface_light_buffer.read(hit.inst - 1)
            surface_acc += surface_light.emission * abs(dot(light_normal, probe_direction)) * light_area / float(spp)
    influx.write(idx, (acc + surface_acc * (40 / math.pi)) * amplifier * albedo)


def collect_vertex_influx(scene: Scene) -> luisa.Buffer:
    vertex_influx_buffer = luisa.Buffer.empty(vertex_buffer.size, float3)
    influx_kernel(vertex_influx_buffer, len(scene.point_lights), len(scene.direction_lights), 2500,
                  dispatch_size=vertex_influx_buffer.size)
    return vertex_influx_buffer


@luisa.func
def diffusion_weight(r):
    a = exp(-r / (3.0 * dmfp))
    return (a + a * a * a) / (8 * math.pi * dmfp * r)


def main():
    global ctx
    ctx = moderngl.create_standalone_context(430)
    res = (1000, 1000)
    camera = Camera(Vector3([0.0, 1.0, 8.0]), Vector3([0.0, 0.5, 0.0]), res, 20.0)
    luisa.init()
    filename = sys.argv[1]
    scene = parse_scene(filename)
    upload_scene(scene)
    shader = ctx.program(
        vertex_shader=
        """
            #version 330
            uniform mat4 mvp;
            in vec4 v_position;
            in vec4 v_efflux;
            out vec4 f_efflux;
            void main()
            {
                gl_Position = mvp * vec4(v_position.xyz, 1.0);
                f_efflux = v_efflux;
            }
        """,
        fragment_shader=
        """
            #version 330
            in vec4 f_efflux;
            out vec4 f_color;
            vec3 aces_tone_mapping(vec3 color)
            {
                vec3 mapped = color * (2.51 * color + 0.03) / (color * (2.43 * color + 0.59) + 0.14);
                return clamp(mapped, 0.0, 1.0);
            }
            void main()
            {
                f_color = vec4(aces_tone_mapping(f_efflux.rgb), 1.0);
            }
        """
    )
    compute_shader = ctx.compute_shader(
        """
            #version 430 core
            layout(local_size_x = 1) in;
            layout(std430, binding=0) buffer flux_in
            {
                vec4 influx[];
            };
            layout(std430, binding=1) buffer vertex_in
            {
                vec4 vertex[];
            };
            layout(std430, binding=2) buffer flux_out
            {
                vec4 efflux[];
            };
            void main()
            {
                int vertex_index = int(gl_GlobalInvocationID);
                vec3 vertex_position = vertex[vertex_index].xyz;
                vec3 acc = vec3(0.0);
                for (int i = 0; i < vertex_count; i++)
                {
                    float r = length(vertex_position - vertex[i].xyz);
                    vec3 a = exp(-r / (3.0 * dmfp));
                    a = (a + a * a * a) / (8.0 * 3.1415926 * dmfp * r);
                    if (i == vertex_index) a = vec3(0.0);
                    acc += influx[i].rgb * a / float(vertex_count);
                }
                efflux[vertex_index] = vec4(acc, 1.0);
            }
        """.replace("vertex_count", str(vertex_buffer.size)).replace("dmfp", f"vec3({dmfp.x}, {dmfp.y}, {dmfp.z})")
    )
    start = time.time()
    influx = collect_vertex_influx(scene)
    print(f"Collecting influx took: {(time.time() - start) * 1000} msec")
    start = time.time()
    influx_buffer_object = ctx.buffer(
        np.array([
            [flux.x, flux.y, flux.z, 0.0] for flux in influx.to_list()
        ]).astype(np.float32).tobytes()
    )
    print(f"Transferring data took: {(time.time() - start) * 1000} msec")
    start = time.time()
    efflux_buffer_object = ctx.buffer(reserve=influx_buffer_object.size)
    influx_buffer_object.bind_to_storage_buffer(0)
    scene.render_model.vertex_buffer_object.bind_to_storage_buffer(1)
    efflux_buffer_object.bind_to_storage_buffer(2)
    compute_shader.run(vertex_buffer.size)
    vao = ctx.vertex_array(
        shader,
        [
            (scene.render_model.vertex_buffer_object, '4f', 'v_position'),
            (efflux_buffer_object, '4f', 'v_efflux'),
        ],
        index_buffer=scene.render_model.index_buffer_object
    )
    shader['mvp'].write(camera.mvp)
    render_target_msaa = ctx.texture(res, 4, samples=4, dtype='f4')
    fbo_msaa = ctx.framebuffer(color_attachments=[render_target_msaa], depth_attachment=ctx.depth_renderbuffer(res, samples=4))
    with ctx.scope(fbo_msaa, moderngl.DEPTH_TEST):
        fbo_msaa.clear(alpha=1.0)
        vao.render()
    render_target = ctx.texture(res, 4, dtype='f4')
    fbo = ctx.framebuffer(color_attachments=[render_target])
    gl.glBindFramebuffer(gl.GL_READ_FRAMEBUFFER, fbo_msaa.glo)
    gl.glBindFramebuffer(gl.GL_DRAW_FRAMEBUFFER, fbo.glo)
    gl.glBlitFramebuffer(0, 0, res[0], res[1], 0, 0, res[0], res[1], gl.GL_COLOR_BUFFER_BIT, gl.GL_LINEAR)
    print(f"Rendering one frame took: {(time.time() - start) * 1000} msec")
    buffer = bytearray(res[0] * res[1] * 4 * 4)
    render_target.read_into(buffer)
    postfix = filename.split('/')[-1].split('\\')[-1].split('.')[0]
    plt.imsave(f"result-{postfix}.png", np.frombuffer(buffer, dtype=np.float32).reshape(res + (-1,))[::-1, ::-1])


if __name__ == "__main__":
    main()
