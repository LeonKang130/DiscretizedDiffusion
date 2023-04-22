from __future__ import annotations
import time
from typing import NamedTuple
from OpenGL import GL as gl
from matplotlib import pyplot as plt
from pyrr import Vector3
from camera import Camera
import luisa
from luisa.mathtypes import *
from luisa import RandomSampler
import moderngl
import tinyobjloader
import numpy as np
import json
from enum import Enum
import math
import sys


DeviceDirectionLight = luisa.StructType(direction=float3, emission=float3)
DevicePointLight = luisa.StructType(position=float3, emission=float3)
DeviceSurfaceLight = luisa.StructType(emission=float3)
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
def uniform_sample_hemisphere(u):
    r = sqrt(1.0 - u.x * u.x)
    phi = 2.0 * 3.1415926 * u.y
    return make_float3(r * cos(phi), r * sin(phi), u.x)


@luisa.func
def uniform_sample_sphere(u):
    cos_theta = 1.0 - 2.0 * u.x
    sin_theta = sqrt(1.0 - cos_theta * cos_theta)
    phi = 2.0 * 3.1415926 * u.y
    return make_float3(sin_theta * cos(phi), sin_theta * sin(phi), cos_theta)


luisa.init()
model_vertex_count: int = 0
vertex_buffer: luisa.Buffer = None
normal_buffer: luisa.Buffer = None
surface_light_buffer: luisa.Buffer = None
point_light_buffer: luisa.Buffer = None
direction_light_buffer: luisa.Buffer = None
surface_light_count: int = 0
point_light_count: int = 0
direction_light_count: int = 0
ctx = moderngl.create_standalone_context(430)
heap = luisa.BindlessArray()
accel: luisa.Accel = luisa.Accel()
vbo: moderngl.Buffer = None
nbo: moderngl.Buffer = None
ibo: moderngl.Buffer = None
surface_area: float = 0
sigma_a: float3 = make_float3(1.0)
sigma_s: float3 = make_float3(1.0)
eta: float = 1.0
g: float = 0.0


class Equation(Enum):
    Normalized = "normalized"
    Dipole = "dipole"
    VPT = "vpt"
    Undefined = "undefined"


equation: Equation = Equation.Undefined


def parse_scene(filename: str):
    with open(filename, 'r') as file:
        scene_data = json.load(file)
        vertex_arrays = []
        normal_arrays = []
        triangle_arrays = []
        reader = tinyobjloader.ObjReader()
        # parse the model to be rendered, add model data to array lists
        offset = 0
        reader.ParseFromFile(scene_data["render_model"])
        attrib = reader.GetAttrib()
        global model_vertex_count
        vertex_array = np.array(attrib.vertices, dtype=np.float32).reshape(-1, 3)
        model_vertex_count = vertex_array.shape[0]
        normal_vectors = np.array(attrib.normals, dtype=np.float32).reshape(-1, 3)
        normal_vectors /= np.linalg.norm(normal_vectors, axis=1).reshape((-1, 1))
        normal_array = np.empty_like(vertex_array, dtype=np.float32)
        for shape in reader.GetShapes():
            for index in shape.mesh.indices:
                normal_array[index.vertex_index] = normal_vectors[index.normal_index]
        vertex_arrays.append(vertex_array)
        normal_arrays.append(normal_array)
        triangle_array = np.array([
            index.vertex_index for shape in reader.GetShapes() for index in shape.mesh.indices
        ])
        # calculate total surface area of the model
        global surface_area
        surface_areas = np.zeros((vertex_array.shape[0], 1), dtype=np.float32)
        for i in range(len(triangle_array) // 3):
            i0, i1, i2 = triangle_array[i:i+3]
            p0, p1, p2 = map(lambda x: vertex_array[x], (i0, i1, i2))
            local_surface_area = np.linalg.norm(np.cross(p1 - p0, p2 - p0)) * 0.5
            if local_surface_area == 0.0:
                continue
            local_normal = np.cross((p1 - p0) / np.linalg.norm(p1 - p0), (p2 - p0) / np.linalg.norm(p2 - p0))
            local_normal /= np.linalg.norm(local_normal)
            projected_area0, projected_area1, projected_area2 = map(
                lambda x: local_surface_area / 3 * np.abs(np.dot(normal_array[x], local_normal)),
                (i0, i1, i2)
            )
            surface_areas[i0] += projected_area0
            surface_areas[i1] += projected_area1
            surface_areas[i2] += projected_area2
            surface_area += projected_area0 + projected_area1 + projected_area2
        triangle_arrays.append(triangle_array)
        # upload vertex array and normal array to GL
        global vbo, ibo, nbo
        vbo = ctx.buffer(np.hstack((vertex_array, surface_areas)))
        ibo = ctx.buffer(triangle_array.astype(np.int32))
        nbo = ctx.buffer(np.hstack((normal_array, np.zeros((normal_array.shape[0], 1), dtype=np.float32))))
        # parse and upload light information
        # add surface light data to array lists
        offset += vertex_arrays[-1].shape[0]
        global surface_light_buffer, direction_light_buffer, point_light_buffer
        global surface_light_count, direction_light_count, point_light_count
        surface_lights = []
        for light in scene_data["surface_lights"]:
            print("Surface light with emission: ", light["emission"])
            surface_lights.append(DeviceSurfaceLight(emission=make_float3(*light["emission"])))
            reader.ParseFromFile(light["model"])
            attrib = reader.GetAttrib()
            vertex_array = np.array(attrib.vertices, dtype=np.float32).reshape(-1, 3)
            normal_vectors = np.array(attrib.normals, dtype=np.float32).reshape(-1, 3)
            normal_vectors /= np.linalg.norm(normal_vectors, axis=1).reshape(-1, 1)
            normal_array = np.empty_like(vertex_array, dtype=np.float32)
            for shape in reader.GetShapes():
                for index in shape.mesh.indices:
                    normal_array[index.vertex_index] = normal_vectors[index.normal_index]
            vertex_arrays.append(vertex_array)
            normal_arrays.append(normal_array)
            triangle_array = np.array([
                index.vertex_index for shape in reader.GetShapes() for index in shape.mesh.indices
            ]) + offset
            triangle_arrays.append(triangle_array)
            offset += vertex_arrays[-1].shape[0]
        surface_light_buffer = luisa.Buffer.empty(max(len(surface_lights), 1), dtype=DeviceSurfaceLight)
        if surface_lights:
            surface_light_buffer.copy_from_list(surface_lights)
            surface_light_count = len(surface_lights)
        direction_lights =\
            [DeviceDirectionLight(
                direction=make_float3(*light["direction"]),
                emission=make_float3(*light("emission"))
            ) for light in scene_data["direction_lights"]]
        direction_light_buffer = luisa.Buffer.empty(max(len(direction_lights), 1), dtype=DeviceDirectionLight)
        if direction_lights:
            direction_light_buffer.copy_from_list(direction_lights)
            direction_light_count = len(direction_lights)
        point_lights = \
            [DevicePointLight(
                position=make_float3(*light["position"]),
                emission=make_float3(*light("emission"))
            ) for light in scene_data["point_lights"]]
        point_light_buffer = luisa.Buffer.empty(max(len(point_lights), 1), dtype=DevicePointLight)
        if point_lights:
            point_light_buffer.copy_from_list(point_lights)
            point_light_count = len(point_lights)
        # combine and upload vertex && normal array lists
        global vertex_buffer, normal_buffer
        vertices = np.concatenate(vertex_arrays)
        normals = np.concatenate(normal_arrays)
        vertex_buffer = luisa.Buffer.empty(len(vertices), float3)
        normal_buffer = luisa.Buffer.empty(len(normals), float3)
        vertex_buffer.copy_from_array(np.hstack((vertices, np.zeros((vertices.shape[0], 1), dtype=np.float32))))
        normal_buffer.copy_from_array(np.hstack((normals, np.zeros((normals.shape[0], 1), dtype=np.float32))))
        # upload array lists to heap
        global heap, accel
        mesh_cnt = len(triangle_arrays)
        for i in range(mesh_cnt):
            triangle_array = triangle_arrays[i].astype(np.int32)
            triangle_buffer = luisa.Buffer(len(triangle_array), dtype=int)
            triangle_buffer.copy_from_array(triangle_array)
            heap.emplace(i, triangle_buffer)
            mesh = luisa.Mesh(vertex_buffer, triangle_buffer)
            accel.add(mesh)
        accel.update()
        heap.update()
        # parse render equation
        global equation
        if str.lower(scene_data["equation"]) == "dipole":
            equation = Equation.Dipole
        elif str.lower(scene_data["equation"]) == "normalized":
            equation = Equation.Normalized
        elif str.lower(scene_data["equation"]) == "vpt":
            equation = Equation.VPT
        else:
            equation = Equation.Undefined
        # parse parameters
        global sigma_a, sigma_s, eta, g
        sigma_a = make_float3(*scene_data["sigma_a"])
        sigma_s = make_float3(*scene_data["sigma_s"])
        eta = scene_data["eta"]
        g = scene_data["g"]


class Parameters(NamedTuple):
    sigma_tr: float3
    dmfp: float3
    albedo: float3
    zr: float3
    zv: float3
    transmittance: float


def calculate_parameters():
    print(sigma_s, sigma_a, g, eta)
    sigma_s_prime = sigma_s * (1.0 - g)
    sigma_t_prime = sigma_s_prime + sigma_a
    alpha_prime = sigma_s_prime / sigma_t_prime
    fresnel = -1.440 / eta / eta + 0.710 / eta + 0.668 + 0.0636 * eta
    a = (1.0 + fresnel) / (1.0 - fresnel)
    albedo = 0.5 * alpha_prime * (1.0 + make_float3(
        math.exp(-4.0 / 3.0 * a * math.sqrt(3.0 * (1.0 - alpha_prime.x))),
        math.exp(-4.0 / 3.0 * a * math.sqrt(3.0 * (1.0 - alpha_prime.y))),
        math.exp(-4.0 / 3.0 * a * math.sqrt(3.0 * (1.0 - alpha_prime.z)))
    )) / (1.0 + make_float3(
        math.sqrt(3.0 * (1.0 - alpha_prime.x)),
        math.sqrt(3.0 * (1.0 - alpha_prime.y)),
        math.sqrt(3.0 * (1.0 - alpha_prime.z)))
    )
    sigma_tr = make_float3(
        math.sqrt(3.0 * (1.0 - alpha_prime.x)),
        math.sqrt(3.0 * (1.0 - alpha_prime.y)),
        math.sqrt(3.0 * (1.0 - alpha_prime.z))
    ) * sigma_t_prime
    s = albedo - 0.8
    s *= s
    s = 1.9 - albedo + 3.5 * s
    dmfp = 1.0 / (s * sigma_t_prime)
    zr = 1.0 / sigma_t_prime
    zv = (1.0 + 4.0 / 3.0 * a) / sigma_t_prime
    reflectance = (1.0 - eta) / (1.0 + eta)
    return Parameters(sigma_tr, dmfp, albedo, zr, zv, 1.0 - reflectance * reflectance)


@luisa.func
def vertex_influx_kernel(influx, spp):
    acc = make_float3(0.0)
    idx = dispatch_id().x
    sampler = RandomSampler(make_int3(idx))
    vertex = vertex_buffer.read(idx).xyz
    normal = normal_buffer.read(idx).xyz
    onb = make_onb(normal)
    for i in range(point_light_count):
        point_light = point_light_buffer.read(i)
        direction = point_light.position - vertex
        probe_ray = make_ray(vertex, normalize(direction), 1e-2, length(direction))
        if not accel.trace_any(probe_ray):
            acc += point_light.emission * max(dot(normal, normalize(direction)), 0.0)
    for i in range(direction_light_count):
        direction_light = direction_light_buffer.read(i)
        direction = normalize(-direction_light.direction)
        probe_ray = make_ray(vertex, direction, 1e-2, 1e10)
        if not accel.trace_any(probe_ray):
            acc += direction_light.emission * max(dot(normal, direction), 0.0)
    surface_acc = make_float3(0.0)
    for i in range(spp):
        direction = cosine_sample_hemisphere(make_float2(sampler.next(), sampler.next()))
        probe_direction = onb.to_world(direction)
        probe_ray = make_ray(vertex, probe_direction, 1e-2, 1e10)
        hit = accel.trace_closest(probe_ray)
        if hit.miss() or hit.inst == 0:
            continue
        else:
            surface_light = surface_light_buffer.read(hit.inst - 1)
            i0 = heap.buffer_read(int, hit.inst, hit.prim * 3)
            i1 = heap.buffer_read(int, hit.inst, hit.prim * 3 + 1)
            i2 = heap.buffer_read(int, hit.inst, hit.prim * 3 + 2)
            n0 = normal_buffer.read(i0)
            n1 = normal_buffer.read(i1)
            n2 = normal_buffer.read(i2)
            n = normalize(hit.interpolate(n0, n1, n2))
            surface_acc += surface_light.emission * abs(dot(n, direction))
    acc += surface_acc / float(spp)
    influx.write(idx, acc)


def collect_vertex_influx() -> luisa.Buffer:
    vertex_influx_buffer = luisa.Buffer.empty(model_vertex_count, float3)
    vertex_influx_kernel(vertex_influx_buffer, 2000, dispatch_size=model_vertex_count)
    return vertex_influx_buffer


epsilon = sys.float_info.epsilon
@luisa.func
def vertex_total_scattering_kernel(efflux, spp):
    acc = make_float3(0.0)
    beta = make_float3(1.0)
    idx = heap.buffer_read(int, 0, dispatch_id().x)
    sampler = RandomSampler(make_int3(idx))
    curr_pos = vertex_buffer.read(idx).xyz
    curr_dir = -normal_buffer.read(idx).xyz
    for i in range(spp):
        for depth in range(20):
            nxt_event = -log(max(epsilon, sampler.next())) / (sigma_a + sigma_s).x
            ray = make_ray(curr_pos, curr_dir, 1e-4, nxt_event)
            hit = accel.trace_closest(ray)
            if hit.miss():
                curr_pos += curr_dir * nxt_event
                curr_dir = uniform_sample_sphere(make_float2(sampler.next(), sampler.next()))
                beta *= sigma_s / (sigma_a + sigma_s) * 4.0 * math.pi
            elif hit.inst == 0:
                i0 = heap.buffer_read(int, 0, hit.prim * 3)
                i1 = heap.buffer_read(int, 0, hit.prim * 3 + 1)
                i2 = heap.buffer_read(int, 0, hit.prim * 3 + 2)
                p0 = vertex_buffer.read(i0)
                p1 = vertex_buffer.read(i1)
                p2 = vertex_buffer.read(i2)
                n0 = normal_buffer.read(i0)
                n1 = normal_buffer.read(i1)
                n2 = normal_buffer.read(i2)
                curr_pos = hit.interpolate(p0, p1, p2)
                n = normalize(hit.interpolate(n0, n1, n2))
                onb = make_onb(n)
                for idx in range(point_light_count):
                    point_light = point_light_buffer.read(idx)
                    direction = point_light.position - curr_pos
                    probe_ray = make_ray(curr_pos, normalize(direction), 1e-3, length(direction))
                    if not accel.trace_any(probe_ray):
                        acc += point_light.emission * max(dot(n, normalize(direction)), 0.0)
                for idx in range(direction_light_count):
                    direction_light = direction_light_buffer.read(idx)
                    direction = normalize(-direction_light.direction)
                    probe_ray = make_ray(curr_pos, direction, 1e-3, 1e10)
                    if not accel.trace_any(probe_ray):
                        acc += direction_light.emission * max(dot(n, direction), 0.0)
                for collection in range(16):
                    curr_dir = onb.to_world(cosine_sample_hemisphere(make_float2(sampler.next(), sampler.next())))
                    ray = make_ray(curr_pos, curr_dir, 1e-4, 1e10)
                    hit = accel.trace_closest(ray)
                    if not hit.miss() and hit.inst != 0:
                        i0 = heap.buffer_read(int, hit.inst, hit.prim * 3)
                        i1 = heap.buffer_read(int, hit.inst, hit.prim * 3 + 1)
                        i2 = heap.buffer_read(int, hit.inst, hit.prim * 3 + 2)
                        n0 = normal_buffer.read(i0)
                        n1 = normal_buffer.read(i1)
                        n2 = normal_buffer.read(i2)
                        n = normalize(hit.interpolate(n0, n1, n2))
                        light = surface_light_buffer.read(hit.inst - 1)
                        acc += (beta * math.pi / 16) * light.emission * abs(dot(curr_dir, n))
                break
            else:
                break
    efflux.write(idx, acc)


def main():
    res = (1000, 1000)
    camera = Camera(Vector3([0.0, 1.0, 8.0]), Vector3([0.0, 0.5, 0.0]), res, 20.0)
    parse_scene(sys.argv[1])
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
    start = time.time()
    influx = collect_vertex_influx()
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
    parameters = calculate_parameters()
    if equation == Equation.Normalized:
        print("Rendering using Normalized BSSRDF")
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
                        if (r <= 0.025) {
                            vec3 a = 1.0 / (dmfp * 0.1 * 3.1415926);
                            acc += influx[i].rgb * a;
                        }
                        else {
                            vec3 a = exp(-r / (3.0 * dmfp));
                            a = (a + a * a * a) / (8.0 * 3.1415926 * dmfp * r);
                            acc += influx[i].rgb * a;
                        }
                    }
                    acc *= albedo * surface_area / float(vertex_count) * transmittance
                    efflux[vertex_index] = vec4(acc, 1.0);
                }
            """
            .replace("vertex_count", f"{model_vertex_count}")
            .replace("dmfp", f"vec3({parameters.dmfp.x}, {parameters.dmfp.y}, {parameters.dmfp.z})")
            .replace("albedo", f"vec3({parameters.albedo.x}, {parameters.albedo.y}, {parameters.albedo.z})")
            .replace("surface_area", f"{surface_area}")
            .replace("transmittance", f"{parameters.transmittance}")
        )
    elif equation == Equation.Dipole:
        print("Rendering using Dipole")
        albedo = sigma_s / (sigma_a + sigma_s)
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
                    vec3 acc = influx[vertex_index].xyz * albedo / 8.0;
                    for (int i = 0; i < vertex_count; i++)
                    {
                        float r = length(vertex_position - vertex[i].xyz);
                        vec3 dr = sqrt(r * r + zr * zr);
                        vec3 dv = sqrt(r * r + zv * zv);
                        vec3 weight =
                            1.0 / (4.0 * 3.1415926) * (
                                zr * (sigma_tr * dr + 1.0) * exp(-sigma_tr * dr) / (dr * dr * dr) +
                                zv * (sigma_tr * dv + 1.0) * exp(-sigma_tr * dv) / (dv * dv * dv)
                            );
                        weight = any(isnan(weight)) ? vec3(0.0) : weight;
                        acc += influx[i].rgb * weight * surface_area / float(vertex_count);
                    }
                    acc *= transmittance;
                    efflux[vertex_index] = vec4(acc, 1.0);
                }
            """
            .replace("vertex_count", f"{model_vertex_count}")
            .replace("sigma_tr", f"vec3({parameters.sigma_tr.x}, {parameters.sigma_tr.y}, {parameters.sigma_tr.z})")
            .replace("zr", f"vec3({parameters.zr.x}, {parameters.zr.y}, {parameters.zr.z})")
            .replace("zv", f"vec3({parameters.zv.x}, {parameters.zv.y}, {parameters.zv.z})")
            .replace("surface_area", f"{surface_area}")
            .replace("albedo", f"vec3({albedo.x}, {albedo.y}, {albedo.z})")
            .replace("transmittance", f"{parameters.transmittance}")
        )
    else:
        print("Undefined diffusion equation")
        exit(0)
    influx_buffer_object.bind_to_storage_buffer(0)
    vbo.bind_to_storage_buffer(1)
    efflux_buffer_object.bind_to_storage_buffer(2)
    compute_shader.run(model_vertex_count)
    vao = ctx.vertex_array(
        shader,
        [
            (vbo, '4f', 'v_position'),
            (efflux_buffer_object, '4f', 'v_efflux'),
        ],
        index_buffer=ibo
    )
    shader['mvp'].write(camera.mvp)
    render_target_msaa = ctx.texture(res, 4, samples=4, dtype='f4')
    fbo_msaa = ctx.framebuffer(color_attachments=[render_target_msaa],
                               depth_attachment=ctx.depth_renderbuffer(res, samples=4))
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
    postfix = sys.argv[1].split('/')[-1].split('\\')[-1].split('.')[0] + '-' + str(equation).lower().split('.')[-1]
    plt.imsave(f"result-{postfix}.png", np.frombuffer(buffer, dtype=np.float32).reshape(res + (-1,))[::-1, ::-1])


if __name__ == "__main__":
    main()
