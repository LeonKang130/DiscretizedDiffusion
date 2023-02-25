from __future__ import annotations

from typing import NamedTuple, List

import numpy as np
import tinyobjloader

import luisa
from luisa.mathtypes import *

MODEL_FILE = "../resources/objects/bunny.obj"


class Mesh(NamedTuple):
    normals: np.ndarray
    vertices: np.ndarray
    triangles: np.ndarray

    @staticmethod
    def from_obj(filename) -> Mesh:
        reader = tinyobjloader.ObjReader()
        if not reader.ParseFromFile(filename):
            exit(-1)
        vertices = np.array(reader.GetAttrib().vertices, dtype=np.float32).reshape(-1, 3)
        normal_vectors = np.array(reader.GetAttrib().normals, dtype=np.float32).reshape(-1, 3)
        normals = np.zeros(vertices.shape, dtype=np.float32)
        for shape in reader.GetShapes():
            for index in shape.mesh.indices:
                normals[index.vertex_index] = normal_vectors[index.normal_index]
        triangles = np.array([index.vertex_index for shape in reader.GetShapes() for index in shape.mesh.indices])
        return Mesh(normals, vertices, triangles)


class DirLight(NamedTuple):
    direction: np.ndarray
    emission: np.ndarray


directional_lights: List[DirLight] = [DirLight(np.array([0.0, -1.0, 0.0]), np.array([1.0, 2.0, 1.0]))]


class PointLight(NamedTuple):
    position: np.ndarray
    emission: np.ndarray
    attenuation: np.ndarray


point_lights: List[PointLight] = [
    PointLight(np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0]))]

luisa.init()
LCDirLight = luisa.StructType(direction=float3, emission=float3)
directional_light_buffer = luisa.Buffer(max(len(directional_lights), 1), LCDirLight)
if len(directional_lights) > 0:
    directional_light_buffer.copy_from_list([
        LCDirLight(direction=make_float3(*light.direction), emission=make_float3(*light.emission)) for light in
        directional_lights
    ])
LCPointLight = luisa.StructType(position=float3, emission=float3, attenuation=float3)
point_light_buffer = luisa.Buffer(max(len(point_lights), 1), LCPointLight)
if len(point_lights) > 0:
    point_light_buffer.copy_from_list([
        LCPointLight(position=make_float3(*light.position), emission=make_float3(*light.emission),
                     attenuation=make_float3(*light.attenuation)) for light in point_lights
    ])
LCVertex = luisa.StructType(position=float3, normal=float3)
mesh = Mesh.from_obj(MODEL_FILE)
vertex_buffer = luisa.Buffer(len(mesh.vertices), LCVertex)
vertex_buffer.copy_from_list([
    LCVertex(position=make_float3(*vertex), normal=make_float3(*normal)) for vertex, normal in
    zip(mesh.vertices, mesh.normals)
])
vertex_influx = luisa.Buffer.zeros(len(mesh.vertices), float3)

NUM_DIRECTIONAL_LIGHTS = len(directional_lights)
NUM_POINT_LIGHTS = len(point_lights)
NUM_VERTICES = len(mesh.vertices)
@luisa.func
def calculate_vertex_influx():
    vertex_index = dispatch_id().x
    acc = make_float3(0.0)
    for i in range(0, NUM_DIRECTIONAL_LIGHTS):
        directional_light = directional_light_buffer.read(i)
        acc += directional_light.emission
    for i in range(0, NUM_POINT_LIGHTS):
        point_light = point_light_buffer.read(i)
        acc += point_light.emission
    vertex_influx.write(vertex_index, acc)


calculate_vertex_influx(dispatch_size=len(mesh.vertices))
vertex_influx_arr = np.empty((len(mesh.vertices) * 4), dtype=np.float32)
vertex_influx.copy_to(vertex_influx_arr)
print(vertex_influx_arr.reshape(-1, 4))

