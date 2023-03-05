import luisa
from luisa.mathtypes import *

Vertex = luisa.StructType(position=float3, normal=float3)
DirLight = luisa.StructType(direction=float3, emission=float3)
PointLight = luisa.StructType(position=float3, emission=float3)
directional_lights = luisa.Buffer.empty(1, DirLight)
point_lights = luisa.Buffer.empty(1, PointLight)


@luisa.func
def collect_influx(accel: luisa.Accel, vertex: luisa.Buffer, ) -> None:
    vertex_index = dispatch_id().x
