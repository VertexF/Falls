#version 450

#extension GL_EXT_shader_16bit_storage: require
#extension GL_EXT_shader_8bit_storage: require

#extension GL_GOOGLE_include_directive: require

#include "mesh.h"

#if 0
// NOTE: this should work, but unfortunately on AMD drivers it doesn't :(
// Because of this, we use the workaround where instead of a spec constant we use a uniform
layout(constant_id = 0) const bool LATE = false;
#else
#define LATE cullData.lateWorkaroundAMD
#endif

layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;

layout(push_constant) uniform block
{
    DrawCullData cullData;
};

layout(binding = 0) readonly buffer Draws
{
    MeshDraw draws[];
};

layout(binding = 1) readonly buffer Meshes
{
    Mesh meshes[];
};

layout(binding = 2) writeonly buffer DrawCommands
{
    MeshDrawCommand drawCommands[];
};

layout(binding = 3) buffer DrawCommandCount
{
    uint drawCommandCount;
};

layout(binding = 4) buffer DrawVisibility
{
    uint drawVisibility[];
};

layout(binding = 5) uniform sampler2D depthPyramid;

// 2D Polyhedral Bounds of a Clipped, Perspective-Projected 3D Sphere. Michael Mare Morgan McGuire. 2013
bool projectedSphere(vec3 C, float r, float zNear, float P00, float P11, out vec4 aabb)
{
    if (C.z < r + zNear) 
    {
        return false;
    }

    vec2 cx = -C.xz;
    vec2 vx = vec2(sqrt(dot(cx, cx) - r * r), r);
    vec2 minx = mat2(vx.x, vx.y, -vx.y, vx.x) * cx;
    vec2 maxx = mat2(vx.x, -vx.y, vx.y, vx.x) * cx;

    vec2 cy = -C.yz;
    vec2 vy = vec2(sqrt(dot(cy, cy) - r * r), r);
    vec2 miny = mat2(vy.x, vy.y, -vy.y, vy.x) * cy;
    vec2 maxy = mat2(vy.x, -vy.y, vy.y, vy.x) * cy;

    aabb = vec4(minx.x / minx.y * P00, miny.x / miny.y * P11, maxx.x / maxx.y * P00, maxy.x / maxy.y * P11);
    aabb = aabb.xwzy * vec4(0.5f, -0.5f, 0.5f, -0.5f) + vec4(0.5f); //Clip space -> uv space

    return true;
}

void main()
{
    uint di = gl_GlobalInvocationID.x;

    if(di >= cullData.drawCount)
    {
        return;
    }

    if(!LATE && drawVisibility[di] == 0)
    {
        return;
    }

    uint meshIndex = draws[di].meshIndex;
    Mesh mesh = meshes[meshIndex];

    vec3 centre = rotateQuat(mesh.centre, draws[di].orientation) * draws[di].scale + draws[di].position;
    float radius = mesh.radius * draws[di].scale;

    bool visible = true;
    // the left/top/right/bottom plane culling utilise frustum symmetry to cull against two planes at the same time.
    visible = visible && centre.z * cullData.frustum[1] - abs(centre.x) * cullData.frustum[0] > -radius;
    visible = visible && centre.z * cullData.frustum[3] - abs(centre.y) * cullData.frustum[2] > -radius;
    // the near/far plane culling uses camera space Z directly
    visible = visible && centre.z + radius > cullData.zNear && centre.z - radius < cullData.zFar;
    visible = visible || cullData.cullingEnabled == 0;

    if(LATE && visible && cullData.occlusionEnabled == 1)
    {
        vec4 aabb;
        if(projectedSphere(centre, radius, cullData.zNear, cullData.P00, cullData.P11, aabb))
        {
            float width = (aabb.z - aabb.x) * cullData.pyramidWidth;
            float height = (aabb.w - aabb.y) * cullData.pyramidHeight;

            float level = floor(log2(max(width, height)));

            // Sampler is set up to do min reduction, so this computes the minimum depth of a 2x2 texel quads
            float depth = textureLod(depthPyramid, (aabb.xy + aabb.zw) * 0.5, level).x;
            float depthSphere = cullData.zNear / (centre.z - radius);

            visible = visible && depthSphere > depth;
        }
    }

    if(visible && (!LATE || drawVisibility[di] == 0))
    {
        uint dci = atomicAdd(drawCommandCount, 1);

        // lod distance i = base * pow(step, i)
        // i = log2(distance / base) / log2(step)
        float lodIndexF = log2(length(centre) / cullData.lodBase) / log2(cullData.lodStep);
        uint lodIndex = min(uint(max(lodIndexF + 1, 0)), mesh.lodCount - 1);

        lodIndex = cullData.lodEnabled == 1 ? lodIndex : 0;

        // TODO: compiler doesn't seem to optimise this into a load directly from meshes array, so this is slow.
        MeshLod lod = meshes[meshIndex].lods[lodIndex];

        drawCommands[dci].drawID = di;
        drawCommands[dci].indexCount = lod.indexCount;
        drawCommands[dci].instanceCount = 1;
        drawCommands[dci].firstIndex = lod.indexOffset;
        drawCommands[dci].vertexOffset = mesh.vertexOffset;
        drawCommands[dci].firstInstance = 0;
        drawCommands[dci].taskOffset = lod.meshletOffset;
        drawCommands[dci].taskX = (lod.meshletCount + 31) / 32;
        drawCommands[dci].taskY = 1;
        drawCommands[dci].taskZ = 1;
    }

    if(LATE)
    {
        drawVisibility[di] = visible ? 1 : 0;
    }
}