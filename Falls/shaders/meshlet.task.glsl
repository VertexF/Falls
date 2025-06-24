#version 450

#extension GL_EXT_shader_16bit_storage: require
#extension GL_EXT_shader_8bit_storage: require
#extension GL_EXT_mesh_shader: require

#extension GL_GOOGLE_include_directive: require

#extension GL_KHR_shader_subgroup_ballot: require

#extension GL_ARB_shader_draw_parameters: require

#include "mesh.h"

#define CULL 1 //TODO

layout(local_size_x = 32, local_size_y = 1, local_size_z = 1) in;

layout(binding = 0) readonly buffer DrawCommands
{
    MeshDrawCommand drawCommands[];
};

layout(binding = 1) readonly buffer Draws
{
    MeshDraw draws[];
};

layout(binding = 2) readonly buffer Meshlets
{
    Meshlet meshlets[];
};

taskPayloadSharedEXT MeshTaskPayload payload;

bool coneCull(vec3 centre, float radius, vec3 coneAxis, float coneCutoff, vec3 cameraPosition)
{
    return dot(centre - cameraPosition, coneAxis) >= coneCutoff * length(centre - cameraPosition) + radius;
}

#if CULL
shared int sharedCount;
#endif

//NOTE: The point of using subgroups is to allow atomic operations to happen in subgroups of the totally work group.
//Meaning that you can say have a subgroup of 16, it then will do an add operation twice instead of 32 times over a workgroup of 32.
void main()
{
    //ti = task shader index.
    //mgi = mesh shader group index.
    //mi = mesh shader index.
    uint ti = gl_LocalInvocationID.x;
    uint mgi = gl_WorkGroupID.x;

    uint drawID = drawCommands[gl_DrawIDARB].drawID;
    MeshDraw meshDraw = draws[drawID];

    //The 32 + ti is the task shader offset into the mesh shader index. 
    //If you multiply the mesh shader group you get the mesh shader index in a particular group.
    uint mi = mgi * 32 + ti + drawCommands[gl_DrawIDARB].taskOffset;

#if CULL
    sharedCount = 0;
    memoryBarrierShared();

    vec3 centre = rotateQuat(meshlets[mi].centre, meshDraw.orientation) * meshDraw.scale + meshDraw.position;
    float radius = meshlets[mi].radius * meshDraw.scale;
    vec3 coneAxis = rotateQuat(vec3(int(meshlets[mi].coneAxis[0]) / 127.f, int(meshlets[mi].coneAxis[1]) / 127.f, int(meshlets[mi].coneAxis[2]) / 127.f), meshDraw.orientation);
    float coneCutoff = int(meshlets[mi].coneCutoff) / 127.f;

    bool accept = !coneCull(centre, radius, coneAxis, coneCutoff, vec3(0, 0, 0));

    //If we haven't culled the meshlet 
    if(accept)
    {
        uint index = atomicAdd(sharedCount, 1);
        payload.meshletIndices[index] = mi;
    }

    payload.drawID = drawID;

    memoryBarrierShared();

    EmitMeshTasksEXT(sharedCount, 1, 1);
#else
    payload.drawID = drawID;
    payload.meshletIndices[ti] = mi;

    EmitMeshTasksEXT(32, 1, 1);
#endif
}