struct Vertex
{
    float vx, vy, vz;
    uint8_t nx, ny, nz, nw;
    float16_t tu, tv;
};

struct Meshlet
{
    // vec3 keeps the Meshlet aligned to 16 bytes which is important because C++ has an alignas() directive
    vec3 centre;
    float radius;
    int8_t coneAxis[3];
    int8_t coneCutoff;

    uint dataOffset;
    uint8_t vertexCount;
    uint8_t triangleCount;
};

struct Globals 
{
    mat4 projection;
};

struct DrawCullData 
{
    float P00, P11, zNear, zFar; // Symmetric projection parameters
    float frustum[4]; // data for left/right/top/bottom frustum
    float lodBase, lodStep; // lod distance i = base * pow(step, i)
    float pyramidWidth, pyramidHeight; // depth pyramid size in texels

    uint drawCount;

    int cullingEnabled;
    int lodEnabled;
    int occlusionEnabled;

    bool lateWorkaroundAMD;
};

struct MeshLod 
{
    uint indexOffset;
    uint indexCount;
    uint meshletOffset;
    uint meshletCount;
};

struct Mesh 
{
    vec3 centre;
    float radius;

    uint vertexOffset;
    uint vertexCount;

    uint lodCount;
    MeshLod lods[8];
};

struct MeshDraw 
{
    vec3 position;
    float scale;
    vec4 orientation;

    uint meshIndex;
    uint vertexOffset; // == meshes[meshIndex].vertexOffset, helps data locality in mesh shader.
};

struct MeshDrawCommand 
{
    uint drawID;
    uint indexCount;
    uint instanceCount;
    uint firstIndex;
    uint vertexOffset;
    uint firstInstance;
    uint taskOffset;
    uint taskX;
    uint taskY;
    uint taskZ;
};

struct MeshTaskPayload 
{
    uint drawID;
    uint meshletIndices[32];
};

vec3 rotateQuat(vec3 v, vec4 q)
{
    return v + 2.0 * cross(q.xyz, cross(q.xyz, v) + q.w * v);
}
