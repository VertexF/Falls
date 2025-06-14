#pragma once

#include <assert.h>

#define GLFW_EXPOSE_NATIVE_WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_MEAN_AND_LEAN
#endif //WIN32_LEAN_AND_MEAN

#include <vector>

#include <volk.h>


#define VK_CHECK(call)\
        do\
        {\
            VkResult result = call;\
            assert(result == VK_SUCCESS);\
        }\
        while(0)\

#ifndef ARRAYSIZE
#define ARRAYSIZE(array) (sizeof(array) / sizeof((array)[0]))
#endif // !ARRAYSIZE
