cmake_minimum_required(VERSION 3.14)

project(Falls)

file(GLOB_RECURSE GLSL_SOURCE_FILES "Falls/shaders/*.glsl")
file(GLOB_RECURSE GLSL_HEADER_FILES "Falls/shaders/*.h")
file(GLOB_RECURSE CPP_SOURCE_FILES "Falls/*.h" "Falls/*.cpp")

add_executable(Falls
               ${CPP_SOURCE_FILES}
               ${GLSL_SOURCE_FILES}
               ${GLSL_HEADER_FILES})
                
set_target_properties(Falls PROPERTIES CXX_STANDARD 11 CXX_STANDARD_REQUIRED YES CXX_EXTENSIONS NO)

target_compile_definitions(Falls PRIVATE GLFW_INCLUDE_NONE GLM_FORCE_XYZW_ONLY GLM_FORCE_QUAT_DATA_XYZW)
target_include_directories(Falls PRIVATE extern/fast_obj)

if(UNIX)
    target_compile_definitions(Falls PRIVATE VK_USE_PLATFORM_XLIB_KHR GLFW_EXPOSE_NATIVE_X11)
    set(VOLK_STATIC_DEFINES "VK_USE_PLATFORM_XLIB_KHR")
elseif(WIN32)
    target_compile_definitions(Falls PRIVATE VK_USE_PLATFORM_WIN32_KHR GLFW_EXPOSE_NATIVE_WIN32)
    target_compile_definitions(Falls PRIVATE WIN32_LEAN_AND_MEAN NOMINMAX)
    set(VOLK_STATIC_DEFINES "VK_USE_PLATFORM_WIN32_KHR")
endif()

add_subdirectory(extern/glfw)
add_subdirectory(extern/glm)
add_subdirectory(extern/volk)
add_subdirectory(extern/meshoptimizer)

target_link_libraries(Falls PRIVATE glfw glm::glm volk meshoptimizer)

if(UNIX)
    set(GLSL_VALIDATOR "glslangValidator")
elseif(${CMAKE_HOST_SYSTEM_PROCESSOR} STREQUAL "AMD64")
    set(GLSL_VALIDATOR "$ENV{VULKAN_SDK}/Bin/glslangValidator.exe")
else()
    set(GLSL_VALIDATOR "$ENV{VULKAN_SDK}/Bin32/glslangValidator.exe")
endif()

# Thanks to evilactuall github guy
set(SPIV_OUTPUT_DIR "${PROJECT_BINARY_DIR}/Falls/shaders")
foreach(GLSL ${GLSL_SOURCE_FILES})
    get_filename_component(STEM ${GLSL} NAME_WLE)
    set(SPIRV "${SPIRV_OUTPUT_DIR}${STEM}.spv")
    add_compile_options(
        OUTPUT ${SPIRV}
        COMMAND ${CMAKE_COMMAND} -E make_directory "${PROJECT_BINARY_DIR}/Falls/shaders"
        COMMAND ${GLSL_VALIDATOR} -V --target-env vulkan1.3 --quiet ${GLSL} -o ${SPIRV}
        DEPENDS ${GLSL} ${GLSL_HEADER_FILES}
    )
    list(APPEND SPIRV_BINRARY_FILES ${SPIRV})
endforeach()

add_custom_target(compile_shaders DEPENDS ${SPIRV_BINRARY_FILES})
if(WIN32)
    set_target_properties(Falls PROPERTIES VS_DEBUGGER_COMMAND_ARGUMENTS "${CMAKE_CURRENT_SOURCE_DIR}/data/kitten.obj")
endif()
add_dependencies(Falls compile_shaders)