#pragma once

struct Swapchain
{
    VkSwapchainKHR swapchain;

    std::vector<VkImage> images;

    uint32_t width;
    uint32_t height;
    uint32_t imageCount;
};

typedef struct GLFWwindow GLFWwindow;

VkSurfaceKHR createSurface(VkInstance instance, GLFWwindow* window);
VkFormat getSwapchainFormat(VkPhysicalDevice physicalDevice, VkSurfaceKHR surface);
void createSwapchain(Swapchain& result, VkPhysicalDevice physicalDevice, VkDevice device, VkSurfaceKHR surface, uint32_t familyIndex,
                     VkFormat format, VkSwapchainKHR oldSwapchain = 0);
void destroySwapchain(VkDevice device, const Swapchain& swapchain);

enum SwapchainStatus
{
    Swapchain_Ready,
    Swapchain_Resized,
    Swapchain_NotReady,
};

SwapchainStatus updateSwapchain(Swapchain& result, VkPhysicalDevice physicalDevice, VkDevice device, VkSurfaceKHR surface, uint32_t familyIndex, VkFormat format);