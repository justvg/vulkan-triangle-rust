use winit::{
    event::{ElementState, Event, KeyboardInput, VirtualKeyCode, WindowEvent},
    event_loop::{ControlFlow, EventLoop},
    window::WindowBuilder,
};

use ash::extensions::khr;
use ash::vk;

use std::ffi::CStr;

pub struct Swapchain {
    swapchain: vk::SwapchainKHR,
    images: Vec<vk::Image>,
    image_views: Vec<vk::ImageView>,

    width: u32,
    height: u32,
}

fn create_instance(entry: &ash::Entry) -> ash::Instance {
    unsafe {
        let app_name = CStr::from_bytes_with_nul_unchecked(b"VulkanApp\0");

        let app_info = vk::ApplicationInfo {
            p_application_name: app_name.as_ptr(),
            application_version: 0,
            p_engine_name: app_name.as_ptr(),
            engine_version: 0,
            api_version: vk::make_api_version(0, 1, 0, 0),
            ..Default::default()
        };

        #[cfg(debug_assertions)] 
        let validation_layers =
            [CStr::from_bytes_with_nul_unchecked(b"VK_LAYER_KHRONOS_validation\0").as_ptr()];

        #[cfg(not(debug_assertions))]
        let validation_layers = [];

        let extensions = [
            khr::Surface::name().as_ptr(),
            khr::Win32Surface::name().as_ptr(),
        ];

        let create_info = vk::InstanceCreateInfo {
            p_application_info: &app_info,
            enabled_layer_count: validation_layers.len() as u32,
            pp_enabled_layer_names: validation_layers.as_ptr(),
            enabled_extension_count: extensions.len() as u32,
            pp_enabled_extension_names: extensions.as_ptr(),
            ..Default::default()
        };

        let instance = entry
            .create_instance(&create_info, None)
            .expect("Can't create Vulkan instance.");

        instance
    }
}

fn create_surface(
    entry: &ash::Entry,
    instance: &ash::Instance,
    window: &winit::window::Window,
) -> vk::SurfaceKHR {
    unsafe {
        ash_window::create_surface(&entry, &instance, &window, None).expect("Can't create surface.")
    }
}

fn get_surface_composite_alpha(
    surface_caps: &vk::SurfaceCapabilitiesKHR,
) -> vk::CompositeAlphaFlagsKHR {
    if surface_caps
        .supported_composite_alpha
        .contains(vk::CompositeAlphaFlagsKHR::OPAQUE)
    {
        return vk::CompositeAlphaFlagsKHR::OPAQUE;
    } else if surface_caps
        .supported_composite_alpha
        .contains(vk::CompositeAlphaFlagsKHR::PRE_MULTIPLIED)
    {
        return vk::CompositeAlphaFlagsKHR::PRE_MULTIPLIED;
    } else if surface_caps
        .supported_composite_alpha
        .contains(vk::CompositeAlphaFlagsKHR::POST_MULTIPLIED)
    {
        return vk::CompositeAlphaFlagsKHR::POST_MULTIPLIED;
    } else {
        return vk::CompositeAlphaFlagsKHR::INHERIT;
    }
}

fn get_surface_capabilities(
    physical_device: vk::PhysicalDevice,
    surface: vk::SurfaceKHR,
    surface_loader: &khr::Surface,
) -> vk::SurfaceCapabilitiesKHR {
    unsafe {
        surface_loader
            .get_physical_device_surface_capabilities(physical_device, surface)
            .expect("Can't get Vulkan surface capabilities.")
    }
}

fn get_graphics_family_index(instance: &ash::Instance, physical_device: vk::PhysicalDevice) -> u32 {
    unsafe {
        let queue_family_properties =
            instance.get_physical_device_queue_family_properties(physical_device);

        for i in 0..queue_family_properties.len() {
            if queue_family_properties[i]
                .queue_flags
                .contains(vk::QueueFlags::GRAPHICS)
            {
                return i as u32;
            }
        }

        vk::QUEUE_FAMILY_IGNORED
    }
}

fn pick_physical_device(
    instance: &ash::Instance,
    surface: vk::SurfaceKHR,
    surface_loader: &khr::Surface,
) -> vk::PhysicalDevice {
    unsafe {
        let physical_devices = instance
            .enumerate_physical_devices()
            .expect("Can't enumerate Vulkan physical devices.");

        let mut discrete = vk::PhysicalDevice::null();
        let mut fallback = vk::PhysicalDevice::null();
        for physical_device in physical_devices {
            let properties = instance.get_physical_device_properties(physical_device);

            let graphics_family_index = get_graphics_family_index(instance, physical_device);
            if graphics_family_index == vk::QUEUE_FAMILY_IGNORED {
                continue;
            }

            let supports_presentation = surface_loader
                .get_physical_device_surface_support(
                    physical_device,
                    graphics_family_index,
                    surface,
                )
                .expect("Can't get physical device surface support info.");
            if !supports_presentation {
                continue;
            }

            if discrete == vk::PhysicalDevice::null()
                && properties.device_type == vk::PhysicalDeviceType::DISCRETE_GPU
            {
                discrete = physical_device;
            } else if fallback == vk::PhysicalDevice::null() {
                fallback = physical_device;
            }
        }

        let physical_device = if discrete != vk::PhysicalDevice::null() {
            discrete
        } else {
            fallback
        };
        if physical_device != vk::PhysicalDevice::null() {
            let properties = instance.get_physical_device_properties(physical_device);
            println!(
                "Selected GPU: {}",
                CStr::from_ptr(properties.device_name.as_ptr())
                    .to_str()
                    .unwrap()
            );
        } else {
            panic!("Can't pick GPU.")
        }

        physical_device
    }
}

fn create_device(
    instance: &ash::Instance,
    physical_device: vk::PhysicalDevice,
    family_index: u32,
) -> ash::Device {
    unsafe {
        let queue_priorities = [1.0];

        let queue_create_infos = [vk::DeviceQueueCreateInfo {
            queue_family_index: family_index,
            queue_count: 1,
            p_queue_priorities: queue_priorities.as_ptr(),
            ..Default::default()
        }];

        let extensions = [khr::Swapchain::name().as_ptr()];

        let create_info = vk::DeviceCreateInfo {
            queue_create_info_count: queue_create_infos.len() as u32,
            p_queue_create_infos: queue_create_infos.as_ptr(),
            enabled_extension_count: extensions.len() as u32,
            pp_enabled_extension_names: extensions.as_ptr(),
            ..Default::default()
        };

        let device = instance
            .create_device(physical_device, &create_info, None)
            .expect("Can't create Vulkan device.");

        device
    }
}

fn get_device_queue(device: &ash::Device, family_index: u32) -> vk::Queue {
    unsafe { device.get_device_queue(family_index, 0) }
}

fn get_swapchain_format(
    physical_device: vk::PhysicalDevice,
    surface: vk::SurfaceKHR,
    surface_loader: &khr::Surface,
) -> vk::Format {
    unsafe {
        let surface_formats = surface_loader
            .get_physical_device_surface_formats(physical_device, surface)
            .expect("Can't get physical device surface formats.");

        if surface_formats.len() == 1 && surface_formats[0].format == vk::Format::UNDEFINED {
            return vk::Format::R8G8B8A8_UNORM;
        } else {
            for format in &surface_formats {
                if format.format == vk::Format::R8G8B8A8_UNORM
                    || format.format == vk::Format::B8G8R8A8_UNORM
                {
                    return format.format;
                }
            }
        }

        surface_formats[0].format
    }
}

fn create_swapchain_khr(
    swapchain_loader: &khr::Swapchain,
    surface: vk::SurfaceKHR,
    surface_caps: &vk::SurfaceCapabilitiesKHR,
    format: vk::Format,
    width: u32,
    height: u32,
    family_index: u32,
    old_swapchain: vk::SwapchainKHR,
) -> vk::SwapchainKHR {
    unsafe {
        let composite_alpha = get_surface_composite_alpha(&surface_caps);

        let create_info = vk::SwapchainCreateInfoKHR {
            surface: surface,
            min_image_count: std::cmp::max(2, surface_caps.min_image_count),
            image_format: format,
            image_color_space: vk::ColorSpaceKHR::SRGB_NONLINEAR,
            image_extent: vk::Extent2D {
                width: width,
                height: height,
            },
            image_array_layers: 1,
            image_usage: vk::ImageUsageFlags::COLOR_ATTACHMENT,
            queue_family_index_count: 1,
            p_queue_family_indices: &family_index,
            pre_transform: vk::SurfaceTransformFlagsKHR::IDENTITY,
            composite_alpha: composite_alpha,
            present_mode: vk::PresentModeKHR::FIFO,
            old_swapchain: old_swapchain,
            ..Default::default()
        };

        let swapchain = swapchain_loader
            .create_swapchain(&create_info, None)
            .expect("Can't create Vulkan swapchain.");

        swapchain
    }
}

fn get_swapchain_images(
    swapchain: vk::SwapchainKHR,
    swapchain_loader: &khr::Swapchain,
) -> Vec<vk::Image> {
    unsafe {
        let swapchain_images = swapchain_loader
            .get_swapchain_images(swapchain)
            .expect("Can't get Vulkan swapchain images.");

        swapchain_images
    }
}

fn create_swapchain(
    device: &ash::Device,
    swapchain_loader: &khr::Swapchain,
    surface: vk::SurfaceKHR,
    surface_caps: &vk::SurfaceCapabilitiesKHR,
    format: vk::Format,
    width: u32,
    height: u32,
    family_index: u32,
    old_swapchain: vk::SwapchainKHR,
) -> Swapchain {
    let swapchain_khr = create_swapchain_khr(
        &swapchain_loader,
        surface,
        &surface_caps,
        format,
        width,
        height,
        family_index,
        old_swapchain,
    );

    let images = get_swapchain_images(swapchain_khr, &swapchain_loader);

    let mut image_views: Vec<vk::ImageView> = Vec::with_capacity(images.len());
    for image in &images {
        let image_view = create_image_view(device, *image, format);

        image_views.push(image_view);
    }

    let swapchain = Swapchain {
        swapchain: swapchain_khr,
        images: images,
        image_views: image_views,
        width: width,
        height: height,
    };

    swapchain
}

fn resize_swapchain_if_changed(
    swapchain: &mut Swapchain,
    device: &ash::Device,
    swapchain_loader: &khr::Swapchain,
    surface: vk::SurfaceKHR,
    surface_caps: &vk::SurfaceCapabilitiesKHR,
    format: vk::Format,
    width: u32,
    height: u32,
    family_index: u32,
) -> bool {
    unsafe {
        let mut resized = false;

        if swapchain.width != width || swapchain.height != height {
            device.device_wait_idle().unwrap();

            for image_view in &swapchain.image_views {
                device.destroy_image_view(*image_view, None);
            }

            let old_swapchain_khr = swapchain.swapchain;
            *swapchain = create_swapchain(
                device,
                swapchain_loader,
                surface,
                surface_caps,
                format,
                width,
                height,
                family_index,
                swapchain.swapchain,
            );

            swapchain_loader.destroy_swapchain(old_swapchain_khr, None);

            resized = true;
        }

        resized
    }
}

fn create_image_view(
    device: &ash::Device,
    image: vk::Image,
    swapchain_format: vk::Format,
) -> vk::ImageView {
    unsafe {
        let image_view_create_info = vk::ImageViewCreateInfo {
            image: image,
            view_type: vk::ImageViewType::TYPE_2D,
            format: swapchain_format,
            subresource_range: vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                level_count: 1,
                layer_count: 1,
                ..Default::default()
            },
            ..Default::default()
        };

        let image_view = device
            .create_image_view(&image_view_create_info, None)
            .expect("Can't create Vulkan image view.");

        image_view
    }
}

fn create_framebuffer(
    device: &ash::Device,
    image_view: vk::ImageView,
    render_pass: vk::RenderPass,
    width: u32,
    height: u32,
) -> vk::Framebuffer {
    unsafe {
        let framebuffer_create_info = vk::FramebufferCreateInfo {
            render_pass: render_pass,
            attachment_count: 1,
            p_attachments: &image_view,
            width: width,
            height: height,
            layers: 1,
            ..Default::default()
        };

        let framebuffer = device
            .create_framebuffer(&framebuffer_create_info, None)
            .expect("Can't create Vulkan framebuffer.");

        framebuffer
    }
}

fn create_command_pool(device: &ash::Device, family_index: u32) -> vk::CommandPool {
    unsafe {
        let create_info = vk::CommandPoolCreateInfo {
            flags: vk::CommandPoolCreateFlags::TRANSIENT
                | vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER,
            queue_family_index: family_index,
            ..Default::default()
        };

        let command_pool = device
            .create_command_pool(&create_info, None)
            .expect("Can't create Vulkan command pool.");

        command_pool
    }
}

fn allocate_command_buffers(
    device: &ash::Device,
    command_pool: vk::CommandPool,
    count: u32,
) -> Vec<vk::CommandBuffer> {
    unsafe {
        let allocate_info = vk::CommandBufferAllocateInfo {
            command_pool: command_pool,
            level: vk::CommandBufferLevel::PRIMARY,
            command_buffer_count: count,
            ..Default::default()
        };

        let command_buffers = device
            .allocate_command_buffers(&allocate_info)
            .expect("Can't allocate Vulkan command buffers.");

        command_buffers
    }
}

fn create_semaphore(device: &ash::Device) -> vk::Semaphore {
    unsafe {
        let create_info = vk::SemaphoreCreateInfo::default();
        let semaphore = device
            .create_semaphore(&create_info, None)
            .expect("Can't create Vulkan semaphore.");

        semaphore
    }
}

fn create_fence(device: &ash::Device) -> vk::Fence {
    unsafe {
        let create_info = vk::FenceCreateInfo {
            flags: vk::FenceCreateFlags::SIGNALED,
            ..Default::default()
        };

        let fence = device
            .create_fence(&create_info, None)
            .expect("Can't create Vulkan fence.");

        fence
    }
}

fn create_render_pass(device: &ash::Device, swapchain_format: vk::Format) -> vk::RenderPass {
    unsafe {
        let attachment_description = [vk::AttachmentDescription {
            format: swapchain_format,
            samples: vk::SampleCountFlags::TYPE_1,
            load_op: vk::AttachmentLoadOp::CLEAR,
            store_op: vk::AttachmentStoreOp::STORE,
            stencil_load_op: vk::AttachmentLoadOp::DONT_CARE,
            stencil_store_op: vk::AttachmentStoreOp::DONT_CARE,
            initial_layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
            final_layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
            ..Default::default()
        }];

        let color_attachment = [vk::AttachmentReference {
            attachment: 0,
            layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
        }];

        let subpass_description = [vk::SubpassDescription {
            pipeline_bind_point: vk::PipelineBindPoint::GRAPHICS,
            color_attachment_count: color_attachment.len() as u32,
            p_color_attachments: color_attachment.as_ptr(),
            ..Default::default()
        }];

        let create_info = vk::RenderPassCreateInfo {
            attachment_count: attachment_description.len() as u32,
            p_attachments: attachment_description.as_ptr(),
            subpass_count: subpass_description.len() as u32,
            p_subpasses: subpass_description.as_ptr(),
            ..Default::default()
        };

        let render_pass = device
            .create_render_pass(&create_info, None)
            .expect("Can't create Vulkan render pass.");

        render_pass
    }
}

fn load_shader(device: &ash::Device, path: &str) -> vk::ShaderModule {
    unsafe {
        let vertex_shader_code = std::fs::read(path).expect("Can't read Vulkan spv file.");

        let shader_module_create_info = vk::ShaderModuleCreateInfo {
            code_size: vertex_shader_code.len(),
            p_code: vertex_shader_code.as_ptr() as *const u32,
            ..Default::default()
        };

        let shader_module = device
            .create_shader_module(&shader_module_create_info, None)
            .expect("Can't create Vulkan shader module.");

        shader_module
    }
}

fn create_pipeline_layout(device: &ash::Device) -> vk::PipelineLayout {
    unsafe {
        let pipeline_layout_create_info = vk::PipelineLayoutCreateInfo {
            ..Default::default()
        };

        let pipeline_layout = device
            .create_pipeline_layout(&pipeline_layout_create_info, None)
            .expect("Can't create Vulkan pipeline layout.");

        pipeline_layout
    }
}

fn create_graphics_pipeline(
    device: &ash::Device,
    vs: vk::ShaderModule,
    fs: vk::ShaderModule,
    pipeline_layout: vk::PipelineLayout,
    render_pass: vk::RenderPass,
) -> vk::Pipeline {
    unsafe {
        let stages = [
            vk::PipelineShaderStageCreateInfo {
                stage: vk::ShaderStageFlags::VERTEX,
                module: vs,
                p_name: CStr::from_bytes_with_nul_unchecked(b"main\0").as_ptr(),
                ..Default::default()
            },
            vk::PipelineShaderStageCreateInfo {
                stage: vk::ShaderStageFlags::FRAGMENT,
                module: fs,
                p_name: CStr::from_bytes_with_nul_unchecked(b"main\0").as_ptr(),
                ..Default::default()
            },
        ];

        let vertex_input_state = vk::PipelineVertexInputStateCreateInfo {
            ..Default::default()
        };

        let input_assembly_state = vk::PipelineInputAssemblyStateCreateInfo {
            topology: vk::PrimitiveTopology::TRIANGLE_LIST,
            ..Default::default()
        };

        let tessellation_state = vk::PipelineTessellationStateCreateInfo {
            ..Default::default()
        };

        let viewport_state = vk::PipelineViewportStateCreateInfo {
            viewport_count: 1,
            scissor_count: 1,
            ..Default::default()
        };

        let rasterization_state = vk::PipelineRasterizationStateCreateInfo {
            line_width: 1.0,
            ..Default::default()
        };

        let multisample_state = vk::PipelineMultisampleStateCreateInfo {
            rasterization_samples: vk::SampleCountFlags::TYPE_1,
            ..Default::default()
        };

        let depth_stencil_state = vk::PipelineDepthStencilStateCreateInfo {
            ..Default::default()
        };

        let pipeline_color_attachment_blend_state = vk::PipelineColorBlendAttachmentState {
            color_write_mask: vk::ColorComponentFlags::R
                | vk::ColorComponentFlags::G
                | vk::ColorComponentFlags::B
                | vk::ColorComponentFlags::A,
            ..Default::default()
        };

        let color_blend_state = vk::PipelineColorBlendStateCreateInfo {
            attachment_count: 1,
            p_attachments: &pipeline_color_attachment_blend_state,
            ..Default::default()
        };

        let dynamic_states = [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];

        let dynamic_state = vk::PipelineDynamicStateCreateInfo {
            dynamic_state_count: dynamic_states.len() as u32,
            p_dynamic_states: dynamic_states.as_ptr(),
            ..Default::default()
        };

        let create_info = vk::GraphicsPipelineCreateInfo {
            stage_count: stages.len() as u32,
            p_stages: stages.as_ptr(),
            p_vertex_input_state: &vertex_input_state,
            p_input_assembly_state: &input_assembly_state,
            p_tessellation_state: &tessellation_state,
            p_viewport_state: &viewport_state,
            p_rasterization_state: &rasterization_state,
            p_multisample_state: &multisample_state,
            p_depth_stencil_state: &depth_stencil_state,
            p_color_blend_state: &color_blend_state,
            p_dynamic_state: &dynamic_state,
            layout: pipeline_layout,
            render_pass: render_pass,
            ..Default::default()
        };

        let graphics_pipeline = device
            .create_graphics_pipelines(vk::PipelineCache::null(), &[create_info], None)
            .expect("Can't create Vulkan graphics pipeline.");

        graphics_pipeline[0]
    }
}

fn main() {
    let event_loop = EventLoop::new();
    let window = WindowBuilder::new()
        .with_title("Vulkan Test")
        .with_inner_size(winit::dpi::LogicalSize::new(
            f64::from(1600.0),
            f64::from(900.0),
        ))
        .build(&event_loop)
        .unwrap();

    let entry = ash::Entry::linked();

    let instance = create_instance(&entry);

    let surface_loader = khr::Surface::new(&entry, &instance);
    let surface = create_surface(&entry, &instance, &window);

    let physical_device = pick_physical_device(&instance, surface, &surface_loader);
    let graphics_family_index = get_graphics_family_index(&instance, physical_device);

    let device = create_device(&instance, physical_device, graphics_family_index);

    let graphics_queue = get_device_queue(&device, graphics_family_index);

    let surface_caps = get_surface_capabilities(physical_device, surface, &surface_loader);
    let swapchain_format = get_swapchain_format(physical_device, surface, &surface_loader);

    let swapchain_loader = khr::Swapchain::new(&instance, &device);
    let mut swapchain = create_swapchain(
        &device,
        &swapchain_loader,
        surface,
        &surface_caps,
        swapchain_format,
        window.inner_size().width,
        window.inner_size().height,
        graphics_family_index,
        vk::SwapchainKHR::null(),
    );

    let command_pool = create_command_pool(&device, graphics_family_index);

    let command_buffers = allocate_command_buffers(&device, command_pool, 2);

    let acquire_semaphore = create_semaphore(&device);
    let submit_semaphore = create_semaphore(&device);

    let command_buffer_fences = [create_fence(&device), create_fence(&device)];

    let render_pass = create_render_pass(&device, swapchain_format);

    let mut framebuffers: Vec<vk::Framebuffer> = Vec::with_capacity(swapchain.image_views.len());
    for image_view in &swapchain.image_views {
        let framebuffer = create_framebuffer(
            &device,
            *image_view,
            render_pass,
            swapchain.width,
            swapchain.height,
        );

        framebuffers.push(framebuffer);
    }

    let vertex_shader = load_shader(&device, "data\\shaders\\triangle.vert.spv");
    let fragment_shader = load_shader(&device, "data\\shaders\\triangle.frag.spv");

    let pipeline_layout = create_pipeline_layout(&device);

    let graphics_pipeline = create_graphics_pipeline(
        &device,
        vertex_shader,
        fragment_shader,
        pipeline_layout,
        render_pass,
    );

    let mut frame_index = 0;
    unsafe {
        event_loop.run(move |event, _, control_flow| {
            *control_flow = ControlFlow::Poll;

            match event {
                Event::WindowEvent {
                    event:
                        WindowEvent::CloseRequested
                        | WindowEvent::KeyboardInput {
                            input:
                                KeyboardInput {
                                    state: ElementState::Pressed,
                                    virtual_keycode: Some(VirtualKeyCode::Escape),
                                    ..
                                },
                            ..
                        },
                    ..
                } => *control_flow = ControlFlow::Exit,
                Event::MainEventsCleared => {
                    if window.inner_size().width > 0 && window.inner_size().height > 0 {
                        if resize_swapchain_if_changed(
                            &mut swapchain,
                            &device,
                            &swapchain_loader,
                            surface,
                            &surface_caps,
                            swapchain_format,
                            window.inner_size().width,
                            window.inner_size().height,
                            graphics_family_index,
                        ) {
                            for i in 0..framebuffers.len() {
                                device.destroy_framebuffer(framebuffers[i], None);

                                framebuffers[i] = create_framebuffer(
                                    &device,
                                    swapchain.image_views[i],
                                    render_pass,
                                    swapchain.width,
                                    swapchain.height,
                                );
                            }
                        }

                        device
                            .wait_for_fences(
                                &[command_buffer_fences[frame_index]],
                                true,
                                std::u64::MAX,
                            )
                            .unwrap();
                        device
                            .reset_fences(&[command_buffer_fences[frame_index]])
                            .unwrap();

                        let image_index = swapchain_loader
                            .acquire_next_image(
                                swapchain.swapchain,
                                std::u64::MAX,
                                acquire_semaphore,
                                vk::Fence::null(),
                            )
                            .unwrap()
                            .0;

                        let command_begin_info = vk::CommandBufferBeginInfo {
                            flags: vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT,
                            ..Default::default()
                        };

                        device
                            .begin_command_buffer(command_buffers[frame_index], &command_begin_info)
                            .unwrap();

                        let render_begin_barrier = vk::ImageMemoryBarrier {
                            dst_access_mask: vk::AccessFlags::COLOR_ATTACHMENT_WRITE,
                            old_layout: vk::ImageLayout::UNDEFINED,
                            new_layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
                            src_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
                            dst_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
                            image: swapchain.images[image_index as usize],
                            subresource_range: vk::ImageSubresourceRange {
                                aspect_mask: vk::ImageAspectFlags::COLOR,
                                level_count: 1,
                                layer_count: 1,
                                ..Default::default()
                            },
                            ..Default::default()
                        };
                        device.cmd_pipeline_barrier(
                            command_buffers[frame_index],
                            vk::PipelineStageFlags::BOTTOM_OF_PIPE,
                            vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
                            vk::DependencyFlags::BY_REGION,
                            &[],
                            &[],
                            &[render_begin_barrier],
                        );

                        let clear_values = [vk::ClearValue {
                            color: vk::ClearColorValue {
                                float32: [0.2, 0.4, 0.6, 0.0],
                            },
                        }];

                        let render_pass_begin_info = vk::RenderPassBeginInfo {
                            render_pass: render_pass,
                            framebuffer: framebuffers[image_index as usize],
                            render_area: vk::Rect2D {
                                offset: vk::Offset2D { x: 0, y: 0 },
                                extent: vk::Extent2D {
                                    width: swapchain.width,
                                    height: swapchain.height,
                                },
                            },
                            clear_value_count: clear_values.len() as u32,
                            p_clear_values: clear_values.as_ptr(),
                            ..Default::default()
                        };

                        device.cmd_begin_render_pass(
                            command_buffers[frame_index],
                            &render_pass_begin_info,
                            vk::SubpassContents::INLINE,
                        );

                        let viewport = vk::Viewport {
                            x: 0.0,
                            y: 0.0,
                            width: swapchain.width as f32,
                            height: swapchain.height as f32,
                            min_depth: 0.0,
                            max_depth: 1.0,
                        };
                        let scissor = vk::Rect2D {
                            offset: vk::Offset2D { x: 0, y: 0 },
                            extent: vk::Extent2D {
                                width: swapchain.width,
                                height: swapchain.height,
                            },
                        };

                        device.cmd_set_viewport(command_buffers[frame_index], 0, &[viewport]);
                        device.cmd_set_scissor(command_buffers[frame_index], 0, &[scissor]);

                        device.cmd_bind_pipeline(
                            command_buffers[frame_index],
                            vk::PipelineBindPoint::GRAPHICS,
                            graphics_pipeline,
                        );
                        device.cmd_draw(command_buffers[frame_index], 3, 1, 0, 0);

                        device.cmd_end_render_pass(command_buffers[frame_index]);

                        let render_end_barrier = vk::ImageMemoryBarrier {
                            src_access_mask: vk::AccessFlags::COLOR_ATTACHMENT_WRITE,
                            old_layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
                            new_layout: vk::ImageLayout::PRESENT_SRC_KHR,
                            src_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
                            dst_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
                            image: swapchain.images[image_index as usize],
                            subresource_range: vk::ImageSubresourceRange {
                                aspect_mask: vk::ImageAspectFlags::COLOR,
                                level_count: 1,
                                layer_count: 1,
                                ..Default::default()
                            },
                            ..Default::default()
                        };
                        device.cmd_pipeline_barrier(
                            command_buffers[frame_index],
                            vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
                            vk::PipelineStageFlags::TOP_OF_PIPE,
                            vk::DependencyFlags::BY_REGION,
                            &[],
                            &[],
                            &[render_end_barrier],
                        );

                        device
                            .end_command_buffer(command_buffers[frame_index])
                            .unwrap();

                        let submit_stage_mask = vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT;
                        let submit_info = [vk::SubmitInfo {
                            wait_semaphore_count: 1,
                            p_wait_semaphores: &acquire_semaphore,
                            p_wait_dst_stage_mask: &submit_stage_mask,
                            command_buffer_count: 1,
                            p_command_buffers: &command_buffers[frame_index],
                            signal_semaphore_count: 1,
                            p_signal_semaphores: &submit_semaphore,
                            ..Default::default()
                        }];
                        device
                            .queue_submit(
                                graphics_queue,
                                &submit_info,
                                command_buffer_fences[frame_index],
                            )
                            .unwrap();

                        let present_info = vk::PresentInfoKHR {
                            wait_semaphore_count: 1,
                            p_wait_semaphores: &submit_semaphore,
                            swapchain_count: 1,
                            p_swapchains: &swapchain.swapchain,
                            p_image_indices: &image_index,
                            ..Default::default()
                        };

                        swapchain_loader
                            .queue_present(graphics_queue, &present_info)
                            .unwrap();

                        frame_index = (frame_index + 1) % 2;
                    }
                }
                _ => (),
            }
        });
    }
}
