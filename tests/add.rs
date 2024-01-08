#[macro_use]
extern crate rustacuda;
extern crate rustacuda_core;

use rustacuda::prelude::*;
use rustacuda::memory::DeviceBox;
use std::error::Error;
use std::ffi::CString;

#[test]
fn test_cuda_initialized_one_addition() -> Result<(), Box<dyn Error>> {
    // Initialize the CUDA API
    rustacuda::init(CudaFlags::empty())?;

    // Get the first device to run on
    let device = Device::get_device(0)?;
    
    // Create a context associated to this device
    let _context = Context::create_and_push(ContextFlags::MAP_HOST | ContextFlags::SCHED_AUTO, device)?;
    
    // Load the module containing the function we want to call
    let module_data = CString::new(include_str!("add/add.ptx"))?;
    let module = Module::load_from_string(&module_data)?;
    
    // Create a stream to submit work to
    let stream = Stream::new(StreamFlags::NON_BLOCKING, None)?;
    
    // Allocate space on the dvice and copy numbers to it.
    let mut x = DeviceBox::new(&10.0f32)?;
    let mut y = DeviceBox::new(&20.0f32)?;
    let mut result = DeviceBox::new(&0.0f32)?;

    // Launching kernels is unsafe Rust (i.e. it is basically a foreign-function call).
    unsafe {
        launch!(module.sum<<<1, 1, 0, stream>>>(
            x.as_device_ptr(),
            y.as_device_ptr(),
            result.as_device_ptr(),
            1
        ))?;
    }

    // The kernel launch is asynchronous, so we wait for the kernel to finish executing
    stream.synchronize()?;

    // Copy the result back to the host
    let mut result_host = 0.0f32;
    result.copy_to(&mut result_host)?;

    println!("Sum is {}", result_host);

    Ok(())
}

#[test]
fn test_cuda_initialized_multiple_additions() -> Result<(), Box<dyn Error>> {
    // Set up context, load the module, and create a stream to run kernels in.
    let _ctx = rustacuda::quick_init()?;
    let ptx = CString::new(include_str!("add/add.ptx"))?;
    let module = Module::load_from_string(&ptx)?;
    let stream = Stream::new(StreamFlags::NON_BLOCKING, None)?;

    // Create buffers for data
    let mut in_x = DeviceBuffer::from_slice(&[1.0f32; 10])?;
    let mut in_y = DeviceBuffer::from_slice(&[2.0f32; 10])?;
    let mut out_1 = DeviceBuffer::from_slice(&[0.0f32; 10])?;
    let mut out_2 = DeviceBuffer::from_slice(&[0.0f32; 10])?;

    // Kernel adds each element in `in_x` and `in_y` and writes the result into `out`.
    unsafe {
        // Launch the kernel with one block of one thread, no dynamic shared memory, on `stream`.
        let result = launch!(module.sum<<<1, 1, 0, stream>>>(
            in_x.as_device_ptr(),
            in_y.as_device_ptr(),
            out_1.as_device_ptr(),
            out_1.len()
        ));

        result?;

        // Launch the kernel again using the `function` form:
        let function_name = CString::new("sum")?;
        let sum = module.get_function(&function_name)?;
        // Launch with 1x1x1 (1) blocks of 10x1x1 (10) threads, to show that you can usue tuples to configure grid and block size.
        let result = launch!(sum<<<(1, 1, 1), (10, 1, 1), 0, stream>>>(
            in_x.as_device_ptr(),
            in_y.as_device_ptr(),
            out_2.as_device_ptr(),
            out_2.len()
        ));

        result?;
    }

    // Wait for kernels to finish executing
    stream.synchronize()?;

    // Copy the results back to host memory
    let mut out_host = [0.0f32; 20];
    out_1.copy_to(&mut out_host[0..10])?;
    out_2.copy_to(&mut out_host[10..20])?;

    for x in out_host.iter() {
        assert_eq!(3.0, *x);
    }

    Ok(())
}