#[cfg(feature="gpu_enabled")] #[macro_use] extern crate rustacuda;
#[cfg(feature="gpu_enabled")] extern crate rustacuda_derive;
#[cfg(feature="gpu_enabled")] extern crate rustacuda_core;
#[cfg(feature="gpu_enabled")] use rustacuda::prelude::*;
#[cfg(feature="gpu_enabled")] use std::ffi::CString;

mod rc4;

use threadpool::ThreadPool;
use std::{env, time::{Instant, Duration}, sync::{Arc, Mutex, mpsc::channel}, thread, error::Error};

#[allow(unused)] const ONE_UNKNOWNS: u128 = 1000000;
#[allow(unused)] const TWO_UNKNOWNS: u128 = 1000000;
#[allow(unused)] const THREE_UNKNOWNS: u128 = 17000000;
#[allow(unused)] const FOUR_UNKNOWNS: u128 = 4295000000;
#[allow(unused)] const FIVE_UNKNOWNS: u128 = 1099512000000;
#[allow(unused)] const SIX_UNKNOWNS: u128 = 281475000000000;
#[allow(unused)] const SEVEN_UNKNOWNS: u128 = 72057590000000000;
#[allow(unused)] const EIGHT_UNKNOWNS: u128 = 18446740000000000000;



/// a multithreaded implementation of a rc4 plaintext attack
/// 
/// arguments should be provided as follows:
/// *0. __name__ - (automatically provided by execution)
/// 1. hex mask of the key
/// 	ex: 0x11??2345??1920
/// 	ex: 0x??????????12839490
/// 2. hex of the desired keystream
/// 	ex: 0x1234567890
/// 3. start_from_bottom
/// 	if true then the search will start from mask of 0 and work up
/// 	if false then the search will start from make of 255 and work down
/// 4. (optional) the drop_n before generating the keystream
/// 5. (optional) number of threads to utilize in the threadpool
/// 
///  ./target/release/rc4_pt_attack {key hex mask} {8 byte keystream} {start from bottom}
///     --drop_n={drop_n} -cpu/-gpu --threads={num_threads or threads per block} --blocks={gpu blocks (GPU ONLY)}
/// panics when unacceptable number of arguments are provided
/// 
/// if no string is found the found key will read: "KEY WAS NOT FOUND"
fn main() -> Result<(), Box<dyn Error>> {
    let cuda_env = "CUDA_ENV";
    let mut cpu = match env::var(cuda_env) {
        Ok(_) => false,
        Err(_) => true,
    };

	let args: Vec<_> = env::args().collect();

	let start_from_bottom = match &args[3][..] {
		"true" | "T" | "True" | "1" => true,
		"false" | "F" | "False" | "0" => false,
        _ => panic!("unknown start_from_bottom argument"),
	};

	let now = Instant::now();

	let mut key = rc4::mask_to_key(args.get(1).unwrap(), start_from_bottom);
	let desired_keystream = rc4::hex_to_u8(args.get(2).unwrap(), args.get(2).unwrap().len());
    let drop_n: u32 = args.get(4).unwrap().strip_prefix("--drop_n=").unwrap().parse().unwrap();

    if !cpu {
        let cpu_gpu_string = args.get(5).unwrap();
        cpu = match &cpu_gpu_string[..] {
            "-cpu" | "cpu" | "-CPU" | "CPU" | "-c" | "-C" | "--cpu" | "--CPU" => true,
            "-gpu" | "gpu" | "-GPU" | "GPU" | "-g" | "-G" | "--gpu" | "--GPU" => false,
            _ => panic!("unknown compute device target"),
        };
    }

    let max_iterations: u128 = 256u128.pow((key.mask_vec.len()) as u32);
    let desired_keystream_length = desired_keystream.len();

    // if desired_keystream.len() != 8 {
    //     println!("unnacceptable desired keystream length for gpu, execution will commence on cpu");
    //     cpu = true;
    // }

    // if !cpu {
    //     if key.key_vec.len() != 72 && key.key_vec.len() != 104 {
    //         println!("CUDA kernel does not exist for given key length, switching to cpu");
    //         cpu = true;
    //     }

    //     if drop_n != 0 && drop_n != 256 && drop_n != 267 {
    //         println!("CUDA kernel does not exist for given drop_n, switching to cpu");
    //         cpu = true;
    //     }
    // }

    #[cfg(not(feature="gpu_enabled"))]
    if cpu {
        let thread_num = match args.get(6) {
            Some(val) => val.strip_prefix("--threads=").unwrap().parse().unwrap(),
            _ => 1,
        };

        // Multithreading stuff //

        // ThreadPool
        let n_workers = thread_num;
        let pool = ThreadPool::new(n_workers);

        // Shared values
        // although it will not be edited it is necessary to make a mutex lock on the desired_keystream vector
        let shared_desired_keystream = Arc::new(Mutex::new(desired_keystream));

        let no_key_found_string = "KEY WAS NOT FOUND";

        // channel for returning the correct key
        let (tx, rx) = channel();

        for i in 0..max_iterations {
            let current_key = key.deep_copy();

            let shared_desired_keystream = shared_desired_keystream.clone();

            let tx = tx.clone();
            pool.execute(move || {
                if i % 0x1000000 == 0 {
                    println!("iteration: {i:X}");
                    let key_hex_representation = rc4::u8_to_hex(&current_key.key_vec, current_key.key_vec.len() as u32);
                    println!("current key value: {key_hex_representation}");
                }

                let mut s = rc4::ksa(&current_key.key_vec);
                let ks = rc4::prga(&mut s, desired_keystream_length as u32, drop_n);

                if i >= max_iterations - 1 {
                    let no_key_found_string = no_key_found_string.to_string();
                    tx.send(no_key_found_string).unwrap();
                }

                if rc4::compare_ks(&ks, & *shared_desired_keystream.lock().unwrap()) {
                    println!("found kex at {i}");
                    let key_hex_representation = rc4::u8_to_hex(&current_key.key_vec, current_key.key_vec.len() as u32);
                    tx.send(key_hex_representation).unwrap();
                    println!("key send");
                }
            });
            // ensure pool doesn't get overloaded with queued tasks
            if i % 0x1000 == 0 {
                pool.join();
            }
            rc4::change_key(&mut key, start_from_bottom);

            let received_val = rx.try_recv();
            match received_val {
                Ok(desired_key_hex) => {
                    println!("found key: {desired_key_hex}");
                    let elapsed = now.elapsed();
                    println!("exeuction took: {:.2?}", elapsed);
                    return Ok(());
                },
                _ => continue,
            };
        }

        loop {
            let received_val = rx.try_recv();
            match received_val {
                Ok(desired_key_hex) => {
                    println!("found key: {desired_key_hex}");
                    let elapsed = now.elapsed();
                    println!("execution took: {:.2?}", elapsed);
                    return Ok(());
                },
                _ => {
                    thread::sleep(Duration::from_secs(1));
                },
            }
        }
    } else {
        panic!("cannot run on gpu if not enabled");
    }

    #[cfg(feature="gpu_enabled")]
    if cpu {
        let thread_num = match args.get(6) {
            Some(val) => val.strip_prefix("--threads=").unwrap().parse().unwrap(),
            _ => 1,
        };

        // Multithreading stuff //
	
	    // ThreadPool
        let n_workers = thread_num;
        let pool = ThreadPool::new(n_workers);

        // Shared values
        // although it will not be edited it is necessary to make a mutex lock on the desired_keystream vector
        let shared_desired_keystream = Arc::new(Mutex::new(desired_keystream));

        let no_key_found_string = "KEY WAS NOT FOUND";

        // channel for returning the correct key
        let (tx, rx) = channel();

        for i in 0..max_iterations {
            let current_key = key.deep_copy();

            let shared_desired_keystream = shared_desired_keystream.clone();

            let tx = tx.clone();
            pool.execute(move || {
                if i % 0x1000000 == 0 {
                    println!("iteration: {i:X}");
                    let key_hex_representation = rc4::u8_to_hex(&current_key.key_vec, current_key.key_vec.len() as u32);
                    println!("current key value: {key_hex_representation}");
                }

                let mut s = rc4::ksa(&current_key.key_vec);
                let ks = rc4::prga(&mut s, desired_keystream_length as u32, drop_n);

                if i >= max_iterations - 1 {
                    let no_key_found_string = no_key_found_string.to_string();
                    tx.send(no_key_found_string).unwrap();
                }

                if rc4::compare_ks(&ks, & *shared_desired_keystream.lock().unwrap()) {
                    println!("found kex at {i}");
                    let key_hex_representation = rc4::u8_to_hex(&current_key.key_vec, current_key.key_vec.len() as u32);
                    tx.send(key_hex_representation).unwrap();
                    println!("key send");
                }
            });
            // ensure pool doesn't get overloaded with queued tasks
            if i % 0x1000 == 0 {
                pool.join();
            }
            rc4::change_key(&mut key, start_from_bottom);

            let received_val = rx.try_recv();
            match received_val {
                Ok(desired_key_hex) => {
                    println!("found key: {desired_key_hex}");
                    let elapsed = now.elapsed();
                    println!("exeuction took: {:.2?}", elapsed);
                    return Ok(());
                },
                _ => continue,
            };
        }

        loop {
            let received_val = rx.try_recv();
            match received_val {
                Ok(desired_key_hex) => {
                    println!("found key: {desired_key_hex}");
                    let elapsed = now.elapsed();
                    println!("execution took: {:.2?}", elapsed);
                    return Ok(());
                },
                _ => {
                    thread::sleep(Duration::from_secs(1));
                },
            }
        }
    } else {
        let threads = match args.get(6) {
            Some(val) => val.strip_prefix("--threads=").unwrap().parse().unwrap(),
            _ => 1,
        };

        let blocks = match args.get(7) {
            Some(val) => val.strip_prefix("--blocks=").unwrap().parse().unwrap(),
            _ => 1,
        };

        let mut keystreams: Vec<u8> = match key.mask_vec.len() {
            1 => Vec::with_capacity((ONE_UNKNOWNS * 8) as usize),
            2 => Vec::with_capacity((TWO_UNKNOWNS * 8) as usize),
            3 => Vec::with_capacity((THREE_UNKNOWNS * 8) as usize),
            4 => Vec::with_capacity((FOUR_UNKNOWNS * 8) as usize),
            5 => Vec::with_capacity((FIVE_UNKNOWNS * 8) as usize),
            6 => Vec::with_capacity((SIX_UNKNOWNS * 8) as usize),
            7 => Vec::with_capacity((SEVEN_UNKNOWNS * 8) as usize),
            8 => Vec::with_capacity((EIGHT_UNKNOWNS * 8) as usize),
            _ => panic!("unsupported number of unknowns :("),
        };

        let key_length = key.key_vec.len();

        // Create all of the possible strings
        println!("creating all possible strings");
        let mut possible_keys: Vec<u8> = Vec::with_capacity((max_iterations as usize) * key.key_vec.len());
        for _ in 0..max_iterations {
            for val in key.key_vec.iter() {
                possible_keys.push(*val);
            }
            rc4::change_key(&mut key, start_from_bottom);
        }

        // Begin rustacuda code
        rustacuda::init(CudaFlags::empty())?;
        let device = Device::get_device(0)?;
        let _ctx = Context::create_and_push(ContextFlags::MAP_HOST | ContextFlags::SCHED_AUTO, device)?;

        let ptx = CString::new(include_str!("cuda/rc4.cu"))?;
        let module = Module::load_from_string(&ptx)?;

        let function_name = match (key.key_vec.len(), drop_n) {
            (72, 0) => CString::new("rc4_keystream_gen_72_drop_0").unwrap(),
            (72, 256) => CString::new("rc4_keystream_gen_72_drop_256").unwrap(),
            (72, 267) => CString::new("rc4_keystream_gen_72_drop_267").unwrap(),
            (104, 0) => CString::new("rc4_keystream_gen_104_drop_0").unwrap(),
            (104, 256) => CString::new("rc4_keystream_gen_104_drop_256").unwrap(),
            (104, 267) => CString::new("rc4_keystream_gen_104_drop_267").unwrap(),
            _ => panic!("unallowed key length - drop_n combination for gpu compute"),
        };
        let kernel_function = module.get_function(&function_name)?;
        
        // The CUDA code will work as follows:
        // 1. all of the provided blocks will be given 1000000 keystreams to compute
        // 2. as each block completes its 1000000 keystreams the stream it is on will run a callback
        // to tell the main thread that the keystreams have been computed.
        // 3. the main thread will copy the given 1000000 keystreams into a buffer of keystreams to process.
        // 4. the main thread will then copy the next 10000000 keys into the previously used device buffer for the given stream.
        // 5. the main thread will tell the block given before to launch the rc4 function again
        // 6. upon receiving the first 1000000 keystreams a second cpu thread will be deployed to begin processing the keystreams.
        // Instantiate vectors for holding
        let mut key_buffers = Vec::with_capacity(blocks);
        let mut keystream_buffers = Vec::with_capacity(blocks);
        let mut streams = Vec::with_capacity(blocks);
        let mut streams_done = Vec::with_capacity(blocks);

        let mut task_order = Vec::new();
        let mut end_order = Vec::new();

        // Initialize all of the buffers and begin the kernels
        for i in 0..blocks {
            let mut key_buffer = DeviceBuffer::from_slice(&possible_keys[(i*1000000)..((i+1)*1000000)]).unwrap();
            let mut keystream_buffer = DeviceBuffer::from_slice(&[0u8; 8000000]).unwrap();
            let stream = Stream::new(StreamFlags::NON_BLOCKING, None)?;
            streams_done.push(false);
            stream.add_callback(Box::new(|status| {
                println!("Device status is {:?}", status);
                match status {
                    Ok(_) => streams_done[i] = true,
                    Err(_) => panic!("error with device"),
                };
            }));

            // begin the kernel
            unsafe {
                let result = launch!(kernel_function<<<1, threads, 0, stream>>>(
                    key_buffer.as_device_ptr(),
                    1000000,
                    keystream_buffer.as_device_ptr()
                ));
                result?;
            }

            key_buffers.push(key_buffer);
            keystream_buffers.push(keystream_buffer);
            streams.push(stream);
            task_order.push(i);
        }

        // Continue to check completed streams and move their values to the cpu thread dedicated to checking for
        // matches.
        let mut current_iteration: u128 = (blocks * 1000000) as u128;
        while current_iteration < max_iterations {
            for i in 0..blocks {
                if streams_done[i] {
                    // Copy all values from out keystream buffer to the out buffer
                    let mut out_buff = [0u8; 1000000];
                    keystream_buffers[i].copy_to(&mut out_buff)?;
                    for val in out_buff {
                        keystreams.push(val);
                    }
                    // Start next launch
                    key_buffers[i] = DeviceBuffer::from_slice(&possible_keys[(current_iteration as usize)..((current_iteration + 1000000) as usize)]).unwrap();
                    keystream_buffers[i] = DeviceBuffer::from_slice(&[0u8; 8000000]).unwrap();
                    let stream = Stream::new(StreamFlags::NON_BLOCKING, None)?;
                    stream.add_callback(Box::new(|status| {
                        println!("Device status is {:?}", status);
                        match status {
                            Ok(_) => streams_done[i] = true,
                            Err(_) => panic!("error with device"),
                        };
                    }));

                    unsafe {
                        let result = launch!(kernel_function<<<1, threads, 0,stream>>>(
                            key_buffers[i].as_device_ptr(),
                            1000000,
                            keystream_buffers[i].as_device_ptr()
                        ));
                        result?;
                    }

                    streams.push(stream);
                    streams_done[i] = false;
                    current_iteration += 1000000;
                    end_order.push(i);
                    task_order.push(i);
                }
            }
        }

        for i in 0..blocks {
            while streams_done[i] == false {
                continue;
            }
        }

        for i in 0..blocks {
            // Copy all values from out keystream buffer to the out buffer
            let mut out_buff = [0u8; 8000000];
            keystream_buffers[i].copy_to(&mut out_buff)?;
            for val in out_buff {
                keystreams.push(val);
            }
            task_order.push(i);
        }

        println!("looking through generated keystreams");
        let correct_key = rc4::find_correct_key(possible_keys, keystreams, task_order, end_order, desired_keystream, key_length, blocks as u128);
        let correct_key_hex = rc4::u8_to_hex(&correct_key, key_length as u32);
        println!("found key of: {correct_key_hex}");
        return Ok(());
    }
}