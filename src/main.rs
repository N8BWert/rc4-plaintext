#[cfg(feature="gpu_enabled")] #[macro_use] extern crate rustacuda;
#[cfg(feature="gpu_enabled")] extern crate rustacuda_derive;
#[cfg(feature="gpu_enabled")] extern crate rustacuda_core;
#[cfg(feature="gpu_enabled")] use rustacuda::prelude::*;
#[cfg(feature="gpu_enabled")] use rustacuda::stream::StreamFlags;
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
#[allow(unused)] const NINE_UNKNOWNS: u128 = 4722366000000000000000;
#[allow(unused)] const TEN_UNKNOWNS: u128 = 1208926000000000000000000;
#[allow(unused)] const ELEVEN_UNKNOWNS: u128 = 309485000000000000000000000;
#[allow(unused)] const TWELVE_UNKNOWNS: u128 = 79228160000000000000000000000;
#[allow(unused)] const THIRTEEN_UNKNOWNS: u128 = 20282410000000000000000000000000;
#[allow(unused)] const FOURTEEN_UNKNOWNS: u128 = 5192297000000000000000000000000000;
#[allow(unused)] const FIFTEEN_UNKNOWNS: u128 = 1329228000000000000000000000000000000;

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
/// 5. (optional) --cpu or --gpu
/// 6. (optional) number of threads to utilize in the threadpool (or threads per gpu block)
/// 7. (optional) number of gpu blocks (GPU ONLY)
/// 
///  ./target/release/rc4_pt_attack {key hex mask} {8 byte keystream} {start from bottom}
///     --drop_n={drop_n} -cpu/-gpu --threads={num_threads or threads per block} --blocks={gpu blocks (GPU ONLY)}
/// panics when unacceptable number of arguments are provided
/// 
/// if no string is found the found key will read: "KEY WAS NOT FOUND"
fn main() -> Result<(), Box<dyn Error>> {
    let cuda_env = "CUDA_LIBRARY_PATH";
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

    if desired_keystream.len() != 8 {
        println!("unnacceptable desired keystream length for gpu, execution will commence on cpu");
        cpu = true;
    }

    if !cpu {
        if key.key_vec.len() != 72 && key.key_vec.len() != 104 {
            println!("CUDA kernel does not exist for given key length, switching to cpu");
            cpu = true;
        }

        if drop_n != 0 && drop_n != 256 && drop_n != 267 {
            println!("CUDA kernel does not exist for given drop_n, switching to cpu");
            cpu = true;
        }
    }

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

        let total_keystreams = match key.mask_vec.len() {
            1 => ONE_UNKNOWNS,
            2 => TWO_UNKNOWNS,
            3 => THREE_UNKNOWNS,
            4 => FOURTEEN_UNKNOWNS,
            5 => FIVE_UNKNOWNS,
            6 => SIX_UNKNOWNS,
            7 => SEVEN_UNKNOWNS,
            8 => EIGHT_UNKNOWNS,
            9 => NINE_UNKNOWNS,
            10 => TEN_UNKNOWNS,
            11 => ELEVEN_UNKNOWNS,
            12 => TWELVE_UNKNOWNS,
            13 => THIRTEEN_UNKNOWNS,
            14 => FOURTEEN_UNKNOWNS,
            15 => FIFTEEN_UNKNOWNS,
            _ => panic!("unspported mask size for key"),
        };

        let key_length = key.key_vec.len();

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
            (64, 0) => CString::new("rc4_keystream_gen_64_drop_0").unwrap(),
            (64, 256) => CString::new("rc4_keystream_gen_64_drop_256").unwrap(),
            (64, 267) => CString::new("rc4_keystream_gen_64_drop_267").unwrap(),
            (128, 0) => CString::new("rc4_keystream_gen_128_drop_0").unwrap(),
            (128, 256) => CString::new("rc4_keystream_gen_128_drop_256").unwrap(),
            (128, 267) => CString::new("rc4_keystream_gen_128_drop_267").unwrap(),
            (32, 0) => CString::new("rc4_keystream_gen_32_drop_0").unwrap(),
            (32, 256) => CString::new("rc4_keystream_gen_32_drop_256").unwrap(),
            (32, 267) => CString::new("rc4_keystream_gen_32_drop_267").unwrap(),
            _ => panic!("unallowed key length - drop_n combination for gpu compute"),
        };
        let kernel_function = module.get_function(&function_name)?;

        let mut current_key = key.deep_copy();
        let mut stream_done = Vec::with_capacity(blocks);
        let mut key_buffers: Vec<DeviceBuffer<u8>> = Vec::with_capacity(blocks);
        let mut keystream_buffers: Vec<DeviceBuffer<u8>> = Vec::with_capacity(blocks);

        for i in 0..blocks {
            let keys = rc4::generate_keys(&mut current_key, start_from_bottom);
            let mut key_buffer = DeviceBuffer::from_slice(&keys[..]).unwrap();
            drop(keys);
            let mut keystream_buffer = DeviceBuffer::from_slice(&[0u8; 8000000]).unwrap();
            let stream = Stream::new(StreamFlags::NON_BLOCKING, None)?;
            stream_done.push(false);

            unsafe {
                let result = launch!(kernel_function<<<1, threads, 0, stream>>>(
                    key_buffer.as_device_ptr(),
                    keystream_buffer.as_device_ptr()
                ));
                result?;
            }

            stream.add_callback(Box::new(|status| {
                println!("Device status is {:?}", status);
                match status {
                    Ok(_) => stream_done[i] = true,
                    Err(_) => panic!("error with device"),
                };
            }))?;

            key_buffers.push(key_buffer);
            keystream_buffers.push(keystream_buffer);
        }

        let mut current_iteration: u128 = (blocks * 1000000) as u128;
        while current_iteration < total_keystreams {
            for i in 0..blocks {
                if stream_done[i] {
                    let mut found_keystreams = [0u8; 8000000];
                    keystream_buffers[i].copy_to(&mut found_keystreams).unwrap();
                    let (found, idx) = rc4::check_long_u8_slice_for_keystream(& found_keystreams[..], &desired_keystream);
                    if found {
                        let mut correct_key = [0u8; 32];
                        let mut iter = key_buffers[i].chunks(key_length);
                        for _ in 0..idx {
                            iter.next().unwrap();
                        }
                        iter.next().unwrap().copy_to(&mut correct_key).unwrap();
                        let correct_key_hex = rc4::u8_to_hex(& correct_key.to_vec(), key_length as u32);
                        println!("Found the key: {correct_key_hex}");
                        return Ok(());
                    } else {
                        current_iteration += 1000000;
                        let keys = rc4::generate_keys(&mut current_key, start_from_bottom);
                        let mut key_buffer = DeviceBuffer::from_slice(&keys[..]).unwrap();
                        drop(keys);
                        let mut keystream_buffer = DeviceBuffer::from_slice(&[0u8; 8000000]).unwrap();
                        let stream = Stream::new(StreamFlags::NON_BLOCKING, None)?;
                        stream_done[i] = false;

                        unsafe {
                            let result = launch!(kernel_function<<<1, threads, 0, stream>>>(
                                key_buffer.as_device_ptr(),
                                keystream_buffer.as_device_ptr()
                            ));
                            result?;
                        }

                        stream.add_callback(Box::new(|status| {
                            println!("Device status is {:?}", status);
                            match status {
                                Ok(_) => stream_done[i] = true,
                                Err(_) => panic!("error with device"),
                            };
                        }))?;

                        key_buffers[i] = key_buffer;
                        keystream_buffers[i] = keystream_buffer;
                    }
                }
            }
        }
        println!("Could not find the key :(");
        return Ok(())
    }
}