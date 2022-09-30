mod rc4;

use threadpool::ThreadPool;
use std::{
	env,
	time::{
        Instant,
        Duration,
    },
	sync::{
		Arc,
		Mutex,
		mpsc::channel
	},
    thread,
};

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
/// panics when unacceptable number of arguments are provided
/// 
/// if no string is found the found key will read: "KEY WAS NOT FOUND"
fn main() {
	let args: Vec<_> = env::args().collect();

	let (drop_n, thread_num) = match args.len() {
		4 => (0, 1),
		5 => (args.get(4).unwrap().parse().unwrap(), 1),
		6 => (args.get(4).unwrap().parse().unwrap(), args.get(5).unwrap().parse().unwrap()),
		_ => panic!("unknown arguments passed into main"),
	};
	let start_from_bottom = match &args[3][..] {
		"true" | "T" | "True" | "1" => true,
		"false" | "F" | "False" | "0" => false,
        _ => panic!("unknown start_from_bottom argument"),
	};

	let now = Instant::now();

	let mut key = rc4::mask_to_key(args.get(1).unwrap(), start_from_bottom);
	let desired_keystream = rc4::hex_to_u8(args.get(2).unwrap(), args.get(2).unwrap().len());
    let max_iterations: u128 = 256u128.pow((key.mask_vec.len()) as u32);
    let desired_keystream_length = desired_keystream.len();

	// Multithreading stuff //
	
	// ThreadPool
	// TODO: add command line argument to set number of workers
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
            if i % 1000000000 == 0 {
                println!("iteration: {i}");
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
		rc4::change_key(&mut key, start_from_bottom);

        let received_val = rx.try_recv();
        match received_val {
            Ok(desired_key_hex) => {
                println!("found key: {desired_key_hex}");
                let elapsed = now.elapsed();
                println!("exeuction took: {:.2?}", elapsed);
                return;
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
                return;
            },
            _ => {
                thread::sleep(Duration::from_secs(1));
            },
        }
    }
}