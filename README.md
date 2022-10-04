# rc4-plaintext
Rust program to bruteforce rc4 hidden key given a known key mask to produce a known keystream.

## Build
run the following command from the root directory (CPU ONLY):
```
cargo build --release
```
#### Experimental
To experimentally build the program utilizing the gpu run the following command from the root directory
```
cargo build --release --features gpu_enabled
```

## Run
To run the program execute the following in the root directory:
```
./target/release/rc4_pt_attack "{known_key_mask}" "{target_plaintext}" {start_from_0: bool} --drop_n={drop_n} --{Optional: gpu|cpu} --threads={Optional: number_of_thread} --blocks={Optional (GPU_ONLY): blocks}
```

## CLI Arguments
1. known_key_mask - hex string with ?? as unknown bits - (?s must be on character boundaries)
2. target_keystream - hex string of ct xor pt (to run on GPU this must be 8 bytes or 16 hex digits)
3. start_from_0 - true => start from 0 and work up; false => start from 255 and work down to find key
4. (Optional) drop_n - the number of bytes to drop before generating a keystream (defaults to 0)
5. (Optional) --cpu or --gpu flag (if your device has compiled the program with gpu support the script can be executed with gpu using --gpu)
5. (Optional) threads - the number of threads in the threadpool to conduct calculations (defaults to 1) - (FOR GPU SPECIFIC THIS IS THE NUMBER OF THREADS PER BLOCK)
6. (Optional (GPU_ONLY)) blocks - the number of gpu blocks to compute on

## Examples
To run a plaintext attack against a 72-bit key of 0x37415869??06??9D?? with a drop_n of 256 bytes and a desired keystream of 0x582B4552EBD97C run the following:
```
./target/release/rc4_pt_attack "0x37415869??06??9D??" "0x582B4552EBD97C" true
```
To run a plaintext attack against a 104-bit key of 0x??????????5560F28DC480C689 with a drop_n of 256 bytes and a desired keystream of 0xD88125F83E6962 starting from 255 and working down using 32 threads
```
./target/release/rc4_pt_attack "0x??????????5560F28DC480C689" "0xD88125F83E6962" false --drop_n=256 -=cpu --theads=32
```
To run a plaintext attack against a 104-bit key of 0x??76??32??78AD2D347D16??CC with a drop_n of 267 bytes and a desired keystream of 0xC5C593CB6AC6F419A3441A starting from 255 and working down using 16 threads
```
./target/release/rc4_pt_attack "0x??76??32??78AD2D347D16??CC" "0xC5C593CB6AC6F419A3441A" false --drop_n=267 --cpu --threads=16
```
To run a plaintext attack against a 104-bit key of 0x??76??32??78AD2D348D16??CC with a drop_n of 267 bytes and a desired keystream of 0xC5C593CB6AC6F419 starting from 255 and working down using 16 theads on 8 blocks of a gpu.
```
./target/release/rc4_pt_attack "0x??76??32??78AD2D347D16??CC" "0xC5C593CB6AC6F419" false --drop_n=267 --gpu --threads=16 --blocks=8
```
