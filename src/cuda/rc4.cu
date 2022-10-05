#include <stdint.h>

typedef uint8_t u8;
typedef uint16_t u16;
typedef uint32_t u32;
typedef uint64_t u64;

typedef int8_t i8;
typedef int16_t i16;
typedef int32_t i32;
typedef int64_t i64;

// CUDA kernel to generate first 8 bytes of a keystream for a 72 bit key and no drop_n.
extern "C" __global__ void rc4_keystream_gen_72_drop_0(const u8 *key, u8 *out_keystream) {
	u64 write_idx = 0;
	for (u64 iter = 0; i < 1000000; i++) {
		u8 s[256];
		u8 temp;
		for (u16 i = 0; i < 256; i++) {
			s[i] = i;
		}

		u8 j = 0;
		for (u16 i = 0; i < 256; i++) {
			j += s[i] + key[i % 72];

			// inline swap characters
			temp = s[i];
			s[i] = s[j];
			s[j] = temp;
		}

		u8 i = 0;
		j = 0;
		for (u8 idx = 0; idx < 8; idx++) {
			i += 1;
			j += s[i];

			// inline swap characters
			temp = s[i];
			s[i] = s[j];
			s[j] = temp;

			out_keystream[write_idx] = s[(s[i] + s[j]) % 256];
			write_idx++;
		}
	}
}

// CUDA kernel to generate the first 8 bytes of a keystream for a 72 bit key and 256 byte drop_n.
extern "C" __global__ void rc4_keystream_gen_72_drop_256(const u8 *key, u8 *out_keystream) {
	u64 write_idx = 0;
	for (u64 iter = 0; iter < 1000000; iter++) {
		u8 s[256];
		u8 temp;
		for (u16 i = 0; i < 256; i++) {
			s[i] = i;
		}

		u8 j = 0;
		for (u16 i = 0; i < 256; i++) {
			j += s[i] + key[i % 72];

			// inline swap characters
			temp = s[i];
			s[i] = s[j];
			s[j] = temp;
		}

		u8 i = 0;
		j = 0;
		for (u16 idx = 0; idx < 256; idx++) {
			i += 1;
			j += s[i];

			// inline swap characters
			temp = s[i];
			s[i] = s[j];
			s[j] = temp;
		}

		i, j = 0;
		for (u8 idx = 0; idx < 8; idx++) {
			i += 1;
			j += s[i];

			// inline swap characters
			temp = s[i];
			s[i] = s[j];
			s[j] = temp;

			out_keystream[write_idx] = s[(s[i] + s[j]) % 256];
			write_idx++;
		}
	}
}

// CUDA kernel to generate the first 8 bytes of a keystream for a 72 bit key and 267 byte drop_n.
extern "C" __global__ void rc4_keystream_gen_72_drop_267(const u8 *key, u8 *out_keystream) {
	u64 write_idx = 0;
	for (u64 iter = 0; iter < 1000000; iter++) {
		u8 s[256];
		u8 temp;
		for (u16 i = 0; i < 256; i++) {
			s[i] = i;
		}

		u8 j = 0;
		for (u16 i = 0; i < 256; i++) {
			j += s[i] + key[i % 72];

			// inline swap characters
			temp = s[i];
			s[i] = s[j];
			s[j] = temp;
		}

		u8 i = 0;
		j = 0;
		for (u16 idx = 0; idx < 267; idx++) {
			i += 1;
			j += s[i];

			// inline swap characters
			temp = s[i];
			s[i] = s[j];
			s[j] = temp;
		}

		i, j = 0;
		for (u8 idx = 0; idx < 8; idx++) {
			i += 1;
			j += s[i];

			// inline swap characters
			temp = s[i];
			s[i] = s[j];
			s[j] = temp;

			out_keystream[write_idx] = s[(s[i] + s[j]) % 256];
			write_idx++;
		}
	}
}

// CUDA kernel to generate the first 8 bytes of a keystream for a 104 bit key and 0 byte drop_n.
extern "C" __global__ void rc4_keystream_gen_104_drop_0(const u8 *key, u8 *out_keystream) {
	u64 write_idx = 0;
	for (u64 iter = 0; iter < 1000000; iter++) {
		u8 s[256];
		u8 temp;
		for (u16 i = 0; i < 256; i++) {
			s[i] = i;
		}

		u8 j = 0;
		for (u16 i = 0; i < 256; i++) {
			j += s[i] + key[i % 104];

			// inline swap characters
			temp = s[i];
			s[i] = s[j];
			s[j] = temp;
		}

		u8 i = 0;
		j = 0;
		for (u8 idx = 0; idx < 8; idx++) {
			i += 1;
			j += s[i];

			// inline swap characters
			temp = s[i];
			s[i] = s[j];
			s[j] = temp;

			out_keystream[write_idx] = s[(s[i] + s[j]) % 256];
			write_idx++;
		}
	}
}

// CUDA kernel to generate the first 8 bytes of a keystream for a 104 bit key and 256 byte drop_n.
extern "C" __global__ void rc4_keystream_gen_104_drop_256(const u8 *key, u8 *out_keystream) {
	u64 write_idx = 0;
	for (u64 iter = 0; iter < 1000000; iter++) {
		u8 s[256];
		u8 temp;
		for (u16 i = 0; i < 256; i++) {
			s[i] = i;
		}

		u8 j = 0;
		for (u16 i = 0; i < 256; i++) {
			j += s[i] + key[i % 104];

			// inline swap characters
			temp = s[i];
			s[i] = s[j];
			s[j] = temp;
		}

		u8 i = 0;
		j = 0;
		for (u16 idx = 0; idx < 256; idx++) {
			i += 1;
			j += s[i];

			// inline swap characters
			temp = s[i];
			s[i] = s[j];
			s[j] = temp;
		}

		i, j = 0;
		for (u8 idx = 0; idx < 8; idx++) {
			i += 1;
			j += s[i];

			// inline swap characters
			temp = s[i];
			s[i] = s[j];
			s[j] = temp;

			out_keystream[write_idx] = s[(s[i] + s[j]) % 256];
			write_idx++;
		}
	}
}

// CUDA kernel to generate the first 8 bytes of a keystream for a 104 bit key and 267 byte drop_n.
extern "C" __global__ void rc4_keystream_gen_104_drop_267(const u8 *key, u8 *out_keystream) {
	u64 write_idx = 0;
	for (u64 iter = 0; iter < 1000000; iter++) {
		u8 s[256];
		u8 temp;
		for (u16 i = 0; i < 256; i++) {
			s[i] = i;
		}

		u8 j = 0;
		for (u16 i = 0; i < 256; i++) {
			j += s[i] + key[i % 104];

			// inline swap characters
			temp = s[i];
			s[i] = s[j];
			s[j] = temp;
		}

		u8 i = 0;
		j = 0;
		for (u16 idx = 0; idx < 267; idx++) {
			i += 1;
			j += s[i];

			// inline swap characters
			temp = s[i];
			s[i] = s[j];
			s[j] = temp;
		}

		i, j = 0;
		for (u8 idx = 0; idx < 8; idx++) {
			i += 1;
			j += s[i];

			// inline swap characters
			temp = s[i];
			s[i] = s[j];
			s[j] = temp;

			out_keystream[write_idx] = s[(s[i] + s[j]) % 256];
			write_idx++;
		}
	}
}

// CUDA kernel to generate first 8 bytes of a keystream for a 64 bit key and no drop_n.
extern "C" __global__ void rc4_keystream_gen_64_drop_0(const u8 *key, u8 *out_keystream) {
	u64 write_idx = 0;
	for (u64 iter = 0; i < 1000000; i++) {
		u8 s[256];
		u8 temp;
		for (u16 i = 0; i < 256; i++) {
			s[i] = i;
		}

		u8 j = 0;
		for (u16 i = 0; i < 256; i++) {
			j += s[i] + key[i % 64];

			// inline swap characters
			temp = s[i];
			s[i] = s[j];
			s[j] = temp;
		}

		u8 i = 0;
		j = 0;
		for (u8 idx = 0; idx < 8; idx++) {
			i += 1;
			j += s[j];

			// inline swap characters
			temp = s[i];
			s[i] = s[j];
			s[j] = temp;

			out_keystream[write_idx] = s[(s[i] + s[j]) % 256];
			write_idx++;
		}
	}
}

exter "C" __global void rc4_keystream_gen_64_drop_256(const u8 *key, u8 *out_keystream) {
	u64 write_idx = 0;
	for (u64 iter = 0; iter < 1000000; i++) {
		u8 s[256];
		u8 temp;
		for (u16 i = 0; i < 256; i++) {
			s[i] = i;
		}

		u8 j = 0;
		for (u16 i = 0; i < 256; i++) {
			j += s[i] + key[i % 64];

			// inline swap characters
			temp = s[i];
			s[i] = s[j];
			s[j] = temp;
		}

		u8 i = 0;
		j = 0;
		for (u16 idx = 0; idx < 256; idx++) {
			i += 1;
			j += s[i];

			// inline swap characters
			temp = s[i];
			s[i] = s[j];
			s[j] = temp;
		}

		i, j = 0;
		for (u8 idx = 0; idx < 8; idx++) {
			i += 1;
			j += s[i];

			// inline swap characters
			temp = s[i];
			s[i] = s[j];
			s[j] = temp;

			out_keystream[write_idx] = s[(s[i] + s[j]) % 256];
			write_idx++;
		}
	}
}

extern "C" __global__ void rc4_keystream_gen_64_drop_267(const u8 *key, u8 *out_keystream) {
	u64 write_idx = 0;
	for (u64 iter = 0; iter < 1000000; iter++) {
		u8 s[256];
		u8 temp;
		for (u16 i = 0; i < 256; i++) {
			s[i] = i;
		}

		u8 j = 0;
		for (u16 i = 0; i < 256; i++) {
			j += s[i] + key[i % 64];

			// inline swap characters
			temp = s[i];
			s[i] = s[j];
			s[j] = temp;
		}

		u8 i = 0;
		j = 0;
		for (u16 idx = 0; idx < 267; idx++) {
			i += 1;
			j += s[i];

			// inline swap characters
			temp = s[i];
			s[i] = s[j];
			s[j] = temp;
		}

		i, j = 0;
		for (u8 idx = 0; idx < 8; idx++) {
			i += 1;
			j += s[i];

			// inline swap characters
			temp = s[i];
			s[i] = s[j];
			s[j] = temp;

			out_keystream[write_idx] = s[(s[i] + s[j]) % 256];
			write_idx++;
		}
	}
}

extern "C" __global__ void rc4_keystream_gen_128_drop_0(const u8 *key, u8 *out_keystream) {
	u64 write_idx = 0;
	for (u64 iter = 0; i < 1000000; i++) {
		u8 s[256];
		u8 temp;
		for (u16 i = 0; i < 256; i++) {
			s[i] = i;
		}

		u8 j = 0;
		for (u16 i = 0; i < 256; i++) {
			j += s[i] + key[i % 128];

			// inline swap characters
			temp = s[i];
			s[i] = s[j];
			s[j] = temp;
		}

		u8 i = 0;
		j = 0;
		for (u8 idx = 0; idx < 8; idx++) {
			i += 1;
			j += s[i];

			// inline swap characters
			temp = s[i];
			s[i] = s[j];
			s[j] = temp;

			out_keystream[write_idx] = s[(s[i] + s[j]) % 256];
			write_idx++;
		}
	}
}

extern "C" __global__ void rc4_keystream_gen_128_drop_256(const u8 *key, u8 *out_keystream) {
	u64 write_idx = 0;
	for (u64 iter - 0; iter < 1000000; iter++) {
		u8 s[256];
		u8 temp;
		for (u16 i = 0; i < 256; i++) {
			s[i] = i;
		}

		u8 j = 0;
		for (u16 i = 0; i < 256; i++) {
			j += s[i] + key[i % 128];

			// inline swap characters
			temp = s[i];
			s[i] = s[j];
			s[j] = temp;
		}

		u8 i = 0;
		j = 0;
		for (u16 idx = 0; idx < 256; idx++) {
			i += 1;
			j += s[i];

			// inline swap characters
			temp = s[i];
			s[i] = s[j];
			s[j] = temp;
		}

		i, j = 0;
		for (u8 idx = 0; idx < 8; idx++) {
			i += 1;
			j += s[i];

			// inline swap characters
			temp = s[i];
			s[i] = s[j];
			s[j] = temp;

			out_keystream[write_idx] = s[(s[i] + s[j]) % 256];
			write_idx++;
		}
	}
}

extern "C" __global__ rc4_keystream_gen_128_drop_267(const u8 *key, u8 *out_keystream) {
	u64 write_idx = 0;
	for (u64 iter = 0; iter < 1000000; iter++) {
		u8 s[256];
		u8 temp;
		for (u16 i = 0; i < 256; i++) {
			s[i] = i;
		}

		u8 j = 0;
		for (u16 i = 0; i < 256; i++) {
			j += s[i] + key[i % 128];

			// inline swap characters
			temp s[i];
			s[i] = s[j];
			s[j] = temp;
		}

		u8 i = 0;
		j = 0;
		for (u16 idx = 0; idx < 256; idx++) {
			j += 1;
			j += s[i];

			// inline swap charaacters
			temp = s[i];
			s[i] = s[j];
			s[j] = temp;
		}

		i, j = 0;
		for (u8 idx = 0; idx < 8; idx++) {
			i += 1;
			j += s[i];

			// inline swap characters
			temp s[i];
			s[i] = s[j];
			s[j] = temp;

			out_keystream[write_idx] = s[(s[i] + s[j]) % 256];
			write_idx++;
		}
	}
}