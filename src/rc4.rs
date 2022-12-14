use std::vec::Vec;

pub struct Key {
	pub key_vec: Vec<u8>,
	pub mask_vec: Vec<u8>,
}

impl Key {
	// I know that in practice this is very computationally expensive, however I also know
	// that the maximum number of bytes in a key is 256 and same with the mask.  This means
	// that in main instead of having each thread copy the key and deal with threads being locked 
	// behind mutex the main thread can by devoted to exclusively performing deep copies for each
	// possible key.
	pub fn deep_copy(&self) -> Key {
		let mut key_vec_copy: Vec<u8> = Vec::with_capacity(self.key_vec.len());
		let mut mask_vec_copy: Vec<u8> = Vec::with_capacity(self.mask_vec.len());

		for key_val in self.key_vec.iter() {
			key_vec_copy.push(*key_val);
		}

		for mask_val in self.mask_vec.iter() {
			mask_vec_copy.push(*mask_val);
		}

		return Key {key_vec: key_vec_copy, mask_vec: mask_vec_copy};
	}
}

fn swap_chars(vec: &mut Vec<u8>, i: usize, j: usize) {
	let temp = vec[i];
	vec[i] = vec[j];
	vec[j] = temp;
}

pub fn hex_to_u8(hex: &str, length: usize) -> Vec<u8> {
	let mut u8_vec: Vec<u8> = Vec::new();

	for i in (2..length).step_by(2) {
		let upper = match hex.bytes().nth(i as usize).unwrap() {
			65..=70 => hex.bytes().nth(i as usize).unwrap() - 65 + 10,
			48..=57 => hex.bytes().nth(i as usize).unwrap() - 48,
			_ => 0,
		};

		let lower = match hex.bytes().nth((i+1) as usize).unwrap() {
			65..=70 => hex.bytes().nth((i+1) as usize).unwrap() - 65 + 10,
			48..=57 => hex.bytes().nth((i+1) as usize).unwrap() - 48,
			_ => 0,
		};

		u8_vec.push((upper << 4) | lower);
	}

	return u8_vec
}

pub fn u8_to_hex(u8_vec: &Vec<u8>, length: u32) -> String {
	let mut hex_vec: Vec<u8> = Vec::new();
	hex_vec.push(48);
	hex_vec.push(120);

	for i in 0..length {
		let upper_val: u8 = (u8_vec.get(i as usize).unwrap() >> 4) & 15;
		hex_vec.push(match upper_val {
			0..=9 => upper_val + 48,
			10..=15 => upper_val + 55,
			_ => 0,
		});

		let lower_val: u8 = u8_vec.get(i as usize).unwrap() & 15;
		hex_vec.push(match lower_val {
			0..=9 => lower_val + 48,
			10..=15 => lower_val + 55,
			_ => 0,
		});
	}

	return String::from_utf8(hex_vec).unwrap();
}

pub fn ksa(key: &Vec<u8>) -> Vec<u8> {
	let mut s: Vec<u8> = Vec::with_capacity(256);

	for i in 0..=255 {
		s.push(i as u8);
	}

	let mut j: u16 = 0;
	for i in 0..=255 {
		j = (j + u16::from(s[i]) + u16::from(key[i % key.len()])) % 256;
		swap_chars(&mut s, i, j as usize);
	}

	return s;
}

pub fn prga(s: &mut Vec<u8>, ks_length: u32, drop_n: u32) -> Vec<u8> {
	let mut ks: Vec<u8> = Vec::with_capacity(ks_length as usize);
	let mut i: u16 = 0;
	let mut j: u16 = 0;
	for _ in 0..drop_n {
		i = (i + 1) % 256;
		j = (j + u16::from(s[i as usize])) % 256;
		swap_chars(s, i as usize, j as usize);
	}

	for _ in 0..ks_length {
		i = (i + 1) % 256;
		j = (j + u16::from(s[i as usize])) % 256;
		swap_chars(s, i as usize, j as usize);
		ks.push(s[((u16::from(s[i as usize]) + u16::from(s[j as usize])) % 256) as usize]);
	}

	return ks;
}

pub fn compare_ks(calc_ks: &Vec<u8>, known_ks: &Vec<u8>) -> bool {
	if calc_ks.len() != known_ks.len() {
		return false;
	}

	for i in 0..calc_ks.len() {
		if calc_ks[i] != known_ks[i] {
			return false;
		}
	}
	return true;
}

pub fn mask_to_key(mask: &str, start_from_bottom: bool) -> Key {
	let mut key_vec: Vec<u8> = Vec::with_capacity(mask.len() - 2);
	let mut unknown_indexes: Vec<u8> = Vec::new();
	for i in (2..mask.len()).step_by(2) {
		let upper = match mask.bytes().nth(i as usize).unwrap() {
			65..=70 => mask.bytes().nth(i as usize).unwrap() - 55,
			48..=57 => mask.bytes().nth(i as usize).unwrap() - 48,
			63 => {
				unknown_indexes.push(((i-2) / 2) as u8);
				match start_from_bottom {
					true => 0,
					false => 255,
				}
			},
			_ => 0,
		};

		let lower = match mask.bytes().nth((i+1) as usize).unwrap() {
			65..=70 => mask.bytes().nth((i+1) as usize).unwrap() - 55,
			48..=57 => mask.bytes().nth((i+1) as usize).unwrap() - 48,
			63 => match start_from_bottom {
				true => 0,
				false => 255,
			},
			_ => 0,
		};

		key_vec.push((upper << 4) | lower);
	}
	return Key {key_vec: key_vec, mask_vec: unknown_indexes};
}

pub fn change_key(key: &mut Key, add: bool) {
	if add {
		for val in key.mask_vec.iter().rev() {
			if key.key_vec[*val as usize] == 255 {
				key.key_vec[*val as usize] = 0;
			} else {
				key.key_vec[*val as usize] += 1;
				break;
			}
		}
	} else {
		for val in key.mask_vec.iter().rev() {
			if key.key_vec[*val as usize] == 0 {
				key.key_vec[*val as usize] = 255;
			} else {
				key.key_vec[*val as usize] -= 1;
				break;
			}
		}
	}
}

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn test_swap_chars() {
		let vec = vec![1, 2, 3, 4, 5];
		let mut vec2 = vec![1, 2, 3, 4, 5];
		swap_chars(&mut vec2, 2, 3);
		assert_eq!(vec2[2], vec[3]);
	}

	#[test]
	fn test_hex_to_u8() {
		let hex_str: &str = "0x9805";
		let u8_vec = hex_to_u8(hex_str, hex_str.len());
		let real_vec = vec![152, 5];
		assert_eq!(u8_vec, real_vec);
	}

	#[test]
	fn test_hex_to_u8_2() {
		let hex_str: &str = "0x78AD2D347D1612CC";
		let u8_vec = hex_to_u8(hex_str, hex_str.len());

		let expected_vec = vec![120, 173, 45, 52, 125, 22, 18, 204];
		assert_eq!(u8_vec, expected_vec);
	}

	#[test]
	fn test_u8_to_hex() {
		let u8_vec = vec![152, 5];
		let hex_str = u8_to_hex(&u8_vec, u8_vec.len() as u32);
		let expected_hex_str = "0x9805";
		assert_eq!(hex_str, expected_hex_str);
	}

	#[test]
	fn test_u8_to_hex_2() {
		let u8_vec = vec![120, 173, 45, 52, 125, 22, 18, 204];
		let hex_str = u8_to_hex(&u8_vec, u8_vec.len() as u32);

		let expected_hex_str = "0x78AD2D347D1612CC";
		assert_eq!(hex_str, expected_hex_str);
	}

	#[test]
	fn test_ksa() {
		let key_vec = vec![72, 69, 76, 76, 79, 32, 87, 79, 82, 76, 68];
		let s = ksa(&key_vec);
		let expected_s = vec![6,106,229,141,126,71,41,128,81,36,83,108,
							247,80,170,110,56,16,142,45,198,31,125,217,
							20,248,99,70,23,178,97,169,159,195,42,252,
							199,85,65,133,8,191,93,72,230,236,19,105,96,
							177,58,27,34,33,0,10,200,132,245,140,232,63,
							50,95,246,94,90,174,160,201,188,78,77,7,172,
							55,76,14,116,171,104,89,48,107,4,74,167,119,
							179,15,54,231,79,59,112,212,113,138,84,11,92,
							192,145,240,37,202,150,219,194,75,114,131,214,
							44,91,25,203,204,148,102,115,207,134,139,73,
							210,28,68,86,2,67,166,222,130,182,47,17,124,
							161,156,151,208,22,64,244,209,175,29,211,1,184,
							234,206,66,53,227,82,154,21,52,163,88,39,158,
							180,87,121,120,57,129,135,176,190,137,168,162,
							101,255,253,216,187,111,250,38,18,254,173,35,
							197,213,223,237,40,100,26,9,155,221,235,136,
							127,69,12,205,189,157,32,181,43,228,238,147,
							239,164,152,183,98,185,233,224,60,225,186,117,
							51,30,109,251,220,103,122,123,218,143,215,61,
							165,3,241,118,242,24,62,13,249,46,149,226,144,
							146,193,49,153,243,196,5];
		assert_eq!(s, expected_s);
	}

	#[test]
	fn test_prga() {
		let key_vec = vec![72, 69, 76, 76, 79, 32, 87, 79, 82, 76, 68];
		let mut s = ksa(&key_vec);
		let ks_length = 8;
		let ks = prga(&mut s, ks_length, 256);

		let expected_ks = [145,108,175,105,177,75,214,80];
		let expected_s = [96,13,162,116,147,67,148,55,156,208,142,28,204,
						20,6,240,35,126,109,183,179,170,89,216,161,88,37,
						168,238,119,234,22,45,100,61,193,70,223,106,99,151,
						172,132,242,47,185,103,50,21,25,205,180,184,252,33,
						128,254,8,124,177,121,17,29,202,71,188,76,244,231,
						206,48,141,3,201,24,149,250,108,92,171,138,159,207,
						7,49,212,203,112,43,117,59,241,230,85,169,65,34,181,
						135,95,125,9,52,190,175,197,5,114,199,217,165,186,
						195,248,86,74,97,139,14,251,122,54,16,211,133,12,15,
						209,0,102,53,210,23,36,243,152,77,79,154,249,157,237,
						2,73,60,130,107,93,113,225,235,98,198,164,146,236,219,
						194,192,56,75,166,64,145,150,189,136,62,10,176,90,91,
						32,40,221,134,57,51,137,127,218,110,163,255,131,46,224,
						182,167,19,174,11,72,81,42,41,178,215,104,220,44,105,
						155,245,118,63,39,140,82,144,83,4,78,30,69,87,232,200,
						173,228,68,158,94,247,84,187,253,214,120,66,80,129,160,
						226,1,58,38,153,239,196,115,18,222,191,229,26,246,143,
						233,27,123,111,101,31,213,227];

		assert_eq!(ks, expected_ks);
		assert_eq!(s, expected_s);
	}

	#[test]
	fn test_compare_ks_equal() {
		let ks1 = vec![1, 3, 2, 244, 232, 200];
		let ks2 = vec![1, 3, 2, 244, 232, 200];
		let equal = compare_ks(&ks1, &ks2);

		let expected = true;
		assert_eq!(equal, expected);
	}

	#[test]
	fn test_compare_ks_unequal() {
		let ks1 = vec![1, 3, 2, 244, 232, 200];
		let ks2 = vec![1, 3, 2, 244, 232, 100];
		let equal = compare_ks(&ks1, &ks2);

		let expected = false;
		assert_eq!(equal, expected);
	}
	
	#[test]
	fn test_mask_to_key_bottom() {
		let mask = "0x??????????78AD2D347D1612CC";
		let key = mask_to_key(mask, true);

		let expected_key_val = vec![0, 0, 0, 0, 0, 120, 173, 45, 
								52, 125, 22, 18, 204];
		let expected_mask = vec![0, 1, 2, 3, 4];
		assert_eq!(key.key_vec, expected_key_val);
		assert_eq!(key.mask_vec, expected_mask);
	}

	#[test]
	fn test_mask_to_key_top() {
		let mask ="0x??????????78AD2D347D1612CC";
		let key = mask_to_key(mask, false);

		let expected_key_val = vec![255, 255, 255, 255, 255, 120, 173,
									45, 52, 125, 22, 18, 204];
		let expected_mask = vec![0, 1, 2, 3, 4];
		assert_eq!(key.key_vec, expected_key_val);
		assert_eq!(key.mask_vec, expected_mask);
	}

	#[test]
	fn test_change_key_add_no_overlap() {
		let mask = "0x??????????78AD2D347D1612CC";
		let mut key = mask_to_key(mask, true);
		change_key(&mut key, true);

		let expected_key_val = vec![0, 0, 0, 0, 1, 120, 173, 45,
									52, 125, 22, 18, 204];
		assert_eq!(key.key_vec, expected_key_val);
	}

	#[test]
	fn test_change_key_add_overlap() {
		let mut key = Key {key_vec: vec![0, 255, 1, 2, 3],
					   mask_vec: vec![0, 1]};
		change_key(&mut key, true);

		let expected_key_val = vec![1, 0, 1, 2, 3];
		assert_eq!(key.key_vec, expected_key_val);
	}

	#[test]
	fn test_change_key_subtract_no_overlap() {
		let mask = "0x??????????78AD2D347D1612CC";
		let mut key = mask_to_key(mask, false);
		change_key(&mut key, false);

		let expected_key_val = vec![255, 255, 255, 255, 254, 120, 173, 
									45, 52, 125, 22, 18, 204];
		assert_eq!(key.key_vec, expected_key_val);
	}

	#[test]
	fn test_change_key_subtract_overlap() {
		let mut key = Key {key_vec: vec![2, 0, 1, 2, 3],
						mask_vec: vec![0, 1]};
		change_key(&mut key, false);

		let expected_key_val = vec![1, 255, 1, 2, 3];
		assert_eq!(key.key_vec, expected_key_val);
	}

	#[test]
	fn test_change_key_add_non_consecutive_no_overlap()  {
		let mut key = Key {key_vec: vec![1, 0, 1, 0, 1],
							mask_vec: vec![0, 1, 3]};
		change_key(&mut key, true);

		let expected_key_val = vec![1, 0, 1, 1, 1];
		assert_eq!(key.key_vec, expected_key_val);
	}

	#[test]
	fn test_change_key_add_non_consecutive_overlap() {
		let mut key = Key {key_vec: vec![1, 0, 1, 255, 1],
							mask_vec: vec![0, 1, 3]};
		change_key(&mut key, true);

		let expected_key_val = vec![1, 1, 1, 0, 1];
		assert_eq!(key.key_vec, expected_key_val);
	}

	#[test]
	fn test_change_key_subtract_non_consecutive_no_overlap() {
		let mut key = Key {key_vec: vec![1, 1, 0, 1, 1],
							mask_vec: vec![0, 1, 3]};
		change_key(&mut key, false);

		let expected_key_val = vec![1, 1, 0, 0, 1];
		assert_eq!(key.key_vec, expected_key_val);
	}

	#[test]
	fn test_change_key_subtract_non_consecutive_overlap() {
		let mut key = Key {key_vec: vec![1, 1, 0, 0, 1],
							mask_vec: vec![0, 1, 3]};
		change_key(&mut key, false);

		let expected_key_val = vec![1, 0, 0, 255, 1];
		assert_eq!(key.key_vec, expected_key_val);
	}

	#[test]
	fn test_key_deep_copy() {

	}
}