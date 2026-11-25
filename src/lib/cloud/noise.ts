/**
 * Deterministic seeded 3D procedural noise library for cloud synthesis.
 * Includes Mulberry32 PRNG, 3D Value/Gradient Simplex-like noise, FBM,
 * ridged/billow noise, 3D Worley cellular noise, and curl/domain-warp helpers.
 */

export interface SeededNoise {
	readonly seed: number;
	/** Deterministic uniform random in [0, 1) */
	rand(): number;
	/** 3D smooth gradient noise in [-1, 1] */
	noise3D(x: number, y: number, z: number): number;
	/** 3D Fractal Brownian Motion in [-1, 1] */
	fbm3D(x: number, y: number, z: number, octaves?: number, lacunarity?: number, gain?: number): number;
	/** 3D billow/cauliflower FBM in [0, 1] */
	billow3D(x: number, y: number, z: number, octaves?: number): number;
	/** 3D Worley cellular distance F1 in [0, 1] (0 at cell center, 1 at edges) */
	worley3D(x: number, y: number, z: number): number;
	/** Inverted Worley puffiness (1 - F1) in [0, 1], ideal for cumulus lobes */
	worleyBillow3D(x: number, y: number, z: number): number;
	/** 3D divergence-free-like curl offset for filament and turbulence warping */
	curl3D(x: number, y: number, z: number, out?: [number, number, number]): [number, number, number];
	/** Domain-warped 3D FBM in [-1, 1] */
	warpedFbm3D(x: number, y: number, z: number, warpStrength: number, octaves?: number): number;
}

/** 12 canonical 3D gradient directions (Perlin/Simplex cube-edge vectors) */
const GRAD3: ReadonlyArray<readonly [number, number, number]> = [
	[1, 1, 0],
	[-1, 1, 0],
	[1, -1, 0],
	[-1, -1, 0],
	[1, 0, 1],
	[-1, 0, 1],
	[1, 0, -1],
	[-1, 0, -1],
	[0, 1, 1],
	[0, -1, 1],
	[0, 1, -1],
	[0, -1, -1]
];

const F3 = 1.0 / 3.0;
const G3 = 1.0 / 6.0;

/**
 * Fast Mulberry32 PRNG factory.
 */
export function createPrng(seed: number): () => number {
	let state = (seed | 0) ^ 0x6d2b79f5;
	return () => {
		state = (state + 0x6d2b79f5) | 0;
		let t = Math.imul(state ^ (state >>> 15), 1 | state);
		t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
		return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
	};
}

/**
 * Mix geographic coordinates and regionSeed into a deterministic 32-bit integer seed.
 */
export function hashRegionSeed(regionSeed: number, latitude: number, longitude: number): number {
	let h = (Math.floor(regionSeed) | 0) ^ 0x9e3779b9;
	const latQ = Math.round(latitude * 100) | 0;
	const lonQ = Math.round(longitude * 100) | 0;
	h = Math.imul(h ^ latQ, 0x85ebca6b);
	h = Math.imul((h ^ (h >>> 13)) ^ lonQ, 0xc2b2ae35);
	return (h ^ (h >>> 16)) >>> 0;
}

/**
 * Create a seeded 3D noise context with precomputed permutation tables.
 */
export function createSeededNoise(seed: number): SeededNoise {
	const rand = createPrng(seed);

	// 512-entry doubled permutation table for branchless modulo-255 indexing
	const perm = new Uint8Array(512);
	const permMod12 = new Uint8Array(512);
	const base = new Uint8Array(256);
	for (let i = 0; i < 256; i++) {
		base[i] = i;
	}
	for (let i = 255; i > 0; i--) {
		const j = Math.floor(rand() * (i + 1));
		const tmp = base[i];
		base[i] = base[j];
		base[j] = tmp;
	}
	for (let i = 0; i < 512; i++) {
		const v = base[i & 255];
		perm[i] = v;
		permMod12[i] = v % 12;
	}

	/**
	 * 3D Simplex Noise (Gustavson's formulation), returns approximately [-1, 1].
	 */
	function noise3D(xin: number, yin: number, zin: number): number {
		let n0 = 0;
		let n1 = 0;
		let n2 = 0;
		let n3 = 0;

		const s = (xin + yin + zin) * F3;
		const i = Math.floor(xin + s);
		const j = Math.floor(yin + s);
		const k = Math.floor(zin + s);
		const t = (i + j + k) * G3;
		const x0 = xin - (i - t);
		const y0 = yin - (j - t);
		const z0 = zin - (k - t);

		let i1: number, j1: number, k1: number;
		let i2: number, j2: number, k2: number;

		if (x0 >= y0) {
			if (y0 >= z0) {
				i1 = 1; j1 = 0; k1 = 0; i2 = 1; j2 = 1; k2 = 0;
			} else if (x0 >= z0) {
				i1 = 1; j1 = 0; k1 = 0; i2 = 1; j2 = 0; k2 = 1;
			} else {
				i1 = 0; j1 = 0; k1 = 1; i2 = 1; j2 = 0; k2 = 1;
			}
		} else {
			if (y0 < z0) {
				i1 = 0; j1 = 0; k1 = 1; i2 = 0; j2 = 1; k2 = 1;
			} else if (x0 < z0) {
				i1 = 0; j1 = 1; k1 = 0; i2 = 0; j2 = 1; k2 = 1;
			} else {
				i1 = 0; j1 = 1; k1 = 0; i2 = 1; j2 = 1; k2 = 0;
			}
		}

		const x1 = x0 - i1 + G3;
		const y1 = y0 - j1 + G3;
		const z1 = z0 - k1 + G3;
		const x2 = x0 - i2 + 2.0 * G3;
		const y2 = y0 - j2 + 2.0 * G3;
		const z2 = z0 - k2 + 2.0 * G3;
		const x3 = x0 - 1.0 + 3.0 * G3;
		const y3 = y0 - 1.0 + 3.0 * G3;
		const z3 = z0 - 1.0 + 3.0 * G3;

		const ii = i & 255;
		const jj = j & 255;
		const kk = k & 255;

		let t0 = 0.6 - x0 * x0 - y0 * y0 - z0 * z0;
		if (t0 > 0) {
			const gi0 = GRAD3[permMod12[ii + perm[jj + perm[kk]]]];
			t0 *= t0;
			n0 = t0 * t0 * (gi0[0] * x0 + gi0[1] * y0 + gi0[2] * z0);
		}

		let t1 = 0.6 - x1 * x1 - y1 * y1 - z1 * z1;
		if (t1 > 0) {
			const gi1 = GRAD3[permMod12[ii + i1 + perm[jj + j1 + perm[kk + k1]]]];
			t1 *= t1;
			n1 = t1 * t1 * (gi1[0] * x1 + gi1[1] * y1 + gi1[2] * z1);
		}

		let t2 = 0.6 - x2 * x2 - y2 * y2 - z2 * z2;
		if (t2 > 0) {
			const gi2 = GRAD3[permMod12[ii + i2 + perm[jj + j2 + perm[kk + k2]]]];
			t2 *= t2;
			n2 = t2 * t2 * (gi2[0] * x2 + gi2[1] * y2 + gi2[2] * z2);
		}

		let t3 = 0.6 - x3 * x3 - y3 * y3 - z3 * z3;
		if (t3 > 0) {
			const gi3 = GRAD3[permMod12[ii + 1 + perm[jj + 1 + perm[kk + 1]]]];
			t3 *= t3;
			n3 = t3 * t3 * (gi3[0] * x3 + gi3[1] * y3 + gi3[2] * z3);
		}

		return 32.0 * (n0 + n1 + n2 + n3);
	}

	function fbm3D(
		x: number,
		y: number,
		z: number,
		octaves = 4,
		lacunarity = 2.02,
		gain = 0.5
	): number {
		let sum = 0;
		let amp = 0.5;
		let freq = 1.0;
		let norm = 0;
		for (let o = 0; o < octaves; o++) {
			sum += amp * noise3D(x * freq, y * freq, z * freq);
			norm += amp;
			freq *= lacunarity;
			amp *= gain;
		}
		return norm > 0 ? sum / norm : 0;
	}

	function billow3D(x: number, y: number, z: number, octaves = 3): number {
		let sum = 0;
		let amp = 0.5;
		let freq = 1.0;
		let norm = 0;
		for (let o = 0; o < octaves; o++) {
			const v = 1.0 - Math.abs(noise3D(x * freq, y * freq, z * freq));
			sum += amp * v * v;
			norm += amp;
			freq *= 2.05;
			amp *= 0.5;
		}
		return norm > 0 ? sum / norm : 0;
	}

	/**
	 * Fast integer-hashed 3D Worley cellular noise (3x3x3 neighborhood).
	 * Returns F1 distance normalized approximately to [0, 1].
	 */
	function worley3D(x: number, y: number, z: number): number {
		const ix = Math.floor(x);
		const iy = Math.floor(y);
		const iz = Math.floor(z);
		const fx = x - ix;
		const fy = y - iy;
		const fz = z - iz;

		let minDistSq = 9.0;

		for (let dz = -1; dz <= 1; dz++) {
			const cz = (iz + dz) & 255;
			for (let dy = -1; dy <= 1; dy++) {
				const cy = (iy + dy) & 255;
				const pyz = perm[cy + perm[cz]];
				for (let dx = -1; dx <= 1; dx++) {
					const cx = (ix + dx) & 255;
					const h = perm[cx + pyz];
					const hx = perm[h] / 255.0;
					const hy = perm[h + 37] / 255.0;
					const hz = perm[h + 73] / 255.0;

					const rx = dx + hx - fx;
					const ry = dy + hy - fy;
					const rz = dz + hz - fz;
					const dSq = rx * rx + ry * ry + rz * rz;
					if (dSq < minDistSq) {
						minDistSq = dSq;
					}
				}
			}
		}

		return Math.min(1.0, Math.sqrt(minDistSq));
	}

	function worleyBillow3D(x: number, y: number, z: number): number {
		const d = worley3D(x, y, z);
		const inv = 1.0 - d;
		return inv * inv;
	}

	function curl3D(
		x: number,
		y: number,
		z: number,
		out: [number, number, number] = [0, 0, 0]
	): [number, number, number] {
		const eps = 0.08;
		const inv2Eps = 0.5 / eps;

		// Use three decorrelated scalar potential fields via fixed offsets
		const dPz_dy =
			(noise3D(x + 31.4, y + eps, z + 17.2) - noise3D(x + 31.4, y - eps, z + 17.2)) * inv2Eps;
		const dPy_dz =
			(noise3D(x + 11.7, y + 47.1, z + eps) - noise3D(x + 11.7, y + 47.1, z - eps)) * inv2Eps;

		const dPx_dz =
			(noise3D(x, y, z + eps) - noise3D(x, y, z - eps)) * inv2Eps;
		const dPz_dx =
			(noise3D(x + 31.4 + eps, y, z + 17.2) - noise3D(x + 31.4 - eps, y, z + 17.2)) * inv2Eps;

		const dPy_dx =
			(noise3D(x + 11.7 + eps, y + 47.1, z) - noise3D(x + 11.7 - eps, y + 47.1, z)) * inv2Eps;
		const dPx_dy =
			(noise3D(x, y + eps, z) - noise3D(x, y - eps, z)) * inv2Eps;

		out[0] = dPz_dy - dPy_dz;
		out[1] = dPx_dz - dPz_dx;
		out[2] = dPy_dx - dPx_dy;
		return out;
	}

	function warpedFbm3D(
		x: number,
		y: number,
		z: number,
		warpStrength: number,
		octaves = 3
	): number {
		const wx = noise3D(x + 5.2, y + 1.3, z + 8.7) * warpStrength;
		const wy = noise3D(x + 19.1, y + 7.4, z + 3.2) * warpStrength;
		const wz = noise3D(x + 2.8, y + 14.9, z + 23.6) * warpStrength;
		return fbm3D(x + wx, y + wy, z + wz, octaves);
	}

	return {
		seed,
		rand,
		noise3D,
		fbm3D,
		billow3D,
		worley3D,
		worleyBillow3D,
		curl3D,
		warpedFbm3D
	};
}

export function clamp(v: number, min: number, max: number): number {
	return v < min ? min : v > max ? max : v;
}

export function smoothstep(edge0: number, edge1: number, x: number): number {
	const t = clamp((x - edge0) / (edge1 - edge0), 0, 1);
	return t * t * (3 - 2 * t);
}

export function lerp(a: number, b: number, t: number): number {
	return a + (b - a) * t;
}
