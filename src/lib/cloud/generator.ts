/**
 * Procedural 3D volumetric cloud point-field generator.
 * Synthesizes up to 150,000 points across 5 distinct meteorological cloud morphologies
 * with baked 3D raymarched self-shadowing and a 2D column optical-depth map for
 * splatted dot-gradient ground shadows.
 */

import type { CloudParams } from '$lib/types';
import type { MeteorologyState } from './meteorology';
import { clamp, createSeededNoise, hashRegionSeed, lerp, smoothstep, type SeededNoise } from './noise';

export interface CloudVolumeBounds {
	halfExtentXZ: number;
	baseY: number;
	heightY: number;
	centerY: number;
}

export interface GeneratedCloudData {
	positions: Float32Array;
	aDensity: Float32Array;
	aHeightNorm: Float32Array;
	aSeed: Float32Array;
	aSize: Float32Array;
	aShade: Float32Array;
	activePoints: number;
	volumeAtlasData: Uint8Array;
	volumeAtlasWidth: number;
	volumeAtlasHeight: number;
	bounds: CloudVolumeBounds;
}

export const VOLUME_GRID_X = 52;
export const VOLUME_GRID_Y = 28;
export const VOLUME_GRID_Z = 52;
export const VOLUME_ATLAS_COLS = 7;
export const VOLUME_ATLAS_ROWS = 4; // 7 * 4 = 28 === VOLUME_GRID_Y
export const VOLUME_ATLAS_WIDTH = VOLUME_GRID_X * VOLUME_ATLAS_COLS; // 364
export const VOLUME_ATLAS_HEIGHT = VOLUME_GRID_Z * VOLUME_ATLAS_ROWS; // 208

const GRID_X = VOLUME_GRID_X;
const GRID_Y = VOLUME_GRID_Y;
const GRID_Z = VOLUME_GRID_Z;
const HALF_EXTENT_XZ = 12.5;

interface ConvectiveCell {
	x: number;
	z: number;
	radius: number;
	height: number;
}

/**
 * Compute world-space vertical placement and thickness from meteorological state.
 */
function computeVolumeBounds(params: CloudParams, meteo: MeteorologyState): CloudVolumeBounds {
	const w = meteo.weights;
	// Map base altitude (0.35..11.2 km) into a visually well-framed world Y range (2.1..7.6)
	const altNorm = clamp((meteo.baseAltitudeKm - 0.35) / 10.5, 0, 1);
	const baseY = lerp(2.1, 7.6, Math.pow(altNorm, 0.85));

	const hCumulus = 4.2 + params.turbulence * 1.5;
	const hStratus = 1.35 + params.humidity * 0.45;
	const hCirrus = 1.55 + params.turbulence * 0.55;
	const hCb = 9.0 + params.turbulence * 2.0;
	const hAlto = 1.85 + params.turbulence * 0.55;

	const heightY =
		w.cumulus * hCumulus +
		w.stratus * hStratus +
		w.cirrus * hCirrus +
		w.cumulonimbus * hCb +
		w.altocumulus * hAlto;

	return {
		halfExtentXZ: HALF_EXTENT_XZ,
		baseY,
		heightY,
		centerY: baseY + heightY * 0.48
	};
}

/**
 * Generate seeded convective dome centers for cumulus & cumulonimbus cells.
 */
function createConvectiveCells(noise: SeededNoise, coverage: number): ConvectiveCell[] {
	const count = Math.max(2, Math.round(2 + coverage * 4.5));
	const cells: ConvectiveCell[] = [];
	// Anchor one strong primary formation near center for balanced framing
	cells.push({
		x: (noise.rand() - 0.5) * 0.28,
		z: (noise.rand() - 0.5) * 0.28,
		radius: 0.36 + coverage * 0.14,
		height: 0.88 + noise.rand() * 0.12
	});
	for (let i = 1; i < count; i++) {
		const angle = (i / (count - 1)) * Math.PI * 2 + (noise.rand() - 0.5) * 0.9;
		const dist = 0.24 + noise.rand() * 0.38;
		cells.push({
			x: Math.cos(angle) * dist,
			z: Math.sin(angle) * dist,
			radius: 0.22 + noise.rand() * 0.22 + coverage * 0.08,
			height: 0.52 + noise.rand() * 0.42
		});
	}
	return cells;
}

/**
 * Evaluate cumulus: flat condensation base + buoyant cauliflower billow domes.
 */
function evalCumulus(
	nx: number,
	ny: number,
	nz: number,
	noise: SeededNoise,
	cells: ConvectiveCell[],
	coverage: number,
	turbulence: number
): number {
	// Crisp flat condensation base
	const baseProfile = smoothstep(0.02, 0.13, ny);
	if (baseProfile <= 0) return 0;

	let domeField = 0;
	for (let i = 0; i < cells.length; i++) {
		const c = cells[i];
		const dx = nx - c.x;
		const dz = nz - c.z;
		// Slightly narrow domes toward their tops for buoyant tower silhouettes
		const rScale = c.radius * (1.08 - ny * 0.48);
		const r2 = (dx * dx + dz * dz) / Math.max(rScale * rScale, 0.01);
		const hNorm = ny / c.height;
		if (r2 < 1.35 && hNorm < 1.1) {
			// Flat-bottomed hemispherical/dome envelope
			const dome = 1.0 - r2 * 0.82 - Math.pow(Math.max(0, hNorm - 0.12), 1.85) * 0.95;
			if (dome > domeField) domeField = dome;
		}
	}

	if (domeField <= -0.25) return 0;

	// Cauliflower puffs via inverted Worley + billow FBM
	const worleyPuff = noise.worleyBillow3D(nx * 3.1 + 1.7, ny * 2.8, nz * 3.1 - 2.3);
	const detail = noise.fbm3D(nx * 2.6, ny * 2.4, nz * 2.6, 3);
	const puffStrength = 0.34 + turbulence * 0.26;
	const raw = domeField * baseProfile + (worleyPuff - 0.38) * puffStrength + detail * 0.22;

	const threshold = lerp(0.32, 0.08, coverage);
	return clamp((raw - threshold) * 1.9, 0, 1);
}

/**
 * Evaluate stratus: wide, low, softly undulating horizontal blanket/sheet.
 */
function evalStratus(
	nx: number,
	ny: number,
	nz: number,
	noise: SeededNoise,
	coverage: number,
	humidity: number,
	turbulence: number
): number {
	const r = Math.hypot(nx, nz);
	const radialMask = 1.0 - smoothstep(0.74, 1.0, r);
	if (radialMask <= 0) return 0;

	// Gentle horizontal undulation
	const undulation = noise.fbm3D(nx * 1.35 + 4.1, 0.25, nz * 1.35 - 3.2, 3);
	const centerY = 0.44 + undulation * (0.12 + turbulence * 0.12);
	const halfThick = 0.22 + humidity * 0.18;
	const verticalDist = Math.abs(ny - centerY) / halfThick;
	const vertMask = 1.0 - smoothstep(0.35, 1.05, verticalDist);
	if (vertMask <= 0) return 0;

	const sheetNoise = noise.fbm3D(nx * 2.1, ny * 1.6, nz * 2.1, 3);
	const threshold = lerp(0.38, -0.35, coverage);
	const density = (sheetNoise + 0.55 - threshold) * vertMask * radialMask;
	return clamp(density * 1.35, 0, 1);
}

/**
 * Evaluate cirrus: high-altitude delicate windswept ice-crystal wisps & filaments.
 */
function evalCirrus(
	along: number,
	ny: number,
	across: number,
	noise: SeededNoise,
	coverage: number,
	turbulence: number
): number {
	const r = Math.hypot(along * 0.85, across * 1.05);
	const radialMask = 1.0 - smoothstep(0.66, 1.0, r);
	if (radialMask <= 0) return 0;

	// Strong anisotropic stretching along wind direction + graceful mares'-tail curl warp
	const warp = noise.fbm3D(along * 0.85 + 7.2, ny * 1.4, across * 2.4 - 5.1, 2);
	const hookY = 0.5 + Math.sin(along * 2.4 + warp * 2.2) * 0.22 + warp * 0.12;
	const verticalBand = 1.0 - smoothstep(0.08, 0.34, Math.abs(ny - hookY));
	if (verticalBand <= 0) return 0;

	// Ridged windswept filaments
	const sx = along * 0.72 + warp * (0.45 + turbulence * 0.35);
	const sz = across * 3.8 + warp * (0.75 + turbulence * 0.55);
	const ridge1 = 1.0 - Math.abs(noise.fbm3D(sx, ny * 1.2, sz, 3));
	const ridge2 = 1.0 - Math.abs(noise.noise3D(sx * 1.7 + 12.3, ny * 2.0, sz * 1.6 - 8.4));
	const filament = Math.pow(clamp(ridge1 * 0.65 + ridge2 * 0.35, 0, 1), 2.35);

	const threshold = lerp(0.48, 0.16, coverage);
	const density = (filament - threshold) * 2.4 * verticalBand * radialMask;
	return clamp(density, 0, 1);
}

/**
 * Evaluate cumulonimbus: massive towering vertical column spreading into a sheared flat anvil top.
 */
function evalCumulonimbus(
	along: number,
	ny: number,
	across: number,
	noise: SeededNoise,
	coverage: number,
	turbulence: number,
	windSpeed: number
): number {
	// Shear tilts the upper tower and stretches the anvil downwind
	const shearFactor = 0.14 + clamp(windSpeed / 40, 0, 1) * 0.24;
	const baseRamp = smoothstep(0.02, 0.09, ny);
	if (baseRamp <= 0) return 0;

	// 1. Broad flat-bottomed inflow base (ny: 0.03 .. 0.28)
	const baseR = Math.hypot(along * 1.1, across * 1.1);
	const baseEnvelope =
		(1.0 - smoothstep(0.32, 0.62, baseR)) *
		baseRamp *
		(1.0 - smoothstep(0.18, 0.34, ny));

	// 2. Towering convective updraft column (ny: 0.08 .. 0.84)
	const towerCenterAlong = ny * shearFactor * 0.75;
	const towerR = Math.hypot((along - towerCenterAlong) * 1.45, across * 1.45);
	const towerRadius = lerp(0.46, 0.34, smoothstep(0.15, 0.65, ny));
	const towerEnvelope =
		(1.0 - smoothstep(towerRadius * 0.45, towerRadius, towerR)) *
		smoothstep(0.05, 0.15, ny) *
		(1.0 - smoothstep(0.76, 0.94, ny));

	// 3. Sheared flat anvil top / incus (ny: 0.66 .. 0.94)
	const anvilAlongOffset = 0.18 + shearFactor * 0.85;
	const anvilAlongNorm = (along - anvilAlongOffset) / 0.78;
	const anvilAcrossNorm = across / 0.62;
	const anvilR = Math.hypot(anvilAlongNorm, anvilAcrossNorm);
	const anvilVert = smoothstep(0.62, 0.75, ny) * (1.0 - smoothstep(0.84, 0.94, ny));
	const anvilEnvelope = (1.0 - smoothstep(0.48, 1.0, anvilR)) * anvilVert * 1.15;

	// 4. Overshooting top dome poking above the tropopause cap
	const domeR = Math.hypot((along - towerCenterAlong) * 2.4, across * 2.4);
	const overshootingDome =
		(1.0 - smoothstep(0.25, 0.58, domeR)) *
		smoothstep(0.74, 0.86, ny) *
		(1.0 - smoothstep(0.92, 1.0, ny));

	const envelope = Math.max(baseEnvelope, towerEnvelope, anvilEnvelope, overshootingDome);
	if (envelope <= 0.02) return 0;

	const worley = noise.worleyBillow3D(along * 3.0, ny * 3.6, across * 3.0);
	const turb = noise.fbm3D(along * 2.4, ny * 3.2, across * 2.4, 3);
	const billowAmt = 0.25 + turbulence * 0.28;
	const threshold = lerp(0.26, 0.06, coverage);

	const raw = envelope + (worley - 0.35) * billowAmt + turb * 0.2 - threshold;
	return clamp(raw * 1.85, 0, 1);
}

/**
 * Evaluate altocumulus: mid-altitude cellular dappled bands / wavelets.
 */
function evalAltocumulus(
	along: number,
	ny: number,
	across: number,
	noise: SeededNoise,
	coverage: number,
	turbulence: number
): number {
	const r = Math.hypot(along, across);
	const radialMask = 1.0 - smoothstep(0.72, 1.0, r);
	if (radialMask <= 0) return 0;

	const vertMask = smoothstep(0.14, 0.34, ny) * (1.0 - smoothstep(0.64, 0.9, ny));
	if (vertMask <= 0) return 0;

	// Harmonic gravity-wave bands perpendicular to wind + cellular dappled Worley puffs
	const warp = noise.fbm3D(along * 1.6 + 3.1, ny * 1.4, across * 1.6 - 4.7, 2);
	const wave = 0.5 + 0.5 * Math.sin(along * 13.5 + warp * (2.4 + turbulence * 2.0));
	const cells = noise.worleyBillow3D(along * 4.4, ny * 2.2, across * 4.4);
	const fine = noise.noise3D(along * 5.2, ny * 3.0, across * 5.2) * 0.16;

	const dappled = wave * 0.58 + cells * 0.55 + fine;
	const threshold = lerp(0.68, 0.34, coverage);
	return clamp((dappled - threshold) * 2.35 * vertMask * radialMask, 0, 1);
}

/**
 * Generate the full 3D procedural cloud point field, baked volumetric self-shadowing,
 * and 2D ground shadow occlusion map.
 */
export function generateCloudField(
	params: CloudParams,
	meteo: MeteorologyState
): GeneratedCloudData {
	const seed = hashRegionSeed(params.regionSeed, params.latitude, params.longitude);
	const noise = createSeededNoise(seed);
	const bounds = computeVolumeBounds(params, meteo);
	const weights = meteo.weights;

	const totalVoxels = GRID_X * GRID_Y * GRID_Z;
	const densityGrid = new Float32Array(totalVoxels);
	const shadeGrid = new Float32Array(totalVoxels);

	const windRad = (params.windDirection * Math.PI) / 180;
	const dirX = Math.sin(windRad);
	const dirZ = -Math.cos(windRad);

	const useCumulus = weights.cumulus > 0.015;
	const useStratus = weights.stratus > 0.015;
	const useCirrus = weights.cirrus > 0.015;
	const useCb = weights.cumulonimbus > 0.015;
	const useAlto = weights.altocumulus > 0.015;

	const convectiveCells = useCumulus ? createConvectiveCells(noise, params.coverage) : [];
	const humidityBoost = 0.72 + clamp(params.humidity, 0, 1) * 0.42;

	// Stage 1: Populate 3D volumetric density grid
	const occupiedIndices: number[] = [];
	const occupiedWeights: number[] = [];
	let totalWeight = 0;

	for (let iy = 0; iy < GRID_Y; iy++) {
		const ny = (iy + 0.5) / GRID_Y;
		const yStride = iy * GRID_X * GRID_Z;
		for (let iz = 0; iz < GRID_Z; iz++) {
			const nz = ((iz + 0.5) / GRID_Z) * 2 - 1;
			const zStride = yStride + iz * GRID_X;
			for (let ix = 0; ix < GRID_X; ix++) {
				const nx = ((ix + 0.5) / GRID_X) * 2 - 1;

				const along = nx * dirX + nz * dirZ;
				const across = -nx * dirZ + nz * dirX;

				let d = 0;
				if (useCumulus) {
					d += weights.cumulus * evalCumulus(nx, ny, nz, noise, convectiveCells, params.coverage, params.turbulence);
				}
				if (useStratus) {
					d += weights.stratus * evalStratus(nx, ny, nz, noise, params.coverage, params.humidity, params.turbulence);
				}
				if (useCirrus) {
					d += weights.cirrus * evalCirrus(along, ny, across, noise, params.coverage, params.turbulence);
				}
				if (useCb) {
					d += weights.cumulonimbus * evalCumulonimbus(along, ny, across, noise, params.coverage, params.turbulence, params.windSpeed);
				}
				if (useAlto) {
					d += weights.altocumulus * evalAltocumulus(along, ny, across, noise, params.coverage, params.turbulence);
				}

				d = clamp(d * humidityBoost, 0, 1);
				if (d > 0.035) {
					const idx = zStride + ix;
					densityGrid[idx] = d;
					const wSample = Math.pow(d, 0.82);
					occupiedIndices.push(idx);
					occupiedWeights.push(wSample);
					totalWeight += wSample;
				}
			}
		}
	}

	// Fallback: if extreme parameter combinations yield an empty volume, seed a subtle wisp puff
	if (occupiedIndices.length === 0) {
		const cx = Math.floor(GRID_X / 2);
		const cy = Math.floor(GRID_Y / 2);
		const cz = Math.floor(GRID_Z / 2);
		for (let dy = -2; dy <= 2; dy++) {
			for (let dz = -3; dz <= 3; dz++) {
				for (let dx = -3; dx <= 3; dx++) {
					const r = Math.hypot(dx / 3, dy / 2, dz / 3);
					if (r < 1) {
						const idx = (cy + dy) * GRID_X * GRID_Z + (cz + dz) * GRID_X + (cx + dx);
						const d = (1 - r) * 0.5;
						densityGrid[idx] = d;
						occupiedIndices.push(idx);
						occupiedWeights.push(d);
						totalWeight += d;
					}
				}
			}
		}
	}

	// Stage 2: Bake 3D directional sun self-shadowing + ambient occlusion on occupied voxels
	const [sunX, sunY, sunZ] = meteo.sunDir;
	// Scale step in voxel-grid units respecting world aspect ratio
	const voxelStepX = (sunX / (HALF_EXTENT_XZ * 2)) * GRID_X;
	const voxelStepY = (sunY / Math.max(bounds.heightY, 1.0)) * GRID_Y;
	const voxelStepZ = (sunZ / (HALF_EXTENT_XZ * 2)) * GRID_Z;
	const stepLen = Math.hypot(voxelStepX, voxelStepY, voxelStepZ) || 1;
	const stepScale = 1.65 / stepLen;
	const dsx = voxelStepX * stepScale;
	const dsy = voxelStepY * stepScale;
	const dsz = voxelStepZ * stepScale;

	const xzPlane = GRID_X * GRID_Z;
	for (let k = 0; k < occupiedIndices.length; k++) {
		const idx = occupiedIndices[k];
		const iy = Math.floor(idx / xzPlane);
		const rem = idx - iy * xzPlane;
		const iz = Math.floor(rem / GRID_X);
		const ix = rem - iz * GRID_X;

		// Raymarch 7 steps toward sun for Beer-Lambert transmittance
		let opticalDepth = 0;
		for (let step = 1; step <= 7; step++) {
			const sx = Math.round(ix + dsx * step);
			const sy = Math.round(iy + dsy * step);
			const sz = Math.round(iz + dsz * step);
			if (sx < 0 || sx >= GRID_X || sy < 0 || sy >= GRID_Y || sz < 0 || sz >= GRID_Z) {
				break;
			}
			const sIdx = sy * xzPlane + sz * GRID_X + sx;
			opticalDepth += densityGrid[sIdx] * (step <= 2 ? 0.55 : 0.38);
		}

		// Local 6-neighbor ambient occlusion
		let neighborSum = 0;
		if (ix > 0) neighborSum += densityGrid[idx - 1];
		if (ix < GRID_X - 1) neighborSum += densityGrid[idx + 1];
		if (iz > 0) neighborSum += densityGrid[idx - GRID_X];
		if (iz < GRID_Z - 1) neighborSum += densityGrid[idx + GRID_X];
		if (iy > 0) neighborSum += densityGrid[idx - xzPlane];
		if (iy < GRID_Y - 1) neighborSum += densityGrid[idx + xzPlane];
		const ao = 1.0 - clamp(neighborSum / 6.0, 0, 1) * 0.45;

		const ny = (iy + 0.5) / GRID_Y;
		const transmittance = Math.exp(-opticalDepth * 1.45);
		const heightLift = 0.24 + 0.76 * Math.pow(ny, 0.72);
		shadeGrid[idx] = clamp(transmittance * 0.68 + heightLift * ao * 0.32, 0.04, 1.0);
	}

	// Stage 3: Pack full 3D density volume into a 2D slice atlas (7x4 tiles of 52x52 = 364x208 RGBA)
	// for real-time 3D GPU raymarching of both ground shadows and cloud self-shadowing.
	const volumeAtlasData = new Uint8Array(VOLUME_ATLAS_WIDTH * VOLUME_ATLAS_HEIGHT * 4);
	for (let iy = 0; iy < GRID_Y; iy++) {
		const tileCol = iy % VOLUME_ATLAS_COLS;
		const tileRow = Math.floor(iy / VOLUME_ATLAS_COLS);
		const sliceOffset = iy * xzPlane;
		for (let iz = 0; iz < GRID_Z; iz++) {
			const atlasY = tileRow * GRID_Z + iz;
			const zRowOffset = sliceOffset + iz * GRID_X;
			for (let ix = 0; ix < GRID_X; ix++) {
				const atlasX = tileCol * GRID_X + ix;
				const d = densityGrid[zRowOffset + ix];
				const v = Math.round(clamp(d, 0, 1) * 255);
				const pIdx = (atlasY * VOLUME_ATLAS_WIDTH + atlasX) * 4;
				volumeAtlasData[pIdx] = v;
				volumeAtlasData[pIdx + 1] = v;
				volumeAtlasData[pIdx + 2] = v;
				volumeAtlasData[pIdx + 3] = 255;
			}
		}
	}

	// Stage 4: Importance-sample target point count from occupied voxels via CDF
	const targetPoints = Math.round(clamp(params.pointCount, 15000, 150000));
	const cdf = new Float32Array(occupiedIndices.length);
	let acc = 0;
	const invTotal = 1.0 / Math.max(totalWeight, 1e-6);
	for (let i = 0; i < occupiedWeights.length; i++) {
		acc += occupiedWeights[i] * invTotal;
		cdf[i] = acc;
	}
	cdf[cdf.length - 1] = 1.0;

	const positions = new Float32Array(targetPoints * 3);
	const aDensity = new Float32Array(targetPoints);
	const aHeightNorm = new Float32Array(targetPoints);
	const aSeed = new Float32Array(targetPoints);
	const aSize = new Float32Array(targetPoints);
	const aShade = new Float32Array(targetPoints);

	const voxelWorldX = (HALF_EXTENT_XZ * 2) / GRID_X;
	const voxelWorldY = bounds.heightY / GRID_Y;
	const voxelWorldZ = (HALF_EXTENT_XZ * 2) / GRID_Z;

	// Extra anisotropic filament jitter for cirrus streaks
	const cirrusJitter = weights.cirrus * 0.85;
	const occLen = occupiedIndices.length;

	for (let p = 0; p < targetPoints; p++) {
		// Binary search in CDF
		const u = noise.rand();
		let lo = 0;
		let hi = occLen - 1;
		while (lo < hi) {
			const mid = (lo + hi) >>> 1;
			if (cdf[mid] < u) {
				lo = mid + 1;
			} else {
				hi = mid;
			}
		}

		const voxelIdx = occupiedIndices[lo];
		const iy = Math.floor(voxelIdx / xzPlane);
		const rem = voxelIdx - iy * xzPlane;
		const iz = Math.floor(rem / GRID_X);
		const ix = rem - iz * GRID_X;

		// Sub-voxel jitter with slight Gaussian-like softening (sum of two uniforms) to eliminate grid seams
		const jx = (noise.rand() + noise.rand() - 1.0) * 0.72;
		const jy = (noise.rand() + noise.rand() - 1.0) * 0.72;
		const jz = (noise.rand() + noise.rand() - 1.0) * 0.72;

		let wx = (ix + 0.5 + jx) * voxelWorldX - HALF_EXTENT_XZ;
		let wy = bounds.baseY + (iy + 0.5 + jy) * voxelWorldY;
		let wz = (iz + 0.5 + jz) * voxelWorldZ - HALF_EXTENT_XZ;

		if (cirrusJitter > 0.05) {
			const streak = (noise.rand() - 0.5) * 1.45 * cirrusJitter;
			wx += dirX * streak;
			wz += dirZ * streak;
		}

		const p3 = p * 3;
		positions[p3] = wx;
		positions[p3 + 1] = wy;
		positions[p3 + 2] = wz;

		const density = densityGrid[voxelIdx];
		const heightNorm = clamp((wy - bounds.baseY) / Math.max(bounds.heightY, 0.1), 0, 1);
		const ptSeed = noise.rand();

		aDensity[p] = density;
		aHeightNorm[p] = heightNorm;
		aSeed[p] = ptSeed;

		// Vary splat diameter by density & morphology (airier smaller dots for cirrus, fuller dots for dense cores)
		const morphSizeScale = 1.0 - weights.cirrus * 0.25 + weights.cumulonimbus * 0.12;
		aSize[p] = (0.62 + density * 0.68 + ptSeed * 0.38) * morphSizeScale;

		// Subtle per-point micro-variation on baked shade for crisp stipple texture
		aShade[p] = clamp(shadeGrid[voxelIdx] + (ptSeed - 0.5) * 0.08, 0.02, 1.0);
	}

	return {
		positions,
		aDensity,
		aHeightNorm,
		aSeed,
		aSize,
		aShade,
		activePoints: targetPoints,
		volumeAtlasData,
		volumeAtlasWidth: VOLUME_ATLAS_WIDTH,
		volumeAtlasHeight: VOLUME_ATLAS_HEIGHT,
		bounds
	};
}
