/**
 * Infinite non-repeating streaming 3D volumetric cloud point-field generator ("inf-dev").
 *
 * Evaluates all 5 meteorological cloud morphologies in unbounded continuous world
 * coordinates (wx, y, wz) using infinite spatial lattice hashing + 3D Simplex/Worley FBM.
 * Maintains a rolling sliding-window 3D voxel volume (52x28x52) packed into a 7x4 2D slice
 * atlas (364x208) for real-time 3D GPU raymarched shadows, and streams new upwind slices
 * incrementally while recycling downwind-exited points — so brand-new clouds continuously
 * roll in from the horizon and never wrap or repeat.
 *
 * Also maintains a live 2D column-density & local-cloud-base lookup so precipitation falls
 * strictly underneath dense cloud cores rather than as a uniform blanket.
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

export interface RainColumnSample {
	x: number;
	y: number;
	z: number;
	density: number;
}

export interface StreamingCloudField {
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
	/** Sub-voxel continuous visual offset in world units (always within [-VOXEL_XZ, +VOXEL_XZ]) */
	subOffsetX: number;
	subOffsetZ: number;
	/**
	 * Advance the infinite sky by (driftX, driftZ) world units.
	 * Returns true when at least one voxel slice shifted and GPU buffers/atlas need re-upload.
	 */
	advance(driftX: number, driftZ: number): boolean;
	/**
	 * Sample a random world position strictly underneath an actual dense cloud column.
	 * Returns null when no sufficiently dense cloud is currently overhead.
	 */
	sampleRainColumn(rand: () => number, minDensity?: number): RainColumnSample | null;
	/** Current fraction [0..1] of the sky window covered by rain-capable cloud columns */
	getOverheadCloudFactor(): number;
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
const XZ_PLANE = GRID_X * GRID_Z;
const HALF_EXTENT_XZ = 12.5;
const SPAN_XZ = HALF_EXTENT_XZ * 2;
const VOXEL_XZ = SPAN_XZ / GRID_X; // ~0.48077 world units per voxel

/**
 * Deterministic fast 2D integer hash -> [0, 1) for infinite world-space cloud cells.
 */
function hash2D(ix: number, iz: number, seed: number): number {
	let h = (Math.imul(ix, 374761393) + Math.imul(iz, 668265263) + Math.imul(seed, 1442695041)) | 0;
	h = Math.imul(h ^ (h >>> 13), 1274126177);
	h = h ^ (h >>> 16);
	return (h >>> 0) / 4294967296;
}

/**
 * Compute world-space vertical placement and thickness from meteorological state.
 */
function computeVolumeBounds(params: CloudParams, meteo: MeteorologyState): CloudVolumeBounds {
	const w = meteo.weights;
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
 * Infinite world-space Cumulus: endless hashed convective domes with flat condensation bases
 * and billowing cauliflower puffs that roll across the sky without ever repeating.
 */
function evalInfiniteCumulus(
	wx: number,
	ny: number,
	wz: number,
	noise: SeededNoise,
	seed: number,
	coverage: number,
	turbulence: number
): number {
	const baseProfile = smoothstep(0.02, 0.13, ny);
	if (baseProfile <= 0) return 0;

	// Infinite jittered lattice of convective towers (cell size ~9.8 world units)
	const cellSize = 9.8;
	const gx = wx / cellSize;
	const gz = wz / cellSize;
	const ix0 = Math.floor(gx);
	const iz0 = Math.floor(gz);

	// Cell spawn probability scales with coverage (0.28..0.88)
	const spawnProb = lerp(0.28, 0.88, clamp(coverage, 0, 1));
	let domeField = -1.0;

	for (let dz = -1; dz <= 1; dz++) {
		const cz = iz0 + dz;
		for (let dx = -1; dx <= 1; dx++) {
			const cx = ix0 + dx;
			// Guarantee one well-framed initial cell near origin (0,0) so initial load is rich
			const isOriginCell = cx === 0 && cz === 0;
			const p = hash2D(cx, cz, seed ^ 0x1a2b3c);
			if (!isOriginCell && p > spawnProb) continue;

			const jx = (hash2D(cx, cz, seed ^ 0x4d5e6f) - 0.5) * 0.62;
			const jz = (hash2D(cx, cz, seed ^ 0x7a8b9c) - 0.5) * 0.62;
			const cellCenterX = (cx + 0.5 + (isOriginCell ? jx * 0.35 : jx)) * cellSize;
			const cellCenterZ = (cz + 0.5 + (isOriginCell ? jz * 0.35 : jz)) * cellSize;

			const rBase =
				(0.32 + hash2D(cx, cz, seed ^ 0x918273) * 0.24 + coverage * 0.12) * cellSize;
			const cellH = 0.56 + hash2D(cx, cz, seed ^ 0x2468ac) * 0.42;

			const ddx = wx - cellCenterX;
			const ddz = wz - cellCenterZ;
			const rScale = rBase * (1.08 - ny * 0.48);
			const r2 = (ddx * ddx + ddz * ddz) / Math.max(rScale * rScale, 0.25);
			const hNorm = ny / cellH;

			if (r2 < 1.35 && hNorm < 1.1) {
				const dome = 1.0 - r2 * 0.82 - Math.pow(Math.max(0, hNorm - 0.12), 1.85) * 0.95;
				if (dome > domeField) domeField = dome;
			}
		}
	}

	if (domeField <= -0.25) return 0;

	const nx = wx / HALF_EXTENT_XZ;
	const nz = wz / HALF_EXTENT_XZ;
	const worleyPuff = noise.worleyBillow3D(nx * 3.1 + 1.7, ny * 2.8, nz * 3.1 - 2.3);
	const detail = noise.fbm3D(nx * 2.6, ny * 2.4, nz * 2.6, 3);
	const puffStrength = 0.34 + turbulence * 0.26;
	const raw = domeField * baseProfile + (worleyPuff - 0.38) * puffStrength + detail * 0.22;

	const threshold = lerp(0.32, 0.08, coverage);
	return clamp((raw - threshold) * 1.9, 0, 1);
}

/**
 * Infinite world-space Stratus: rolling coastal sheets with natural synoptic clearings.
 */
function evalInfiniteStratus(
	wx: number,
	ny: number,
	wz: number,
	noise: SeededNoise,
	coverage: number,
	humidity: number,
	turbulence: number
): number {
	const nx = wx / HALF_EXTENT_XZ;
	const nz = wz / HALF_EXTENT_XZ;

	const undulation = noise.fbm3D(nx * 1.1 + 4.1, 0.25, nz * 1.1 - 3.2, 3);
	const centerY = 0.44 + undulation * (0.12 + turbulence * 0.12);
	const halfThick = 0.22 + humidity * 0.18;
	const verticalDist = Math.abs(ny - centerY) / halfThick;
	const vertMask = 1.0 - smoothstep(0.35, 1.05, verticalDist);
	if (vertMask <= 0) return 0;

	// Broad synoptic banks so stratus decks naturally thicken and part over time
	const macroBank = noise.fbm3D(nx * 0.45 - 2.7, 0.5, nz * 0.45 + 6.3, 2);
	const sheetNoise = noise.fbm3D(nx * 1.85, ny * 1.6, nz * 1.85, 3);
	const threshold = lerp(0.42, -0.32, coverage);
	const density = (sheetNoise * 0.65 + macroBank * 0.48 + 0.45 - threshold) * vertMask;
	return clamp(density * 1.35, 0, 1);
}

/**
 * Infinite world-space Cirrus: endless windswept ice-crystal filaments stretched along wind.
 */
function evalInfiniteCirrus(
	alongWorld: number,
	ny: number,
	acrossWorld: number,
	noise: SeededNoise,
	coverage: number,
	turbulence: number
): number {
	const along = alongWorld / HALF_EXTENT_XZ;
	const across = acrossWorld / HALF_EXTENT_XZ;

	// Large-scale synoptic cirrus streaks & clearings
	const patch = noise.fbm3D(along * 0.35 + 1.9, 0.4, across * 0.55 - 3.4, 2) * 0.5 + 0.5;
	const patchMask = smoothstep(lerp(0.62, 0.22, coverage), lerp(0.86, 0.54, coverage), patch);
	if (patchMask <= 0.01) return 0;

	const warp = noise.fbm3D(along * 0.85 + 7.2, ny * 1.4, across * 2.4 - 5.1, 2);
	const hookY = 0.5 + Math.sin(along * 2.4 + warp * 2.2) * 0.22 + warp * 0.12;
	const verticalBand = 1.0 - smoothstep(0.08, 0.34, Math.abs(ny - hookY));
	if (verticalBand <= 0) return 0;

	const sx = along * 0.72 + warp * (0.45 + turbulence * 0.35);
	const sz = across * 3.8 + warp * (0.75 + turbulence * 0.55);
	const ridge1 = 1.0 - Math.abs(noise.fbm3D(sx, ny * 1.2, sz, 3));
	const ridge2 = 1.0 - Math.abs(noise.noise3D(sx * 1.7 + 12.3, ny * 2.0, sz * 1.6 - 8.4));
	const filament = Math.pow(clamp(ridge1 * 0.65 + ridge2 * 0.35, 0, 1), 2.35);

	const threshold = lerp(0.46, 0.14, coverage);
	const density = (filament - threshold) * 2.5 * verticalBand * patchMask;
	return clamp(density, 0, 1);
}

/**
 * Infinite world-space Cumulonimbus: endless passing storm supercells with towering updraft
 * columns, sheared flat downwind anvils, and overshooting domes.
 */
function evalInfiniteCumulonimbus(
	wx: number,
	ny: number,
	wz: number,
	dirX: number,
	dirZ: number,
	noise: SeededNoise,
	seed: number,
	coverage: number,
	turbulence: number,
	windSpeed: number
): number {
	const baseRamp = smoothstep(0.02, 0.09, ny);
	if (baseRamp <= 0) return 0;

	// Supercells live on a wide infinite lattice (cell size ~21 world units)
	const cellSize = 21.0;
	const gx = wx / cellSize;
	const gz = wz / cellSize;
	const ix0 = Math.floor(gx);
	const iz0 = Math.floor(gz);

	const shearFactor = 0.14 + clamp(windSpeed / 40, 0, 1) * 0.24;
	const spawnProb = lerp(0.35, 0.78, clamp(coverage, 0, 1));

	let envelope = 0;
	let bestAlong = 0;
	let bestAcross = 0;

	for (let dz = -1; dz <= 1; dz++) {
		const cz = iz0 + dz;
		for (let dx = -1; dx <= 1; dx++) {
			const cx = ix0 + dx;
			const isOriginCell = cx === 0 && cz === 0;
			if (!isOriginCell && hash2D(cx, cz, seed ^ 0x55aa33) > spawnProb) continue;

			const jx = (hash2D(cx, cz, seed ^ 0x112233) - 0.5) * 0.52;
			const jz = (hash2D(cx, cz, seed ^ 0x445566) - 0.5) * 0.52;
			const cellCenterX = (cx + (isOriginCell ? 0.02 + jx * 0.15 : 0.5 + jx)) * cellSize;
			const cellCenterZ = (cz + (isOriginCell ? 0.02 + jz * 0.15 : 0.5 + jz)) * cellSize;

			const scale = isOriginCell ? 1.0 : 0.76 + hash2D(cx, cz, seed ^ 0x778899) * 0.38;
			const relX = (wx - cellCenterX) / (HALF_EXTENT_XZ * scale);
			const relZ = (wz - cellCenterZ) / (HALF_EXTENT_XZ * scale);

			if (relX * relX + relZ * relZ > 1.7) continue;

			const along = relX * dirX + relZ * dirZ;
			const across = -relX * dirZ + relZ * dirX;

			// 1. Broad flat-bottomed inflow base
			const baseR = Math.hypot(along * 1.1, across * 1.1);
			const baseEnv =
				(1.0 - smoothstep(0.32, 0.62, baseR)) *
				baseRamp *
				(1.0 - smoothstep(0.18, 0.34, ny));

			// 2. Towering convective updraft column
			const towerCenterAlong = ny * shearFactor * 0.75;
			const towerR = Math.hypot((along - towerCenterAlong) * 1.45, across * 1.45);
			const towerRadius = lerp(0.46, 0.34, smoothstep(0.15, 0.65, ny));
			const towerEnv =
				(1.0 - smoothstep(towerRadius * 0.45, towerRadius, towerR)) *
				smoothstep(0.05, 0.15, ny) *
				(1.0 - smoothstep(0.76, 0.94, ny));

			// 3. Sheared flat anvil top (incus)
			const anvilAlongOffset = 0.18 + shearFactor * 0.85;
			const anvilR = Math.hypot((along - anvilAlongOffset) / 0.78, across / 0.62);
			const anvilVert = smoothstep(0.62, 0.75, ny) * (1.0 - smoothstep(0.84, 0.94, ny));
			const anvilEnv = (1.0 - smoothstep(0.48, 1.0, anvilR)) * anvilVert * 1.15;

			// 4. Overshooting top dome
			const domeR = Math.hypot((along - towerCenterAlong) * 2.4, across * 2.4);
			const domeEnv =
				(1.0 - smoothstep(0.25, 0.58, domeR)) *
				smoothstep(0.74, 0.86, ny) *
				(1.0 - smoothstep(0.92, 1.0, ny));

			const cellEnv = Math.max(baseEnv, towerEnv, anvilEnv, domeEnv);
			if (cellEnv > envelope) {
				envelope = cellEnv;
				bestAlong = along;
				bestAcross = across;
			}
		}
	}

	if (envelope <= 0.02) return 0;

	const worley = noise.worleyBillow3D(bestAlong * 3.0 + wx * 0.06, ny * 3.6, bestAcross * 3.0 + wz * 0.06);
	const turb = noise.fbm3D(wx * 0.18, ny * 3.2, wz * 0.18, 3);
	const billowAmt = 0.25 + turbulence * 0.28;
	const threshold = lerp(0.26, 0.06, coverage);

	const raw = envelope + (worley - 0.35) * billowAmt + turb * 0.2 - threshold;
	return clamp(raw * 1.85, 0, 1);
}

/**
 * Infinite world-space Altocumulus: endless passing cellular dappled bands & wavelets.
 */
function evalInfiniteAltocumulus(
	alongWorld: number,
	ny: number,
	acrossWorld: number,
	noise: SeededNoise,
	coverage: number,
	turbulence: number
): number {
	const vertMask = smoothstep(0.14, 0.34, ny) * (1.0 - smoothstep(0.64, 0.9, ny));
	if (vertMask <= 0) return 0;

	const along = alongWorld / HALF_EXTENT_XZ;
	const across = acrossWorld / HALF_EXTENT_XZ;

	// Large-scale synoptic patches so dappled sheets drift in and out naturally
	const macro = noise.fbm3D(along * 0.42 - 1.3, 0.3, across * 0.42 + 2.8, 2) * 0.5 + 0.5;
	const macroMask = smoothstep(lerp(0.56, 0.18, coverage), lerp(0.84, 0.48, coverage), macro);
	if (macroMask <= 0.01) return 0;

	const warp = noise.fbm3D(along * 1.6 + 3.1, ny * 1.4, across * 1.6 - 4.7, 2);
	const wave = 0.5 + 0.5 * Math.sin(along * 13.5 + warp * (2.4 + turbulence * 2.0));
	const cells = noise.worleyBillow3D(along * 4.4, ny * 2.2, across * 4.4);
	const fine = noise.noise3D(along * 5.2, ny * 3.0, across * 5.2) * 0.16;

	const dappled = wave * 0.58 + cells * 0.55 + fine;
	const threshold = lerp(0.68, 0.34, coverage);
	return clamp((dappled - threshold) * 2.35 * vertMask * macroMask, 0, 1);
}

/**
 * Create a live infinite streaming procedural cloud field ("inf-dev").
 */
export function createStreamingCloudField(
	params: CloudParams,
	meteo: MeteorologyState,
	initialScrollX = 0,
	initialScrollZ = 0
): StreamingCloudField {
	const seed = hashRegionSeed(params.regionSeed, params.latitude, params.longitude);
	const noise = createSeededNoise(seed);
	const bounds = computeVolumeBounds(params, meteo);
	const weights = meteo.weights;

	const windRad = (params.windDirection * Math.PI) / 180;
	const dirX = Math.sin(windRad);
	const dirZ = -Math.cos(windRad);

	const useCumulus = weights.cumulus > 0.015;
	const useStratus = weights.stratus > 0.015;
	const useCirrus = weights.cirrus > 0.015;
	const useCb = weights.cumulonimbus > 0.015;
	const useAlto = weights.altocumulus > 0.015;
	const humidityBoost = 0.72 + clamp(params.humidity, 0, 1) * 0.42;

	const totalVoxels = GRID_X * GRID_Y * GRID_Z;
	const densityGrid = new Float32Array(totalVoxels);
	const scratchGrid = new Float32Array(totalVoxels);

	// 2D column summaries for localized rain spawning strictly under dense clouds
	const colMaxDensity = new Float32Array(XZ_PLANE);
	const colIntegral = new Float32Array(XZ_PLANE);
	const colBaseY = new Float32Array(XZ_PLANE);
	const rainCandidateCols = new Int32Array(XZ_PLANE);
	const rainCandidateCdf = new Float32Array(XZ_PLANE);
	let rainCandidateCount = 0;
	let overheadCloudFactor = 0;

	const volumeAtlasData = new Uint8Array(VOLUME_ATLAS_WIDTH * VOLUME_ATLAS_HEIGHT * 4);

	const targetPoints = Math.round(clamp(params.pointCount, 15000, 150000));
	const positions = new Float32Array(targetPoints * 3);
	const aDensity = new Float32Array(targetPoints);
	const aHeightNorm = new Float32Array(targetPoints);
	const aSeed = new Float32Array(targetPoints);
	const aSize = new Float32Array(targetPoints);
	const aShade = new Float32Array(targetPoints);

	// Track which points are currently parked (inactive) vs live
	const freeSlots = new Int32Array(targetPoints);
	let freeCount = 0;

	const voxelWorldY = bounds.heightY / GRID_Y;
	const cirrusJitter = weights.cirrus * 0.85;
	const morphSizeScale = 1.0 - weights.cirrus * 0.25 + weights.cumulonimbus * 0.12;

	// Continuous world-space scroll & integer voxel grid origin
	let scrollX = initialScrollX;
	let scrollZ = initialScrollZ;
	let gridOriginX = Math.round(scrollX / VOXEL_XZ);
	let gridOriginZ = Math.round(scrollZ / VOXEL_XZ);

	// Reference point density per voxel weight unit (calibrated on initial populate)
	let pointsPerWeight = 1.0;

	/**
	 * Evaluate combined procedural cloud density at an infinite world-space coordinate.
	 */
	function evalWorldDensity(wx: number, ny: number, wz: number): number {
		let d = 0;
		if (useCumulus) {
			d += weights.cumulus * evalInfiniteCumulus(wx, ny, wz, noise, seed, params.coverage, params.turbulence);
		}
		if (useStratus) {
			d += weights.stratus * evalInfiniteStratus(wx, ny, wz, noise, params.coverage, params.humidity, params.turbulence);
		}
		if (useCirrus || useAlto) {
			const alongWorld = wx * dirX + wz * dirZ;
			const acrossWorld = -wx * dirZ + wz * dirX;
			if (useCirrus) {
				d += weights.cirrus * evalInfiniteCirrus(alongWorld, ny, acrossWorld, noise, params.coverage, params.turbulence);
			}
			if (useAlto) {
				d += weights.altocumulus * evalInfiniteAltocumulus(alongWorld, ny, acrossWorld, noise, params.coverage, params.turbulence);
			}
		}
		if (useCb) {
			d += weights.cumulonimbus * evalInfiniteCumulonimbus(
				wx,
				ny,
				wz,
				dirX,
				dirZ,
				noise,
				seed,
				params.coverage,
				params.turbulence,
				params.windSpeed
			);
		}
		d = clamp(d * humidityBoost, 0, 1);
		return d > 0.035 ? d : 0;
	}

	/**
	 * Populate a single vertical column (ix, iz) in densityGrid from infinite world coordinates.
	 */
	function evaluateColumn(ix: number, iz: number): void {
		const localX = (ix + 0.5) * VOXEL_XZ - HALF_EXTENT_XZ;
		const localZ = (iz + 0.5) * VOXEL_XZ - HALF_EXTENT_XZ;
		// Subtracting gridOrigin * VOXEL_XZ means positive wind drift moves clouds in +dir
		const worldX = localX - gridOriginX * VOXEL_XZ;
		const worldZ = localZ - gridOriginZ * VOXEL_XZ;

		const colIdx = iz * GRID_X + ix;
		for (let iy = 0; iy < GRID_Y; iy++) {
			const ny = (iy + 0.5) / GRID_Y;
			densityGrid[iy * XZ_PLANE + colIdx] = evalWorldDensity(worldX, ny, worldZ);
		}
	}

	/**
	 * Pack densityGrid into the 7x4 2D slice atlas and refresh 2D rain-column lookup tables.
	 */
	function syncAtlasAndRainColumns(): void {
		// 1. Pack 3D volume atlas (364x208 RGBA)
		for (let iy = 0; iy < GRID_Y; iy++) {
			const tileCol = iy % VOLUME_ATLAS_COLS;
			const tileRow = Math.floor(iy / VOLUME_ATLAS_COLS);
			const sliceOffset = iy * XZ_PLANE;
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

		// 2. Refresh 2D column density & local cloud base height for localized rainfall
		let candidateCount = 0;
		let totalRainWeight = 0;
		let coveredCols = 0;

		for (let iz = 0; iz < GRID_Z; iz++) {
			const localZ = (iz + 0.5) * VOXEL_XZ - HALF_EXTENT_XZ;
			for (let ix = 0; ix < GRID_X; ix++) {
				const localX = (ix + 0.5) * VOXEL_XZ - HALF_EXTENT_XZ;
				const colIdx = iz * GRID_X + ix;

				// Keep rain strictly within the visible circular sky plate
				const radial = Math.hypot(localX, localZ) / HALF_EXTENT_XZ;
				if (radial > 0.92) {
					colMaxDensity[colIdx] = 0;
					colIntegral[colIdx] = 0;
					colBaseY[colIdx] = bounds.baseY;
					continue;
				}

				let maxD = 0;
				let sumD = 0;
				let firstOccupiedIy = -1;
				for (let iy = 0; iy < GRID_Y; iy++) {
					const d = densityGrid[iy * XZ_PLANE + colIdx];
					if (d > 0.08) {
						if (firstOccupiedIy === -1) firstOccupiedIy = iy;
						if (d > maxD) maxD = d;
						sumD += d;
					}
				}

				// Normalized column optical thickness in [0..1]
				const colStrength = 1.0 - Math.exp(-sumD * 0.42);
				colMaxDensity[colIdx] = maxD;
				colIntegral[colIdx] = colStrength;
				colBaseY[colIdx] =
					firstOccupiedIy >= 0
						? bounds.baseY + (firstOccupiedIy + 0.35) * voxelWorldY
						: bounds.baseY;

				if (colStrength > 0.22) {
					coveredCols++;
					const edgeFade = 1.0 - smoothstep(0.74, 0.92, radial);
					const w = Math.pow(colStrength, 1.85) * edgeFade;
					if (w > 0.01) {
						rainCandidateCols[candidateCount] = colIdx;
						totalRainWeight += w;
						rainCandidateCdf[candidateCount] = totalRainWeight;
						candidateCount++;
					}
				}
			}
		}

		if (candidateCount > 0 && totalRainWeight > 0) {
			const inv = 1.0 / totalRainWeight;
			for (let i = 0; i < candidateCount; i++) {
				rainCandidateCdf[i] *= inv;
			}
			rainCandidateCdf[candidateCount - 1] = 1.0;
		}
		rainCandidateCount = candidateCount;
		overheadCloudFactor = clamp(coveredCols / (XZ_PLANE * 0.55), 0, 1);
	}

	/**
	 * Write a single cloud point at slot `p` inside voxel (ix, iy, iz).
	 */
	function writePointAtVoxel(p: number, ix: number, iy: number, iz: number, density: number): void {
		const jx = (noise.rand() + noise.rand() - 1.0) * 0.72;
		const jy = (noise.rand() + noise.rand() - 1.0) * 0.72;
		const jz = (noise.rand() + noise.rand() - 1.0) * 0.72;

		let wx = (ix + 0.5 + jx) * VOXEL_XZ - HALF_EXTENT_XZ;
		const wy = bounds.baseY + (iy + 0.5 + jy) * voxelWorldY;
		let wz = (iz + 0.5 + jz) * VOXEL_XZ - HALF_EXTENT_XZ;

		if (cirrusJitter > 0.05) {
			const streak = (noise.rand() - 0.5) * 1.45 * cirrusJitter;
			wx += dirX * streak;
			wz += dirZ * streak;
		}

		const p3 = p * 3;
		positions[p3] = wx;
		positions[p3 + 1] = wy;
		positions[p3 + 2] = wz;

		const heightNorm = clamp((wy - bounds.baseY) / Math.max(bounds.heightY, 0.1), 0, 1);
		const ptSeed = noise.rand();

		// Local 6-neighbor ambient occlusion
		const idx = iy * XZ_PLANE + iz * GRID_X + ix;
		let neighborSum = 0;
		if (ix > 0) neighborSum += densityGrid[idx - 1];
		if (ix < GRID_X - 1) neighborSum += densityGrid[idx + 1];
		if (iz > 0) neighborSum += densityGrid[idx - GRID_X];
		if (iz < GRID_Z - 1) neighborSum += densityGrid[idx + GRID_X];
		if (iy > 0) neighborSum += densityGrid[idx - XZ_PLANE];
		if (iy < GRID_Y - 1) neighborSum += densityGrid[idx + XZ_PLANE];
		const ao = 1.0 - clamp(neighborSum / 6.0, 0, 1) * 0.48;

		aDensity[p] = density;
		aHeightNorm[p] = heightNorm;
		aSeed[p] = ptSeed;
		aSize[p] = (0.62 + density * 0.68 + ptSeed * 0.38) * morphSizeScale;
		aShade[p] = clamp(ao + (ptSeed - 0.5) * 0.08, 0.08, 1.0);
	}

	function parkPoint(p: number): void {
		aDensity[p] = 0;
		const p3 = p * 3;
		positions[p3] = 0;
		positions[p3 + 1] = -999;
		positions[p3 + 2] = 0;
		freeSlots[freeCount++] = p;
	}

	// -------------------------------------------------------------------------
	// Initial Full Volume Evaluation & Importance-Sampled Point Population
	// -------------------------------------------------------------------------
	for (let iz = 0; iz < GRID_Z; iz++) {
		for (let ix = 0; ix < GRID_X; ix++) {
			evaluateColumn(ix, iz);
		}
	}

	const occupiedIndices: number[] = [];
	const occupiedWeights: number[] = [];
	let totalWeight = 0;

	for (let idx = 0; idx < totalVoxels; idx++) {
		const d = densityGrid[idx];
		if (d > 0.035) {
			const w = Math.pow(d, 0.82);
			occupiedIndices.push(idx);
			occupiedWeights.push(w);
			totalWeight += w;
		}
	}

	// Calibrate steady-state point density per voxel weight so incoming clouds match initial density
	pointsPerWeight = totalWeight > 1e-3 ? targetPoints / totalWeight : targetPoints / 800;

	if (occupiedIndices.length > 0) {
		const cdf = new Float32Array(occupiedIndices.length);
		let acc = 0;
		const invTotal = 1.0 / Math.max(totalWeight, 1e-6);
		for (let i = 0; i < occupiedWeights.length; i++) {
			acc += occupiedWeights[i] * invTotal;
			cdf[i] = acc;
		}
		cdf[cdf.length - 1] = 1.0;

		const occLen = occupiedIndices.length;
		for (let p = 0; p < targetPoints; p++) {
			const u = noise.rand();
			let lo = 0;
			let hi = occLen - 1;
			while (lo < hi) {
				const mid = (lo + hi) >>> 1;
				if (cdf[mid] < u) lo = mid + 1;
				else hi = mid;
			}
			const voxelIdx = occupiedIndices[lo];
			const iy = Math.floor(voxelIdx / XZ_PLANE);
			const rem = voxelIdx - iy * XZ_PLANE;
			const iz = Math.floor(rem / GRID_X);
			const ix = rem - iz * GRID_X;
			writePointAtVoxel(p, ix, iy, iz, densityGrid[voxelIdx]);
		}
	} else {
		for (let p = 0; p < targetPoints; p++) {
			parkPoint(p);
		}
	}

	syncAtlasAndRainColumns();

	// -------------------------------------------------------------------------
	// Rolling Incremental Slice Shift (Infinite Non-Repeating Streaming)
	// -------------------------------------------------------------------------
	const stripVoxels: number[] = [];
	const stripWeights: number[] = [];

	function shiftGridAndStream(shiftX: number, shiftZ: number): void {
		gridOriginX += shiftX;
		gridOriginZ += shiftZ;

		// 1. Shift 3D densityGrid by (+shiftX, +shiftZ) so existing clouds move smoothly in +dir
		scratchGrid.fill(0);
		for (let iy = 0; iy < GRID_Y; iy++) {
			const yOff = iy * XZ_PLANE;
			for (let iz = 0; iz < GRID_Z; iz++) {
				const srcZ = iz - shiftZ;
				if (srcZ < 0 || srcZ >= GRID_Z) continue;
				const dstRow = yOff + iz * GRID_X;
				const srcRow = yOff + srcZ * GRID_X;
				for (let ix = 0; ix < GRID_X; ix++) {
					const srcX = ix - shiftX;
					if (srcX < 0 || srcX >= GRID_X) continue;
					scratchGrid[dstRow + ix] = densityGrid[srcRow + srcX];
				}
			}
		}
		densityGrid.set(scratchGrid);

		// 2. Procedurally generate ONLY the newly entered upwind column strip(s)
		stripVoxels.length = 0;
		stripWeights.length = 0;
		let stripTotalWeight = 0;

		for (let iz = 0; iz < GRID_Z; iz++) {
			const isNewZ =
				(shiftZ > 0 && iz < shiftZ) || (shiftZ < 0 && iz >= GRID_Z + shiftZ);
			for (let ix = 0; ix < GRID_X; ix++) {
				const isNewX =
					(shiftX > 0 && ix < shiftX) || (shiftX < 0 && ix >= GRID_X + shiftX);
				if (!isNewX && !isNewZ) continue;

				evaluateColumn(ix, iz);
				const colIdx = iz * GRID_X + ix;
				for (let iy = 0; iy < GRID_Y; iy++) {
					const vIdx = iy * XZ_PLANE + colIdx;
					const d = densityGrid[vIdx];
					if (d > 0.035) {
						const w = Math.pow(d, 0.82);
						stripVoxels.push(vIdx);
						stripWeights.push(w);
						stripTotalWeight += w;
					}
				}
			}
		}

		// 3. Shift all live points by (+shiftX * VOXEL_XZ, +shiftZ * VOXEL_XZ) and park exited ones
		const dxWorld = shiftX * VOXEL_XZ;
		const dzWorld = shiftZ * VOXEL_XZ;
		const boundLimit = HALF_EXTENT_XZ + VOXEL_XZ * 0.5;

		for (let p = 0; p < targetPoints; p++) {
			if (aDensity[p] <= 0.005) continue;
			const p3 = p * 3;
			const nx = positions[p3] + dxWorld;
			const nz = positions[p3 + 2] + dzWorld;
			if (
				nx < -boundLimit ||
				nx > boundLimit ||
				nz < -boundLimit ||
				nz > boundLimit
			) {
				parkPoint(p);
			} else {
				positions[p3] = nx;
				positions[p3 + 2] = nz;
			}
		}

		// 4. Spawn fresh points into the newly entered upwind strip from the free pool
		if (stripVoxels.length > 0 && stripTotalWeight > 0) {
			const desiredSpawn = Math.round(stripTotalWeight * pointsPerWeight);
			const spawnCount = Math.min(desiredSpawn, targetPoints);

			// If a very dense cloud rolls in while freeSlots is low, gently reclaim outer-edge points
			if (freeCount < spawnCount) {
				const needed = spawnCount - freeCount;
				let reclaimed = 0;
				const stride = 7;
				let cursor = Math.floor(noise.rand() * targetPoints);
				for (let attempts = 0; attempts < targetPoints && reclaimed < needed; attempts++) {
					const p = cursor;
					cursor = (cursor + stride) % targetPoints;
					if (aDensity[p] <= 0.005) continue;
					const p3 = p * 3;
					// Prefer reclaiming points already near the invisible outer circular fade rim
					const r = Math.hypot(positions[p3], positions[p3 + 2]) / HALF_EXTENT_XZ;
					if (r > 0.86 || attempts > targetPoints * 0.65) {
						parkPoint(p);
						reclaimed++;
					}
				}
			}

			const actualSpawn = Math.min(spawnCount, freeCount);
			if (actualSpawn > 0) {
				const cdf = new Float32Array(stripVoxels.length);
				let acc = 0;
				const inv = 1.0 / stripTotalWeight;
				for (let i = 0; i < stripWeights.length; i++) {
					acc += stripWeights[i] * inv;
					cdf[i] = acc;
				}
				cdf[cdf.length - 1] = 1.0;
				const sLen = stripVoxels.length;

				for (let k = 0; k < actualSpawn; k++) {
					const p = freeSlots[--freeCount];
					const u = noise.rand();
					let lo = 0;
					let hi = sLen - 1;
					while (lo < hi) {
						const mid = (lo + hi) >>> 1;
						if (cdf[mid] < u) lo = mid + 1;
						else hi = mid;
					}
					const voxelIdx = stripVoxels[lo];
					const iy = Math.floor(voxelIdx / XZ_PLANE);
					const rem = voxelIdx - iy * XZ_PLANE;
					const iz = Math.floor(rem / GRID_X);
					const ix = rem - iz * GRID_X;
					writePointAtVoxel(p, ix, iy, iz, densityGrid[voxelIdx]);
				}
			}
		}

		// 5. Refresh 3D volume atlas & 2D localized rain columns
		syncAtlasAndRainColumns();
	}

	const field: StreamingCloudField = {
		positions,
		aDensity,
		aHeightNorm,
		aSeed,
		aSize,
		aShade,
		activePoints: targetPoints - freeCount,
		volumeAtlasData,
		volumeAtlasWidth: VOLUME_ATLAS_WIDTH,
		volumeAtlasHeight: VOLUME_ATLAS_HEIGHT,
		bounds,
		subOffsetX: scrollX - gridOriginX * VOXEL_XZ,
		subOffsetZ: scrollZ - gridOriginZ * VOXEL_XZ,

		advance(driftX: number, driftZ: number): boolean {
			scrollX += driftX;
			scrollZ += driftZ;

			const rawShiftX = Math.trunc((scrollX - gridOriginX * VOXEL_XZ) / VOXEL_XZ);
			const rawShiftZ = Math.trunc((scrollZ - gridOriginZ * VOXEL_XZ) / VOXEL_XZ);

			let shifted = false;
			if (rawShiftX !== 0 || rawShiftZ !== 0) {
				// Clamp huge spikes (e.g. tab switch) to a reasonable shift
				const sx = clamp(rawShiftX, -4, 4);
				const sz = clamp(rawShiftZ, -4, 4);
				if (Math.abs(rawShiftX) > 4 || Math.abs(rawShiftZ) > 4) {
					gridOriginX = Math.round(scrollX / VOXEL_XZ) - sx;
					gridOriginZ = Math.round(scrollZ / VOXEL_XZ) - sz;
				}
				shiftGridAndStream(sx, sz);
				shifted = true;
			}

			field.subOffsetX = scrollX - gridOriginX * VOXEL_XZ;
			field.subOffsetZ = scrollZ - gridOriginZ * VOXEL_XZ;
			field.activePoints = targetPoints - freeCount;
			return shifted;
		},

		sampleRainColumn(rand: () => number, minDensity = 0.22): RainColumnSample | null {
			if (rainCandidateCount === 0) return null;
			const u = rand();
			let lo = 0;
			let hi = rainCandidateCount - 1;
			while (lo < hi) {
				const mid = (lo + hi) >>> 1;
				if (rainCandidateCdf[mid] < u) lo = mid + 1;
				else hi = mid;
			}
			const colIdx = rainCandidateCols[lo];
			const density = colIntegral[colIdx];
			if (density < minDensity) return null;

			const iz = Math.floor(colIdx / GRID_X);
			const ix = colIdx - iz * GRID_X;
			const localX = (ix + rand()) * VOXEL_XZ - HALF_EXTENT_XZ + field.subOffsetX;
			const localZ = (iz + rand()) * VOXEL_XZ - HALF_EXTENT_XZ + field.subOffsetZ;
			return {
				x: localX,
				y: colBaseY[colIdx],
				z: localZ,
				density
			};
		},

		getOverheadCloudFactor(): number {
			return overheadCloudFactor;
		}
	};

	return field;
}
