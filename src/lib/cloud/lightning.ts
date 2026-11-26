/**
 * Procedural monochrome thunderstorm lightning system.
 * Evaluates meteorological storm triggers, generates midpoint-displaced branching
 * cloud-to-ground bolts (~70%) and intra-cloud sheet-lightning arcs (~30%) strictly
 * underneath dense overhead cloud columns, projects expanding stippled halftone
 * ground strike shockwave rings at y = 0.05, and exposes live sheet-lightning
 * flash intensity and world position for real-time 3D cloud illumination.
 */

import * as THREE from 'three';
import type { CloudParams, ConcreteCloudType } from '$lib/types';
import type { RainColumnSampler } from './rain';
import { clamp, createPrng, lerp } from './noise';

export interface LightningStatus {
	lightningActive: boolean;
	flashIntensity: number; // 0..1 live decay for cloud sheet-lightning bloom
	flashPos: [number, number, number]; // world position of current strike/flash
}

export interface LightningSystem {
	update(
		dt: number,
		elapsed: number,
		params: CloudParams,
		cloudBaseY: number,
		dominantType: ConcreteCloudType,
		sampleRainColumn?: RainColumnSampler,
		overheadCloudFactor?: number
	): LightningStatus;
	dispose(): void;
}

type LightningMode = 'auto' | 'on' | 'off';

interface BoltSegment {
	x0: number;
	y0: number;
	z0: number;
	x1: number;
	y1: number;
	z1: number;
	/** 1.0 = main trunk, 0.58 = primary fork, 0.32 = sub-twig */
	branchWeight: number;
	/** 0..1 normalized progress along trunk/branch for crisp tip taper */
	progress0: number;
	progress1: number;
}

interface ActiveBolt {
	active: boolean;
	isCloudToGround: boolean;
	age: number;
	duration: number;
	/** Peak sheet-lightning bloom multiplier (intra-cloud scatters brighter inside cloud) */
	sheetBoost: number;
	/** Random phase for double-flicker re-strike timing */
	restrikeTime: number;
	restrikePeak: number;
	flickerSeed: number;
	flashPos: [number, number, number];
	segments: BoltSegment[];
}

interface GroundShockwave {
	active: boolean;
	x: number;
	z: number;
	age: number;
	duration: number;
	maxRadius: number;
	intensity: number;
	seed: number;
}

const MAX_ACTIVE_BOLTS = 3;
const MAX_CORE_SEGMENTS = 240;
const MAX_CORE_VERTS = MAX_CORE_SEGMENTS * 2;
const MAX_HALO_SEGMENTS = 384;
const MAX_HALO_VERTS = MAX_HALO_SEGMENTS * 2;
const MAX_SHOCKWAVES = 6;
const GROUND_STRIKE_Y = 0.04;
const GROUND_SHOCKWAVE_Y = 0.05;
const MAX_GROUND_RADIUS = 13.4;

const BOLT_CORE_VERTEX_SHADER = /* glsl */ `
attribute float aBranch;
attribute float aIntensity;
attribute float aProgress;

varying float vBranch;
varying float vIntensity;
varying float vProgress;

void main() {
	vBranch = aBranch;
	vIntensity = aIntensity;
	vProgress = aProgress;
	gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
}
`;

const BOLT_CORE_FRAGMENT_SHADER = /* glsl */ `
uniform float uDarkMode;

varying float vBranch;
varying float vIntensity;
varying float vProgress;

void main() {
	if (vIntensity <= 0.01) discard;

	// Forks decay slightly faster than the hot central plasma trunk
	float branchCurve = mix(pow(vIntensity, 1.55), vIntensity, vBranch);
	float tipTaper = 1.0 - smoothstep(0.75, 1.0, vProgress) * (1.0 - vBranch) * 0.45;
	float alpha = clamp(branchCurve * tipTaper * mix(0.78, 1.0, vBranch), 0.0, 1.0);
	if (alpha <= 0.02) discard;

	float lum;
	if (uDarkMode < 0.5) {
		// LIGHT MODE (#f2f2f0 paper): razor-sharp stark black (#050505) core with dark charcoal forks
		lum = mix(0.22, 0.02, vBranch);
	} else {
		// DARK MODE (#0a0a0a obsidian): pure blazing white (#ffffff) core with bright silver forks
		lum = mix(0.76, 1.0, vBranch);
	}

	gl_FragColor = vec4(vec3(lum), alpha);
}
`;

const BOLT_HALO_FRAGMENT_SHADER = /* glsl */ `
uniform float uDarkMode;

varying float vBranch;
varying float vIntensity;
varying float vProgress;

void main() {
	if (vIntensity <= 0.01) discard;

	float branchCurve = mix(pow(vIntensity, 1.65), vIntensity, vBranch);
	float tipTaper = 1.0 - smoothstep(0.72, 1.0, vProgress) * (1.0 - vBranch) * 0.5;
	float strength = branchCurve * tipTaper;

	float lum;
	float alpha;
	if (uDarkMode < 0.5) {
		// LIGHT MODE: crisp pure-white flash aura flanking the stark black core so bolts pop on #f2f2f0
		lum = 1.0;
		alpha = clamp(strength * mix(0.45, 0.85, vBranch), 0.0, 0.88);
	} else {
		// DARK MODE: subtle monochrome silver-grey glow halo around the pure white core
		lum = mix(0.55, 0.88, vBranch);
		alpha = clamp(strength * mix(0.24, 0.48, vBranch), 0.0, 0.52);
	}

	if (alpha <= 0.015) discard;
	gl_FragColor = vec4(vec3(lum), alpha);
}
`;

const SHOCKWAVE_VERTEX_SHADER = /* glsl */ `
attribute vec4 aWave; // x: ageNorm [0..1], y: intensity [0..1], z: seed [0..1], w: active (0 or 1)

varying vec2 vUv;
varying vec4 vWave;

void main() {
	vUv = uv * 2.0 - 1.0;
	vWave = aWave;
	vec4 worldPos = modelMatrix * instanceMatrix * vec4(position, 1.0);
	gl_Position = projectionMatrix * viewMatrix * worldPos;
}
`;

const SHOCKWAVE_FRAGMENT_SHADER = /* glsl */ `
uniform float uDarkMode;

varying vec2 vUv;
varying vec4 vWave;

float bayer4x4(vec2 coord) {
	ivec2 p = ivec2(mod(floor(coord), 4.0));
	int idx = p.y * 4 + p.x;
	if (idx == 0) return 0.0 / 16.0;
	if (idx == 1) return 8.0 / 16.0;
	if (idx == 2) return 2.0 / 16.0;
	if (idx == 3) return 10.0 / 16.0;
	if (idx == 4) return 12.0 / 16.0;
	if (idx == 5) return 4.0 / 16.0;
	if (idx == 6) return 14.0 / 16.0;
	if (idx == 7) return 6.0 / 16.0;
	if (idx == 8) return 3.0 / 16.0;
	if (idx == 9) return 11.0 / 16.0;
	if (idx == 10) return 1.0 / 16.0;
	if (idx == 11) return 9.0 / 16.0;
	if (idx == 12) return 15.0 / 16.0;
	if (idx == 13) return 7.0 / 16.0;
	if (idx == 14) return 13.0 / 16.0;
	return 5.0 / 16.0;
}

void main() {
	if (vWave.w < 0.5) discard;

	float age = clamp(vWave.x, 0.0, 1.0);
	float intensity = vWave.y;
	float seed = vWave.z;

	float r = length(vUv);
	if (r > 1.0) discard;

	// Fast initial blast expansion easing out smoothly
	float ease = 1.0 - pow(1.0 - age, 3.0);
	float ringR = mix(0.10, 0.92, ease);
	float ringWidth = mix(0.14, 0.055, age);

	// Primary expanding shockwave ring + trailing inner echo ring
	float primaryBand = 1.0 - smoothstep(ringWidth * 0.3, ringWidth, abs(r - ringR));
	float echoR = ringR * 0.62;
	float echoBand = (1.0 - smoothstep(0.025, 0.07, abs(r - echoR))) * (1.0 - smoothstep(0.12, 0.72, age)) * 0.6;

	// Brief high-contrast impact core flash at strike point
	float centerFlash = (1.0 - smoothstep(0.04, 0.22, r)) * (1.0 - smoothstep(0.0, 0.34, age));

	// Angular stencil halftone stipple dots around the shockwave circumference
	float angle = atan(vUv.y, vUv.x) + seed * 6.2831853;
	float dotCount = floor(mix(22.0, 34.0, seed));
	float sector = fract((angle / 6.2831853) * dotCount);
	float angularDist = abs(sector - 0.5) * 2.0;
	float dotFill = mix(0.82, 0.38, age);
	float stippleMask = 1.0 - smoothstep(dotFill - 0.22, dotFill + 0.06, angularDist);

	// Secondary finer halftone dots for inner echo
	float innerSector = fract((angle / 6.2831853) * (dotCount * 0.5) + 0.25);
	float innerDist = abs(innerSector - 0.5) * 2.0;
	float innerStipple = 1.0 - smoothstep(dotFill - 0.24, dotFill + 0.05, innerDist);

	float pattern = max(centerFlash, max(primaryBand * stippleMask, echoBand * innerStipple));
	if (pattern <= 0.02) discard;

	// Ordered 4x4 Bayer dither dissolve as shockwave dissipates
	float fade = (1.0 - smoothstep(0.28, 1.0, age)) * smoothstep(0.0, 0.03, age);
	float bayer = bayer4x4(gl_FragCoord.xy);
	float strength = pattern * fade * intensity;
	if (strength < bayer * 0.3) discard;

	float lum;
	if (uDarkMode < 0.5) {
		// LIGHT MODE: crisp stark black (#050505) impact ring with white center hot-spot
		lum = centerFlash > 0.65 && age < 0.14
			? 1.0
			: mix(0.24, 0.02, clamp(strength, 0.0, 1.0));
	} else {
		// DARK MODE: pure white (#ffffff) shockwave fading to monochrome silver-grey
		lum = mix(0.55, 1.0, clamp(strength, 0.0, 1.0));
	}

	float alpha = clamp(strength * 0.92, 0.0, 0.94);
	gl_FragColor = vec4(vec3(lum), alpha);
}
`;

interface Vec3 {
	x: number;
	y: number;
	z: number;
}

/**
 * Generate a jagged 3D polyline via recursive midpoint displacement.
 * Displaces perpendicular to the segment axis, tapering near both endpoints.
 */
function buildJaggedPolyline(
	start: Vec3,
	end: Vec3,
	subdivisions: number,
	jitterScale: number,
	rand: () => number
): Vec3[] {
	let pts: Vec3[] = [
		{ x: start.x, y: start.y, z: start.z },
		{ x: end.x, y: end.y, z: end.z }
	];

	const totalLen = Math.hypot(end.x - start.x, end.y - start.y, end.z - start.z);
	let disp = Math.max(0.18, totalLen * jitterScale);

	for (let level = 0; level < subdivisions; level++) {
		const next: Vec3[] = [pts[0]];
		const segCount = pts.length - 1;
		for (let i = 0; i < segCount; i++) {
			const a = pts[i];
			const b = pts[i + 1];
			const mx = (a.x + b.x) * 0.5;
			const my = (a.y + b.y) * 0.5;
			const mz = (a.z + b.z) * 0.5;

			// Taper lateral jitter gently at the very start & end so anchors stay clean
			const tMid = (i + 0.5) / segCount;
			const endTaper = Math.sin(Math.PI * clamp(tMid, 0.08, 0.92));
			const amp = disp * (0.45 + 0.55 * endTaper);

			next.push({
				x: mx + (rand() * 2 - 1) * amp,
				y: my + (rand() * 2 - 1) * amp * 0.38,
				z: mz + (rand() * 2 - 1) * amp
			});
			next.push(b);
		}
		pts = next;
		disp *= 0.54;
	}

	return pts;
}

function appendPolylineSegments(
	out: BoltSegment[],
	pts: Vec3[],
	branchWeight: number,
	maxSegments: number
): void {
	const count = pts.length - 1;
	if (count <= 0) return;
	for (let i = 0; i < count; i++) {
		if (out.length >= maxSegments) return;
		const a = pts[i];
		const b = pts[i + 1];
		out.push({
			x0: a.x,
			y0: a.y,
			z0: a.z,
			x1: b.x,
			y1: b.y,
			z1: b.z,
			branchWeight,
			progress0: i / count,
			progress1: (i + 1) / count
		});
	}
}

/**
 * Evaluate realistic double-flicker lightning intensity envelope in [0..1].
 * Sharp initial return stroke, brief decay, quick secondary re-strike pulse, then fade.
 */
function evalDoubleFlickerEnvelope(
	ageNorm: number,
	restrikeTime: number,
	restrikePeak: number,
	flickerSeed: number
): number {
	if (ageNorm <= 0 || ageNorm >= 1) return 0;

	// Primary return stroke: instantaneous rise, crisp exponential falloff
	const rise = clamp(ageNorm / 0.045, 0, 1);
	const primary = rise * Math.exp(-Math.max(0, ageNorm - 0.045) * 6.8);

	// Secondary re-strike pulse
	const dt2 = ageNorm - restrikeTime;
	let secondary = 0;
	if (dt2 > -0.02) {
		const rise2 = clamp((dt2 + 0.02) / 0.04, 0, 1);
		const decay2 = dt2 > 0.02 ? Math.exp(-(dt2 - 0.02) * 5.4) : 1.0;
		secondary = restrikePeak * rise2 * decay2;
	}

	// Subtle high-frequency ionized arc micro-flicker
	const micro = 0.88 + 0.12 * Math.sin(ageNorm * 88.0 + flickerSeed * 6.2831853);
	const tailFade = 1.0 - clamp((ageNorm - 0.82) / 0.18, 0, 1);

	return clamp(Math.max(primary, secondary) * micro * tailFade, 0, 1);
}

/**
 * Smoother sheet-lightning bloom envelope for illuminating surrounding 3D cloud points.
 */
function evalSheetFlashEnvelope(
	ageNorm: number,
	restrikeTime: number,
	restrikePeak: number,
	sheetBoost: number
): number {
	if (ageNorm <= 0 || ageNorm >= 1) return 0;
	const rise = clamp(ageNorm / 0.06, 0, 1);
	const primary = rise * Math.exp(-Math.max(0, ageNorm - 0.06) * 4.4);

	const dt2 = ageNorm - restrikeTime;
	let secondary = 0;
	if (dt2 > -0.03) {
		const rise2 = clamp((dt2 + 0.03) / 0.055, 0, 1);
		const decay2 = dt2 > 0.025 ? Math.exp(-(dt2 - 0.025) * 3.8) : 1.0;
		secondary = restrikePeak * 0.92 * rise2 * decay2;
	}

	const tailFade = 1.0 - clamp((ageNorm - 0.78) / 0.22, 0, 1);
	return clamp(Math.max(primary, secondary) * sheetBoost * tailFade, 0, 1);
}

export function createLightningSystem(scene: THREE.Scene): LightningSystem {
	const rand = createPrng(0x4c494748); // 'LIGH'
	const group = new THREE.Group();
	group.name = 'MonochromeLightningSystem';

	// Core bolt LineSegments
	const corePositions = new Float32Array(MAX_CORE_VERTS * 3);
	const coreBranch = new Float32Array(MAX_CORE_VERTS);
	const coreIntensity = new Float32Array(MAX_CORE_VERTS);
	const coreProgress = new Float32Array(MAX_CORE_VERTS);

	const coreGeo = new THREE.BufferGeometry();
	const corePosAttr = new THREE.BufferAttribute(corePositions, 3);
	corePosAttr.setUsage(THREE.DynamicDrawUsage);
	const coreBranchAttr = new THREE.BufferAttribute(coreBranch, 1);
	coreBranchAttr.setUsage(THREE.DynamicDrawUsage);
	const coreIntensityAttr = new THREE.BufferAttribute(coreIntensity, 1);
	coreIntensityAttr.setUsage(THREE.DynamicDrawUsage);
	const coreProgressAttr = new THREE.BufferAttribute(coreProgress, 1);
	coreProgressAttr.setUsage(THREE.DynamicDrawUsage);

	coreGeo.setAttribute('position', corePosAttr);
	coreGeo.setAttribute('aBranch', coreBranchAttr);
	coreGeo.setAttribute('aIntensity', coreIntensityAttr);
	coreGeo.setAttribute('aProgress', coreProgressAttr);
	coreGeo.setDrawRange(0, 0);

	const boltUniforms = {
		uDarkMode: { value: 0 }
	};

	const coreMat = new THREE.ShaderMaterial({
		uniforms: boltUniforms,
		vertexShader: BOLT_CORE_VERTEX_SHADER,
		fragmentShader: BOLT_CORE_FRAGMENT_SHADER,
		transparent: true,
		depthWrite: false,
		depthTest: true
	});

	const coreMesh = new THREE.LineSegments(coreGeo, coreMat);
	coreMesh.frustumCulled = false;
	coreMesh.renderOrder = 3;
	group.add(coreMesh);

	// Secondary halo / contrast flanking LineSegments
	const haloPositions = new Float32Array(MAX_HALO_VERTS * 3);
	const haloBranch = new Float32Array(MAX_HALO_VERTS);
	const haloIntensity = new Float32Array(MAX_HALO_VERTS);
	const haloProgress = new Float32Array(MAX_HALO_VERTS);

	const haloGeo = new THREE.BufferGeometry();
	const haloPosAttr = new THREE.BufferAttribute(haloPositions, 3);
	haloPosAttr.setUsage(THREE.DynamicDrawUsage);
	const haloBranchAttr = new THREE.BufferAttribute(haloBranch, 1);
	haloBranchAttr.setUsage(THREE.DynamicDrawUsage);
	const haloIntensityAttr = new THREE.BufferAttribute(haloIntensity, 1);
	haloIntensityAttr.setUsage(THREE.DynamicDrawUsage);
	const haloProgressAttr = new THREE.BufferAttribute(haloProgress, 1);
	haloProgressAttr.setUsage(THREE.DynamicDrawUsage);

	haloGeo.setAttribute('position', haloPosAttr);
	haloGeo.setAttribute('aBranch', haloBranchAttr);
	haloGeo.setAttribute('aIntensity', haloIntensityAttr);
	haloGeo.setAttribute('aProgress', haloProgressAttr);
	haloGeo.setDrawRange(0, 0);

	const haloMat = new THREE.ShaderMaterial({
		uniforms: boltUniforms,
		vertexShader: BOLT_CORE_VERTEX_SHADER,
		fragmentShader: BOLT_HALO_FRAGMENT_SHADER,
		transparent: true,
		depthWrite: false,
		depthTest: true
	});

	const haloMesh = new THREE.LineSegments(haloGeo, haloMat);
	haloMesh.frustumCulled = false;
	haloMesh.renderOrder = 2;
	group.add(haloMesh);

	// Ground strike halftone shockwave rings (InstancedMesh at y = 0.05)
	const shockwaves: GroundShockwave[] = [];
	for (let i = 0; i < MAX_SHOCKWAVES; i++) {
		shockwaves.push({
			active: false,
			x: 0,
			z: 0,
			age: 0,
			duration: 0.65,
			maxRadius: 2.8,
			intensity: 1,
			seed: rand()
		});
	}
	let shockwaveCursor = 0;

	const shockwaveGeo = new THREE.PlaneGeometry(1, 1);
	shockwaveGeo.rotateX(-Math.PI / 2);

	const shockwaveData = new Float32Array(MAX_SHOCKWAVES * 4);
	const shockwaveAttr = new THREE.InstancedBufferAttribute(shockwaveData, 4);
	shockwaveAttr.setUsage(THREE.DynamicDrawUsage);
	shockwaveGeo.setAttribute('aWave', shockwaveAttr);

	const shockwaveUniforms = {
		uDarkMode: { value: 0 }
	};

	const shockwaveMat = new THREE.ShaderMaterial({
		uniforms: shockwaveUniforms,
		vertexShader: SHOCKWAVE_VERTEX_SHADER,
		fragmentShader: SHOCKWAVE_FRAGMENT_SHADER,
		transparent: true,
		depthWrite: false,
		depthTest: true
	});

	const shockwaveMesh = new THREE.InstancedMesh(shockwaveGeo, shockwaveMat, MAX_SHOCKWAVES);
	shockwaveMesh.frustumCulled = false;
	shockwaveMesh.renderOrder = 1;
	shockwaveMesh.instanceMatrix.setUsage(THREE.DynamicDrawUsage);

	const dummyMatrix = new THREE.Matrix4();
	const dummyPos = new THREE.Vector3();
	const dummyQuat = new THREE.Quaternion();
	const dummyScale = new THREE.Vector3(1, 1, 1);

	for (let i = 0; i < MAX_SHOCKWAVES; i++) {
		dummyPos.set(0, -10, 0);
		dummyScale.set(0.01, 1, 0.01);
		dummyMatrix.compose(dummyPos, dummyQuat, dummyScale);
		shockwaveMesh.setMatrixAt(i, dummyMatrix);
	}
	shockwaveMesh.instanceMatrix.needsUpdate = true;
	group.add(shockwaveMesh);

	scene.add(group);

	// Bolt pool
	const bolts: ActiveBolt[] = [];
	for (let i = 0; i < MAX_ACTIVE_BOLTS; i++) {
		bolts.push({
			active: false,
			isCloudToGround: true,
			age: 0,
			duration: 0.4,
			sheetBoost: 1.0,
			restrikeTime: 0.42,
			restrikePeak: 0.85,
			flickerSeed: rand(),
			flashPos: [0, 3.0, 0],
			segments: []
		});
	}
	let boltCursor = 0;

	let prevMode: LightningMode = 'off';
	let wasConditionMet = false;
	let nextStrikeTimer = 0.4;
	let lastFlashPos: [number, number, number] = [0, 3.5, 0];

	function triggerShockwave(x: number, z: number): void {
		if (Math.hypot(x, z) > MAX_GROUND_RADIUS) return;
		const w = shockwaves[shockwaveCursor];
		shockwaveCursor = (shockwaveCursor + 1) % MAX_SHOCKWAVES;
		w.active = true;
		w.x = x;
		w.z = z;
		w.age = 0;
		w.duration = 0.56 + rand() * 0.28;
		w.maxRadius = 2.2 + rand() * 1.45;
		w.intensity = 0.85 + rand() * 0.15;
		w.seed = rand();
	}

	function spawnBolt(
		params: CloudParams,
		cloudBaseY: number,
		sampleRainColumn?: RainColumnSampler
	): boolean {
		let colX = 0;
		let colY = Math.max(2.0, cloudBaseY);
		let colZ = 0;

		if (sampleRainColumn) {
			const col = sampleRainColumn(rand, 0.28);
			if (!col) {
				// Never fire lightning out of clear sky when sampler is provided
				return false;
			}
			colX = col.x;
			colY = Math.max(1.8, col.y || cloudBaseY);
			colZ = col.z;
		}

		// Find an inactive slot, or recycle oldest
		let slot = bolts[boltCursor];
		for (let i = 0; i < MAX_ACTIVE_BOLTS; i++) {
			if (!bolts[i].active) {
				slot = bolts[i];
				break;
			}
		}
		boltCursor = (boltCursor + 1) % MAX_ACTIVE_BOLTS;

		const isCloudToGround = rand() < 0.7;
		const segs: BoltSegment[] = [];
		const maxSegsPerBolt = Math.floor(MAX_CORE_SEGMENTS / MAX_ACTIVE_BOLTS);
		const jitterScale = 0.21 + clamp(params.turbulence, 0, 1) * 0.1;

		if (isCloudToGround) {
			const startY = colY + 0.8;
			const driftAngle = rand() * Math.PI * 2;
			const driftDist = 0.35 + rand() * 1.45;
			let gx = colX + Math.cos(driftAngle) * driftDist;
			let gz = colZ + Math.sin(driftAngle) * driftDist;

			const gRadius = Math.hypot(gx, gz);
			if (gRadius > MAX_GROUND_RADIUS) {
				const scale = MAX_GROUND_RADIUS / gRadius;
				gx *= scale;
				gz *= scale;
			}

			const trunkPts = buildJaggedPolyline(
				{ x: colX, y: startY, z: colZ },
				{ x: gx, y: GROUND_STRIKE_Y, z: gz },
				4,
				jitterScale,
				rand
			);
			appendPolylineSegments(segs, trunkPts, 1.0, maxSegsPerBolt);

			// 2..4 dramatic forked branches splitting off the main trunk
			const branchCount = 2 + Math.floor(rand() * 3);
			for (let b = 0; b < branchCount; b++) {
				const minIdx = Math.max(2, Math.floor(trunkPts.length * 0.16));
				const maxIdx = Math.min(trunkPts.length - 3, Math.floor(trunkPts.length * 0.76));
				const idx = minIdx + Math.floor(rand() * Math.max(1, maxIdx - minIdx + 1));
				const origin = trunkPts[idx];

				const angle = rand() * Math.PI * 2;
				const dropHeight = Math.max(0.45, (origin.y - GROUND_STRIKE_Y) * (0.35 + rand() * 0.35));
				const horizReach = 0.75 + rand() * 1.65;

				const endPt: Vec3 = {
					x: origin.x + Math.cos(angle) * horizReach,
					y: Math.max(GROUND_STRIKE_Y + 0.18, origin.y - dropHeight),
					z: origin.z + Math.sin(angle) * horizReach
				};

				const branchPts = buildJaggedPolyline(origin, endPt, 3, jitterScale * 1.08, rand);
				appendPolylineSegments(segs, branchPts, 0.58, maxSegsPerBolt);

				// Occasional crisp secondary sub-twig
				if (rand() < 0.65 && branchPts.length > 4) {
					const twigIdx = 2 + Math.floor(rand() * (branchPts.length - 4));
					const twigOrigin = branchPts[twigIdx];
					const twigAngle = angle + (rand() < 0.5 ? -1 : 1) * (0.45 + rand() * 0.65);
					const twigReach = 0.4 + rand() * 0.75;
					const twigEnd: Vec3 = {
						x: twigOrigin.x + Math.cos(twigAngle) * twigReach,
						y: Math.max(GROUND_STRIKE_Y + 0.12, twigOrigin.y - (0.35 + rand() * 0.6)),
						z: twigOrigin.z + Math.sin(twigAngle) * twigReach
					};
					const twigPts = buildJaggedPolyline(twigOrigin, twigEnd, 2, jitterScale * 1.12, rand);
					appendPolylineSegments(segs, twigPts, 0.32, maxSegsPerBolt);
				}
			}

			slot.flashPos = [colX, colY + 0.9, colZ];
			slot.sheetBoost = 0.88 + rand() * 0.14;
			triggerShockwave(gx, gz);
		} else {
			// Intra-cloud horizontal / diagonal sheet-lightning arc near cloudBaseY + 1.2
			const startY = Math.max(2.0, colY) + 0.95 + (rand() - 0.3) * 0.65;
			const startPt: Vec3 = { x: colX, y: startY, z: colZ };

			let endX = colX;
			let endY = Math.max(2.0, cloudBaseY) + 1.2 + (rand() - 0.5) * 1.2;
			let endZ = colZ;

			const secondCol = sampleRainColumn ? sampleRainColumn(rand, 0.22) : null;
			if (secondCol && Math.hypot(secondCol.x - colX, secondCol.z - colZ) > 2.2) {
				const dx = secondCol.x - colX;
				const dz = secondCol.z - colZ;
				const dist = Math.hypot(dx, dz);
				const clampedDist = Math.min(dist, 6.8);
				endX = colX + (dx / dist) * clampedDist;
				endZ = colZ + (dz / dist) * clampedDist;
				endY = Math.max(2.0, secondCol.y || cloudBaseY) + 1.25 + (rand() - 0.5) * 0.85;
			} else {
				const angle = rand() * Math.PI * 2;
				const span = 3.2 + rand() * 3.4;
				endX = colX + Math.cos(angle) * span;
				endZ = colZ + Math.sin(angle) * span;
				const rEnd = Math.hypot(endX, endZ);
				if (rEnd > MAX_GROUND_RADIUS * 0.9) {
					const s = (MAX_GROUND_RADIUS * 0.9) / rEnd;
					endX *= s;
					endZ *= s;
				}
			}

			const arcPts = buildJaggedPolyline(startPt, { x: endX, y: endY, z: endZ }, 4, jitterScale * 1.05, rand);
			appendPolylineSegments(segs, arcPts, 1.0, maxSegsPerBolt);

			// 2..3 intra-cloud sheet tendrils
			const tendrilCount = 2 + Math.floor(rand() * 2);
			for (let t = 0; t < tendrilCount; t++) {
				const idx = 2 + Math.floor(rand() * Math.max(1, arcPts.length - 5));
				const origin = arcPts[idx];
				const angle = rand() * Math.PI * 2;
				const reach = 1.2 + rand() * 1.8;
				const endPt: Vec3 = {
					x: origin.x + Math.cos(angle) * reach,
					y: Math.max(1.8, origin.y + (rand() - 0.45) * 1.1),
					z: origin.z + Math.sin(angle) * reach
				};
				const tendrilPts = buildJaggedPolyline(origin, endPt, 3, jitterScale * 1.15, rand);
				appendPolylineSegments(segs, tendrilPts, 0.55, maxSegsPerBolt);
			}

			slot.flashPos = [(colX + endX) * 0.5, (startY + endY) * 0.5, (colZ + endZ) * 0.5];
			slot.sheetBoost = 1.0;
		}

		slot.active = true;
		slot.isCloudToGround = isCloudToGround;
		slot.age = 0;
		slot.duration = 0.32 + rand() * 0.16;
		slot.restrikeTime = 0.38 + rand() * 0.14;
		slot.restrikePeak = 0.74 + rand() * 0.22;
		slot.flickerSeed = rand();
		slot.segments = segs;

		lastFlashPos = [slot.flashPos[0], slot.flashPos[1], slot.flashPos[2]];
		return true;
	}

	function scheduleNextStrike(mode: LightningMode): number {
		if (mode === 'on') {
			return lerp(1.1, 3.2, rand());
		}
		return lerp(1.8, 5.5, rand());
	}

	function update(
		dt: number,
		_elapsed: number,
		params: CloudParams,
		cloudBaseY: number,
		dominantType: ConcreteCloudType,
		sampleRainColumn?: RainColumnSampler,
		overheadCloudFactor = 1.0
	): LightningStatus {
		const mode: LightningMode = params.lightningMode ?? 'auto';
		const darkVal = params.surfaceMode === 'dark' ? 1.0 : 0.0;
		boltUniforms.uDarkMode.value = darkVal;
		shockwaveUniforms.uDarkMode.value = darkVal;

		const safeDt = Math.min(Math.max(dt, 0), 0.1);
		const simDt = params.paused ? 0 : safeDt * params.timeSpeed;

		if (mode === 'off') {
			prevMode = 'off';
			wasConditionMet = false;
			for (let i = 0; i < MAX_ACTIVE_BOLTS; i++) {
				bolts[i].active = false;
			}
			for (let i = 0; i < MAX_SHOCKWAVES; i++) {
				shockwaves[i].active = false;
				shockwaveData[i * 4 + 3] = 0;
			}
			coreGeo.setDrawRange(0, 0);
			haloGeo.setDrawRange(0, 0);
			shockwaveAttr.needsUpdate = true;
			group.visible = false;
			return {
				lightningActive: false,
				flashIntensity: 0,
				flashPos: [0, Math.max(2.0, cloudBaseY), 0]
			};
		}

		// Evaluate meteorological / manual activation
		let conditionMet = false;
		if (mode === 'on') {
			conditionMet = overheadCloudFactor > 0.04;
		} else {
			const stormReady =
				dominantType === 'cumulonimbus' ||
				(params.turbulence >= 0.72 && params.humidity >= 0.72);
			conditionMet = stormReady && overheadCloudFactor > 0.1;
		}

		// Fire an initial bolt promptly (~0.2..0.48s) when switching to 'on' so users see it immediately
		if (mode === 'on' && (prevMode !== 'on' || !wasConditionMet) && conditionMet) {
			nextStrikeTimer = Math.min(nextStrikeTimer, 0.18 + rand() * 0.28);
		} else if (mode === 'auto' && !wasConditionMet && conditionMet) {
			nextStrikeTimer = Math.min(nextStrikeTimer, 0.55 + rand() * 0.95);
		}

		prevMode = mode;
		wasConditionMet = conditionMet;

		// Advance strike countdown & spawn bolts
		if (conditionMet && simDt > 0) {
			nextStrikeTimer -= simDt;
			if (nextStrikeTimer <= 0) {
				const spawned = spawnBolt(params, cloudBaseY, sampleRainColumn);
				nextStrikeTimer = spawned ? scheduleNextStrike(mode) : 0.35;
			}
		}

		// Update active bolts & assemble LineSegments vertex buffers
		let coreVertIdx = 0;
		let haloVertIdx = 0;
		let maxSheetFlash = 0;
		let anyBoltActive = false;
		let activeFlashPos: [number, number, number] = lastFlashPos;

		for (let b = 0; b < MAX_ACTIVE_BOLTS; b++) {
			const bolt = bolts[b];
			if (!bolt.active) continue;

			if (simDt > 0) {
				bolt.age += simDt;
			}

			const ageNorm = bolt.age / bolt.duration;
			if (ageNorm >= 1.0) {
				bolt.active = false;
				continue;
			}

			anyBoltActive = true;
			const boltIntensity = evalDoubleFlickerEnvelope(
				ageNorm,
				bolt.restrikeTime,
				bolt.restrikePeak,
				bolt.flickerSeed
			);
			const sheetFlash = evalSheetFlashEnvelope(
				ageNorm,
				bolt.restrikeTime,
				bolt.restrikePeak,
				bolt.sheetBoost
			);

			if (sheetFlash > maxSheetFlash) {
				maxSheetFlash = sheetFlash;
				activeFlashPos = bolt.flashPos;
			}

			const segs = bolt.segments;
			for (let s = 0; s < segs.length; s++) {
				if (coreVertIdx + 2 > MAX_CORE_VERTS) break;
				const seg = segs[s];

				const p0 = coreVertIdx * 3;
				corePositions[p0] = seg.x0;
				corePositions[p0 + 1] = seg.y0;
				corePositions[p0 + 2] = seg.z0;
				corePositions[p0 + 3] = seg.x1;
				corePositions[p0 + 4] = seg.y1;
				corePositions[p0 + 5] = seg.z1;

				coreBranch[coreVertIdx] = seg.branchWeight;
				coreBranch[coreVertIdx + 1] = seg.branchWeight;
				coreIntensity[coreVertIdx] = boltIntensity;
				coreIntensity[coreVertIdx + 1] = boltIntensity;
				coreProgress[coreVertIdx] = seg.progress0;
				coreProgress[coreVertIdx + 1] = seg.progress1;
				coreVertIdx += 2;

				// Emit two slightly offset flanking lines on main trunk & primary forks for thickness/halo
				if (seg.branchWeight >= 0.55 && haloVertIdx + 4 <= MAX_HALO_VERTS) {
					const offset = seg.branchWeight > 0.8 ? 0.055 : 0.032;
					// Offset 1 (+X, +Z)
					let hp = haloVertIdx * 3;
					haloPositions[hp] = seg.x0 + offset;
					haloPositions[hp + 1] = seg.y0;
					haloPositions[hp + 2] = seg.z0 + offset * 0.6;
					haloPositions[hp + 3] = seg.x1 + offset;
					haloPositions[hp + 4] = seg.y1;
					haloPositions[hp + 5] = seg.z1 + offset * 0.6;
					haloBranch[haloVertIdx] = seg.branchWeight;
					haloBranch[haloVertIdx + 1] = seg.branchWeight;
					haloIntensity[haloVertIdx] = boltIntensity;
					haloIntensity[haloVertIdx + 1] = boltIntensity;
					haloProgress[haloVertIdx] = seg.progress0;
					haloProgress[haloVertIdx + 1] = seg.progress1;
					haloVertIdx += 2;

					// Offset 2 (-X, -Z)
					hp = haloVertIdx * 3;
					haloPositions[hp] = seg.x0 - offset;
					haloPositions[hp + 1] = seg.y0;
					haloPositions[hp + 2] = seg.z0 - offset * 0.6;
					haloPositions[hp + 3] = seg.x1 - offset;
					haloPositions[hp + 4] = seg.y1;
					haloPositions[hp + 5] = seg.z1 - offset * 0.6;
					haloBranch[haloVertIdx] = seg.branchWeight;
					haloBranch[haloVertIdx + 1] = seg.branchWeight;
					haloIntensity[haloVertIdx] = boltIntensity;
					haloIntensity[haloVertIdx + 1] = boltIntensity;
					haloProgress[haloVertIdx] = seg.progress0;
					haloProgress[haloVertIdx + 1] = seg.progress1;
					haloVertIdx += 2;
				}
			}
		}

		coreGeo.setDrawRange(0, coreVertIdx);
		if (coreVertIdx > 0) {
			corePosAttr.needsUpdate = true;
			coreBranchAttr.needsUpdate = true;
			coreIntensityAttr.needsUpdate = true;
			coreProgressAttr.needsUpdate = true;
		}

		haloGeo.setDrawRange(0, haloVertIdx);
		if (haloVertIdx > 0) {
			haloPosAttr.needsUpdate = true;
			haloBranchAttr.needsUpdate = true;
			haloIntensityAttr.needsUpdate = true;
			haloProgressAttr.needsUpdate = true;
		}

		// Update ground strike shockwave rings
		let anyShockwaveActive = false;
		let shockwaveDirty = false;
		for (let i = 0; i < MAX_SHOCKWAVES; i++) {
			const w = shockwaves[i];
			const idx = i * 4;
			if (!w.active) {
				if (shockwaveData[idx + 3] !== 0) {
					shockwaveData[idx + 3] = 0;
					shockwaveDirty = true;
				}
				continue;
			}

			if (simDt > 0) {
				w.age += simDt / w.duration;
			}

			if (w.age >= 1.0) {
				w.active = false;
				shockwaveData[idx] = 1.0;
				shockwaveData[idx + 1] = 0.0;
				shockwaveData[idx + 3] = 0.0;
				shockwaveDirty = true;
				continue;
			}

			anyShockwaveActive = true;
			const diameter = w.maxRadius * 2.0;
			dummyPos.set(w.x, GROUND_SHOCKWAVE_Y, w.z);
			dummyScale.set(diameter, 1, diameter);
			dummyMatrix.compose(dummyPos, dummyQuat, dummyScale);
			shockwaveMesh.setMatrixAt(i, dummyMatrix);

			shockwaveData[idx] = w.age;
			shockwaveData[idx + 1] = w.intensity;
			shockwaveData[idx + 2] = w.seed;
			shockwaveData[idx + 3] = 1.0;
			shockwaveDirty = true;
		}

		if (shockwaveDirty) {
			shockwaveMesh.instanceMatrix.needsUpdate = true;
			shockwaveAttr.needsUpdate = true;
		}

		const lightningActive = conditionMet || anyBoltActive;
		group.visible = anyBoltActive || anyShockwaveActive;

		return {
			lightningActive,
			flashIntensity: maxSheetFlash,
			flashPos: [activeFlashPos[0], activeFlashPos[1], activeFlashPos[2]]
		};
	}

	function dispose(): void {
		scene.remove(group);
		coreGeo.dispose();
		coreMat.dispose();
		haloGeo.dispose();
		haloMat.dispose();
		shockwaveGeo.dispose();
		shockwaveMat.dispose();
	}

	return {
		update,
		dispose
	};
}
