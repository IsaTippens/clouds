/**
 * Procedural monochrome precipitation system.
 * Evaluates meteorological rain triggers & squall waves, renders wind-slanted
 * falling monochrome rain streaks from cloudBaseY to y = 0, and projects
 * expanding dotted/stippled halftone ground splash rings at y = 0.04.
 */

import * as THREE from 'three';
import type { CloudParams, ConcreteCloudType } from '$lib/types';
import type { RainColumnSample } from './generator';
import { clamp, createPrng, lerp, smoothstep } from './noise';

export interface RainStatus {
	rainActive: boolean;
	rainRateMmHr: number;
}

export type RainColumnSampler = (rand: () => number, minDensity?: number) => RainColumnSample | null;

export interface RainSystem {
	update(
		dt: number,
		elapsed: number,
		params: CloudParams,
		cloudBaseY: number,
		dominantType: ConcreteCloudType,
		sampleRainColumn?: RainColumnSampler,
		overheadCloudFactor?: number
	): RainStatus;
	dispose(): void;
}

const MAX_DROPS = 1100;
const TOTAL_DROP_VERTS = MAX_DROPS * 2;
const MAX_SPLASHES = 96;
const GROUND_SPLASH_Y = 0.04;
const MAX_GROUND_RADIUS = 13.8;

interface RainDrop {
	active: boolean;
	x: number;
	y: number;
	z: number;
	spawnY: number;
	speed: number;
	length: number;
	alpha: number;
	jitterX: number;
	jitterZ: number;
}

interface SplashRing {
	active: boolean;
	x: number;
	z: number;
	age: number;
	duration: number;
	maxRadius: number;
	intensity: number;
	seed: number;
}

const RAIN_VERTEX_SHADER = /* glsl */ `
attribute float aEnd;
attribute float aAlpha;

varying float vEnd;
varying float vAlpha;

void main() {
	vEnd = aEnd;
	vAlpha = aAlpha;
	gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
}
`;

const RAIN_FRAGMENT_SHADER = /* glsl */ `
uniform float uDarkMode;
uniform float uGlobalStrength;

varying float vEnd;
varying float vAlpha;

void main() {
	if (vAlpha <= 0.005 || uGlobalStrength <= 0.005) discard;

	// Taper softly at upper tail (0) and sharpen at falling tip (1)
	float taper = smoothstep(0.0, 0.85, vEnd);
	float alpha = vAlpha * taper * uGlobalStrength * 0.78;
	if (alpha <= 0.01) discard;

	float lum = uDarkMode < 0.5
		? mix(0.36, 0.06, vEnd)
		: mix(0.42, 0.96, vEnd);

	gl_FragColor = vec4(vec3(lum), alpha);
}
`;

const SPLASH_VERTEX_SHADER = /* glsl */ `
attribute vec4 aSplash; // x: ageNorm [0..1], y: intensity [0..1], z: seed [0..1], w: active (0 or 1)

varying vec2 vUv;
varying vec4 vSplash;

void main() {
	vUv = uv * 2.0 - 1.0;
	vSplash = aSplash;
	vec4 worldPos = modelMatrix * instanceMatrix * vec4(position, 1.0);
	gl_Position = projectionMatrix * viewMatrix * worldPos;
}
`;

const SPLASH_FRAGMENT_SHADER = /* glsl */ `
uniform float uDarkMode;

varying vec2 vUv;
varying vec4 vSplash;

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
	float isActive = vSplash.w;
	if (isActive < 0.5) discard;

	float age = clamp(vSplash.x, 0.0, 1.0);
	float intensity = vSplash.y;
	float seed = vSplash.z;

	float r = length(vUv);
	if (r > 1.0) discard;

	float ease = 1.0 - pow(1.0 - age, 2.4);
	float ringR = mix(0.14, 0.88, ease);
	float ringWidth = mix(0.16, 0.085, age);

	// Primary expanding ring + subtle inner echo ripple
	float primaryBand = 1.0 - smoothstep(ringWidth * 0.35, ringWidth, abs(r - ringR));
	float echoR = ringR * 0.56;
	float echoBand = (1.0 - smoothstep(0.03, 0.08, abs(r - echoR))) * (1.0 - smoothstep(0.15, 0.72, age)) * 0.55;

	// Brief central splat dot at initial impact
	float centerDot = (1.0 - smoothstep(0.06, 0.18, r)) * (1.0 - smoothstep(0.0, 0.28, age));

	// Angular stipple dot segmentation around the ring circumference (stencil halftone style)
	float angle = atan(vUv.y, vUv.x) + seed * 6.2831853;
	float dotCount = floor(mix(10.0, 18.0, seed));
	float sector = fract((angle / 6.2831853) * dotCount);
	float angularDist = abs(sector - 0.5) * 2.0;
	// Dots shrink slightly as the splash ring expands and dissipates
	float dotFill = mix(0.76, 0.42, age);
	float stippleMask = 1.0 - smoothstep(dotFill - 0.22, dotFill + 0.06, angularDist);

	float pattern = max(centerDot, (primaryBand + echoBand) * stippleMask);
	if (pattern <= 0.02) discard;

	// Ordered Bayer dither dissolve as ring fades out
	float fade = (1.0 - smoothstep(0.38, 1.0, age)) * smoothstep(0.0, 0.06, age);
	float bayer = bayer4x4(gl_FragCoord.xy);
	float strength = pattern * fade * intensity;
	if (strength < bayer * 0.28) discard;

	float lum = uDarkMode < 0.5
		? mix(0.28, 0.06, clamp(strength, 0.0, 1.0))
		: mix(0.55, 0.96, clamp(strength, 0.0, 1.0));

	float alpha = clamp(strength * 0.85, 0.0, 0.88);
	gl_FragColor = vec4(vec3(lum), alpha);
}
`;

export function createRainSystem(scene: THREE.Scene): RainSystem {
	const rand = createPrng(0x5241494e);
	const group = new THREE.Group();
	group.name = 'MonochromeRainSystem';

	// Rain drop pool
	const drops: RainDrop[] = [];
	for (let i = 0; i < MAX_DROPS; i++) {
		drops.push({
			active: false,
			x: 0,
			y: -1,
			z: 0,
			spawnY: 3.0,
			speed: 12.5 + rand() * 5.5,
			length: 0.52 + rand() * 0.58,
			alpha: 0.45 + rand() * 0.55,
			jitterX: (rand() - 0.5) * 0.8,
			jitterZ: (rand() - 0.5) * 0.8
		});
	}

	const rainPositions = new Float32Array(TOTAL_DROP_VERTS * 3);
	const rainEnd = new Float32Array(TOTAL_DROP_VERTS);
	const rainAlpha = new Float32Array(TOTAL_DROP_VERTS);

	for (let i = 0; i < MAX_DROPS; i++) {
		rainEnd[i * 2] = 0.0; // Upper tail
		rainEnd[i * 2 + 1] = 1.0; // Lower falling head
	}

	const rainGeo = new THREE.BufferGeometry();
	const rainPosAttr = new THREE.BufferAttribute(rainPositions, 3);
	rainPosAttr.setUsage(THREE.DynamicDrawUsage);
	const rainAlphaAttr = new THREE.BufferAttribute(rainAlpha, 1);
	rainAlphaAttr.setUsage(THREE.DynamicDrawUsage);

	rainGeo.setAttribute('position', rainPosAttr);
	rainGeo.setAttribute('aEnd', new THREE.BufferAttribute(rainEnd, 1));
	rainGeo.setAttribute('aAlpha', rainAlphaAttr);

	const rainUniforms = {
		uDarkMode: { value: 0 },
		uGlobalStrength: { value: 0 }
	};

	const rainMat = new THREE.ShaderMaterial({
		uniforms: rainUniforms,
		vertexShader: RAIN_VERTEX_SHADER,
		fragmentShader: RAIN_FRAGMENT_SHADER,
		transparent: true,
		depthWrite: false,
		depthTest: true
	});

	const rainMesh = new THREE.LineSegments(rainGeo, rainMat);
	rainMesh.frustumCulled = false;
	group.add(rainMesh);

	// Ground splash rings (InstancedMesh of horizontal quads at y = 0.04)
	const splashes: SplashRing[] = [];
	for (let i = 0; i < MAX_SPLASHES; i++) {
		splashes.push({
			active: false,
			x: 0,
			z: 0,
			age: 0,
			duration: 0.45,
			maxRadius: 0.55,
			intensity: 0,
			seed: rand()
		});
	}
	let splashCursor = 0;

	const splashGeo = new THREE.PlaneGeometry(1, 1);
	splashGeo.rotateX(-Math.PI / 2);

	const splashData = new Float32Array(MAX_SPLASHES * 4);
	const splashAttr = new THREE.InstancedBufferAttribute(splashData, 4);
	splashAttr.setUsage(THREE.DynamicDrawUsage);
	splashGeo.setAttribute('aSplash', splashAttr);

	const splashUniforms = {
		uDarkMode: { value: 0 }
	};

	const splashMat = new THREE.ShaderMaterial({
		uniforms: splashUniforms,
		vertexShader: SPLASH_VERTEX_SHADER,
		fragmentShader: SPLASH_FRAGMENT_SHADER,
		transparent: true,
		depthWrite: false,
		depthTest: true
	});

	const splashMesh = new THREE.InstancedMesh(splashGeo, splashMat, MAX_SPLASHES);
	splashMesh.frustumCulled = false;
	splashMesh.instanceMatrix.setUsage(THREE.DynamicDrawUsage);

	const dummyMatrix = new THREE.Matrix4();
	const dummyPos = new THREE.Vector3();
	const dummyQuat = new THREE.Quaternion();
	const dummyScale = new THREE.Vector3(1, 1, 1);

	for (let i = 0; i < MAX_SPLASHES; i++) {
		dummyPos.set(0, -10, 0);
		dummyScale.set(0.01, 1, 0.01);
		dummyMatrix.compose(dummyPos, dummyQuat, dummyScale);
		splashMesh.setMatrixAt(i, dummyMatrix);
	}
	splashMesh.instanceMatrix.needsUpdate = true;

	group.add(splashMesh);
	scene.add(group);

	let rainClock = 0;
	let currentStrength = 0;
	let seededInitialDrops = false;

	function spawnDrop(
		drop: RainDrop,
		cloudBaseY: number,
		sampleRainColumn: RainColumnSampler | undefined,
		initialFill: boolean
	): void {
		const col = sampleRainColumn ? sampleRainColumn(rand, 0.24) : null;
		if (!col) {
			// No dense cloud overhead: keep drop inactive so rain never falls from clear sky
			drop.active = false;
			drop.y = -1;
			return;
		}

		drop.active = true;
		drop.x = col.x;
		drop.z = col.z;

		const localBaseY = Math.max(1.4, col.y || cloudBaseY);
		drop.spawnY = localBaseY;
		if (initialFill) {
			drop.y = rand() * localBaseY;
		} else {
			drop.y = localBaseY + rand() * 0.25;
		}
		drop.speed = 11.5 + rand() * 6.0;
		drop.length = 0.5 + rand() * 0.62;
		// Stronger, darker rain curtains under thicker cloud cores
		const densityBoost = clamp((col.density - 0.2) / 0.75, 0.25, 1.0);
		drop.alpha = (0.38 + rand() * 0.62) * densityBoost;
	}

	function triggerSplash(x: number, z: number, strength: number): void {
		if (Math.hypot(x, z) > MAX_GROUND_RADIUS) return;
		const s = splashes[splashCursor];
		splashCursor = (splashCursor + 1) % MAX_SPLASHES;
		s.active = true;
		s.x = x;
		s.z = z;
		s.age = 0;
		s.duration = 0.34 + rand() * 0.28;
		s.maxRadius = lerp(0.42, 0.92, clamp(strength, 0, 1)) * (0.75 + rand() * 0.5);
		s.intensity = clamp(0.48 + strength * 0.52, 0.3, 1.0);
		s.seed = rand();
	}

	function computeTargetStrength(
		params: CloudParams,
		dominantType: ConcreteCloudType,
		clock: number
	): number {
		const mode = params.rainMode ?? 'auto';
		const intensity = clamp(params.rainIntensity ?? 0.65, 0, 1);

		if (mode === 'off') {
			return 0;
		}
		if (mode === 'on') {
			return Math.max(0.25, intensity);
		}

		// 'auto' meteorological trigger
		const isStorm = dominantType === 'cumulonimbus';
		const isSaturated = params.humidity >= 0.76 && params.coverage >= 0.55;
		if (!isStorm && !isSaturated) {
			return 0;
		}

		let meteoBase: number;
		if (isStorm) {
			const humBoost = clamp((params.humidity - 0.45) / 0.55, 0, 1);
			meteoBase = 0.68 + 0.32 * humBoost;
		} else {
			const humFactor = clamp((params.humidity - 0.76) / 0.24, 0, 1);
			const covFactor = clamp((params.coverage - 0.55) / 0.45, 0, 1);
			meteoBase = 0.36 + 0.44 * humFactor + 0.2 * covFactor;
		}

		// Slow passing squall waves so rain naturally ebbs and surges
		const squallWave =
			0.62 +
			0.26 * Math.sin(clock * 0.22 + 0.6) +
			0.14 * Math.sin(clock * 0.57 + 2.15);

		return clamp(meteoBase * clamp(squallWave, 0.16, 1.05) * intensity, 0, 1);
	}

	function update(
		dt: number,
		_elapsed: number,
		params: CloudParams,
		cloudBaseY: number,
		dominantType: ConcreteCloudType,
		sampleRainColumn?: RainColumnSampler,
		overheadCloudFactor = 1.0
	): RainStatus {
		const isDark = params.surfaceMode === 'dark';
		const darkVal = isDark ? 1.0 : 0.0;
		rainUniforms.uDarkMode.value = darkVal;
		splashUniforms.uDarkMode.value = darkVal;

		const simDt = params.paused ? 0 : dt * params.timeSpeed;
		rainClock += simDt;

		// Modulate rain strength by actual overhead cloud cover (zero rain under clear sky!)
		const cloudPresence = smoothstep(0.02, 0.28, overheadCloudFactor);
		const targetStrength = computeTargetStrength(params, dominantType, rainClock) * cloudPresence;

		// Smooth ramp for organic rain onset/dissipation; instant response when toggled while paused
		if (params.paused) {
			currentStrength = targetStrength;
		} else {
			const rate = targetStrength > currentStrength ? 1.6 : 1.15;
			currentStrength += (targetStrength - currentStrength) * Math.min(1, dt * rate);
			if (targetStrength === 0 && currentStrength < 0.005) {
				currentStrength = 0;
			}
		}

		const rainActive = currentStrength > 0.02 || targetStrength > 0.02;
		const effectiveStrength = rainActive ? clamp(currentStrength, 0, 1) : 0;
		rainUniforms.uGlobalStrength.value = effectiveStrength;

		// Realistic precipitation rate in mm/hr
		const maxMmHr = dominantType === 'cumulonimbus' ? 52.0 : 32.0;
		const rainRateMmHr = rainActive
			? Math.round((Math.pow(effectiveStrength, 1.45) * maxMmHr + 0.9 * effectiveStrength) * 10) / 10
			: 0;

		// Check if any splash rings are still finishing their dissipation
		let anySplashActive = false;
		for (let i = 0; i < MAX_SPLASHES; i++) {
			if (splashes[i].active) {
				anySplashActive = true;
				break;
			}
		}

		if (!rainActive && !anySplashActive) {
			group.visible = false;
			return { rainActive: false, rainRateMmHr: 0 };
		}

		group.visible = true;

		const windRad = (params.windDirection * Math.PI) / 180;
		const dirX = Math.sin(windRad);
		const dirZ = -Math.cos(windRad);
		const safeBaseY = Math.max(1.6, cloudBaseY);

		// Horizontal wind drift velocity for rain slant
		const windDriftScale = params.windSpeed * 0.24;
		const turbScale = params.turbulence * 1.4;

		const activeDrops = rainActive
			? Math.max(60, Math.round(MAX_DROPS * Math.pow(effectiveStrength, 0.82)))
			: 0;

		if (!seededInitialDrops && activeDrops > 0) {
			for (let i = 0; i < MAX_DROPS; i++) {
				spawnDrop(drops[i], safeBaseY, sampleRainColumn, true);
			}
			seededInitialDrops = true;
		}

		// Update rain streaks (only emitting vertices for drops spawned beneath real clouds)
		const splashChance = clamp(0.14 + effectiveStrength * 0.28, 0.1, 0.45);
		let writtenDrops = 0;
		for (let i = 0; i < activeDrops; i++) {
			const d = drops[i];
			const vx = dirX * windDriftScale + d.jitterX * turbScale;
			const vz = dirZ * windDriftScale + d.jitterZ * turbScale;
			const vy = d.speed * (0.85 + 0.35 * effectiveStrength);

			if (!d.active) {
				spawnDrop(d, safeBaseY, sampleRainColumn, true);
				if (!d.active) continue;
			}

			if (simDt > 0) {
				d.x += vx * simDt;
				d.z += vz * simDt;
				d.y -= vy * simDt;

				if (d.y <= 0) {
					if (rand() < splashChance) {
						triggerSplash(d.x, d.z, effectiveStrength);
					}
					spawnDrop(d, safeBaseY, sampleRainColumn, false);
					if (!d.active) continue;
				} else if (d.y > d.spawnY + 0.9) {
					d.y = d.spawnY + rand() * 0.35;
				}
			} else if (d.y < 0 || d.y > d.spawnY + 0.8) {
				spawnDrop(d, safeBaseY, sampleRainColumn, true);
				if (!d.active) continue;
			}

			// Compute slanted streak endpoints (tail -> head) bounded under this drop's local cloud base
			const streakTime = (d.length * (0.75 + 0.55 * effectiveStrength)) / Math.max(vy, 1.0);
			const headY = Math.max(0.02, d.y);
			const tailY = Math.min(d.spawnY + 0.25, headY + vy * streakTime);
			const actualDt = (tailY - headY) / Math.max(vy, 1.0);
			const tailX = d.x - vx * actualDt;
			const tailZ = d.z - vz * actualDt;

			const v0 = writtenDrops * 2;
			const v1 = v0 + 1;
			const p0 = v0 * 3;
			const p1 = v1 * 3;
			writtenDrops++;

			rainPositions[p0] = tailX;
			rainPositions[p0 + 1] = tailY;
			rainPositions[p0 + 2] = tailZ;
			rainPositions[p1] = d.x;
			rainPositions[p1 + 1] = headY;
			rainPositions[p1 + 2] = d.z;

			// Soft fade near local cloud base & radial domain edge
			const baseFade = 1.0 - smoothstep(d.spawnY - 0.2, d.spawnY + 0.3, headY);
			const groundFade = smoothstep(0.0, 0.25, headY);
			const radialFade = 1.0 - smoothstep(10.5, MAX_GROUND_RADIUS, Math.hypot(d.x, d.z));
			const dropAlpha = d.alpha * baseFade * radialFade;

			rainAlpha[v0] = dropAlpha * 0.25;
			rainAlpha[v1] = dropAlpha * (0.65 + 0.35 * groundFade);
		}

		rainGeo.setDrawRange(0, writtenDrops * 2);
		rainPosAttr.needsUpdate = true;
		rainAlphaAttr.needsUpdate = true;

		// Update ground splash rings
		let splashDirty = false;
		for (let i = 0; i < MAX_SPLASHES; i++) {
			const s = splashes[i];
			const idx = i * 4;
			if (!s.active) {
				if (splashData[idx + 3] !== 0) {
					splashData[idx + 3] = 0;
					splashDirty = true;
				}
				continue;
			}

			if (simDt > 0) {
				s.age += simDt / s.duration;
			}

			if (s.age >= 1.0) {
				s.active = false;
				splashData[idx] = 1.0;
				splashData[idx + 1] = 0.0;
				splashData[idx + 3] = 0.0;
				splashDirty = true;
				continue;
			}

			const diameter = s.maxRadius * 2.0;
			dummyPos.set(s.x, GROUND_SPLASH_Y, s.z);
			dummyScale.set(diameter, 1, diameter);
			dummyMatrix.compose(dummyPos, dummyQuat, dummyScale);
			splashMesh.setMatrixAt(i, dummyMatrix);

			splashData[idx] = s.age;
			splashData[idx + 1] = s.intensity;
			splashData[idx + 2] = s.seed;
			splashData[idx + 3] = 1.0;
			splashDirty = true;
		}

		if (splashDirty) {
			splashMesh.instanceMatrix.needsUpdate = true;
			splashAttr.needsUpdate = true;
		}

		return {
			rainActive,
			rainRateMmHr
		};
	}

	function dispose(): void {
		scene.remove(group);
		rainGeo.dispose();
		rainMat.dispose();
		splashGeo.dispose();
		splashMat.dispose();
	}

	return {
		update,
		dispose
	};
}
