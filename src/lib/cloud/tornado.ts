/**
 * Procedural monochrome point-cloud tornado vortex system.
 * Renders:
 * 1. Swirling 3D point-cloud funnel (~5,500 splatted monochrome dots) with fast
 *    cyclonic rotation, upward helical draft, sinuous S-curve sway, wide flaring
 *    wall-cloud collar, and 4x4 Bayer ordered dithering.
 * 2. Swirling ground debris/dust collar (~600 splatted points) whipping outward
 *    and upward around the touchdown base (y = 0.05..1.5).
 * 3. Rotating procedural halftone dot-gradient spiral footprint on a horizontal
 *    ground quad at y = 0.045 (radius ~4.2) centered under the touchdown.
 *
 * Strictly monochrome (black, grey, white — zero chromatic colors).
 */

import * as THREE from 'three';
import type { CloudParams, ConcreteCloudType } from '$lib/types';
import type { RainColumnSampler } from './rain';
import { clamp, createPrng, lerp, smoothstep } from './noise';

export interface TornadoStatus {
	tornadoActive: boolean;
	funnelStrength: number; // 0..1
	anchorX: number;
	anchorZ: number;
	topY: number;
}

export interface TornadoSystem {
	update(
		dt: number,
		elapsed: number,
		params: CloudParams,
		cloudBaseY: number,
		dominantType: ConcreteCloudType,
		sampleRainColumn?: RainColumnSampler,
		overheadCloudFactor?: number
	): TornadoStatus;
	dispose(): void;
}

type TornadoMode = 'auto' | 'on' | 'off';

const FUNNEL_POINT_COUNT = 5500;
const DEBRIS_POINT_COUNT = 600;
const FOOTPRINT_Y = 0.045;
const FOOTPRINT_RADIUS = 4.2;
const MAX_ANCHOR_RADIUS = 8.4;

const BAYER_4X4_GLSL = /* glsl */ `
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
`;

// -----------------------------------------------------------------------------
// 1. Swirling 3D Point-Cloud Funnel Shaders
// -----------------------------------------------------------------------------
const FUNNEL_VERTEX_SHADER = /* glsl */ `
attribute float aHeightNorm;
attribute float aAngle;
attribute float aRadialJitter;
attribute float aSeed;
attribute float aSize;

uniform float uTime;
uniform float uFunnelStrength;
uniform float uTopY;
uniform vec2 uAnchorXZ;
uniform vec2 uWindDir;
uniform float uWindLean;
uniform float uSpinSpeed;
uniform float uSwayAmp;
uniform float uPointSize;
uniform float uPixelRatio;

varying float vHeight;
varying float vSheath;
varying float vShade;
varying float vSeed;
varying float vAlpha;

void main() {
	if (uFunnelStrength <= 0.004) {
		gl_Position = vec4(2.0, 2.0, 2.0, 1.0);
		gl_PointSize = 0.0;
		vHeight = 0.0;
		vSheath = 0.0;
		vShade = 0.0;
		vSeed = 0.0;
		vAlpha = 0.0;
		return;
	}

	// Smooth upward helical draft along the laminar vortex wall [0..1]
	float riseRate = (0.11 + aSeed * 0.06) * (0.72 + 0.38 * uSpinSpeed);
	float hRaw = fract(aHeightNorm + uTime * riseRate);

	// Staged tornadogenesis descent:
	// - Early (strength 0.05..0.50): funnel protrudes down from wall cloud while debris spins up below
	// - Mature (strength >= 0.62): reach = 1.0 -> complete unbroken funnel connecting wall cloud to ground (y = 0)
	float reach = smoothstep(0.05, 0.62, uFunnelStrength);
	float h = mix(1.0 - reach, 1.0, hRaw);
	float worldY = h * uTopY;

	// Complete sculptural supercell funnel profile:
	// - defined ground touchdown column (r ~ 0.30..0.48)
	// - bold, continuous, unbroken mid-column cone
	// - wide flaring lowered Wall Cloud pedestal merging into cloud base (h > 0.72)
	float baseTipR = mix(0.28, 0.46, uFunnelStrength);
	float midTaper = pow(h, 1.38) * 1.02;
	float wallCloudCollar = pow(smoothstep(0.72, 1.0, h), 1.8) * 1.75;
	float profileR = baseTipR + midTaper + wallCloudCollar;

	// Sinuous narrow rope during early formation & late rope-out; full sculpted cone when mature
	float ropeScale = mix(0.38, 1.0, smoothstep(0.08, 0.72, uFunnelStrength));
	profileR *= ropeScale;

	// Dense, cohesive laminar Vortex Wall surrounding a hollow low-pressure Core (Eye),
	// with a thin wispy outer condensation sheath
	float isSheath = smoothstep(0.30, 1.0, aRadialJitter);
	float wallThickness = mix(0.055, 0.16, h);
	float sheathOffset = isSheath * mix(0.12, 0.36, h) * ropeScale;
	float r = max(0.12, profileR + aRadialJitter * wallThickness + sheathOffset);

	// Readable, stately cyclonic rotation: faster near ground touchdown, slower at wall-cloud collar
	float omega = mix(4.2, 0.85, pow(h, 0.62)) * uSpinSpeed;
	// Smooth braided helical spiral + subtle orbiting multi-vortex suction ripples when mature
	float angle = aAngle + uTime * omega + h * 4.4;
	float matureMask = smoothstep(0.55, 0.95, uFunnelStrength);
	float suctionVortices = sin(angle * 2.0 - uTime * 2.2 + h * 5.0) * 0.055 * (1.0 - h * 0.55) * matureMask;
	r += suctionVortices * ropeScale;

	// Graceful elephant-trunk S-curve sway (slow, sculptural, never frantic)
	float swayEnv = sin(h * 3.14159265) * (0.65 + 0.35 * (1.0 - h));
	float ropeWhip = mix(1.3, 1.0, smoothstep(0.15, 0.72, uFunnelStrength));
	float amp = uSwayAmp * 0.68 * swayEnv * ropeWhip;
	float swayX = sin(h * 2.6 - uTime * 0.62) * amp + cos(h * 4.2 + uTime * 0.45) * amp * 0.2;
	float swayZ = cos(h * 2.4 - uTime * 0.56 + 1.15) * amp + sin(h * 3.8 - uTime * 0.48) * amp * 0.2;

	// Gentle downwind tilt with altitude (vertical wind shear signature)
	vec2 lean = uWindDir * (pow(h, 1.35) * uWindLean * 0.8);

	vec3 pos;
	pos.x = uAnchorXZ.x + lean.x + swayX + cos(angle) * r;
	pos.y = worldY;
	pos.z = uAnchorXZ.y + lean.y + swayZ + sin(angle) * r;

	// Directional cylindrical shading so the monochrome funnel reads as a smooth sculpted cone
	vec3 radialNorm = normalize(vec3(cos(angle), 0.22 * h, sin(angle)));
	vec3 keyDir = normalize(vec3(0.55, 0.52, 0.65));
	float cylShade = dot(radialNorm, keyDir) * 0.5 + 0.5;
	float coreContrast = mix(0.24, 0.92, cylShade) * mix(0.86, 1.06, h);

	// Keep the full column unbroken from y=0 to wall cloud (only feather the very top cloud merge)
	float endFade = smoothstep(0.0, 0.02, hRaw) * (1.0 - smoothstep(0.95, 1.0, hRaw));
	float strengthFade = smoothstep(0.02, 0.24, uFunnelStrength);

	vHeight = h;
	vSheath = isSheath;
	vShade = clamp(coreContrast, 0.0, 1.0);
	vSeed = aSeed;
	vAlpha = endFade * strengthFade * mix(0.98, 0.58, isSheath);

	vec4 mvPosition = modelViewMatrix * vec4(pos, 1.0);
	gl_Position = projectionMatrix * mvPosition;

	float dist = max(-mvPosition.z, 1.5);
	float sizeScale = mix(0.95, 1.32, pow(h, 1.1)) * mix(1.08, 1.18, isSheath);
	float sizePx = uPointSize * aSize * sizeScale * uPixelRatio * (35.0 / dist);
	gl_PointSize = clamp(sizePx * (0.62 + 0.38 * strengthFade), 1.6, 26.0);
}
`;

const FUNNEL_FRAGMENT_SHADER = /* glsl */ `
uniform float uDarkMode;
uniform float uShadowDither;

varying float vHeight;
varying float vSheath;
varying float vShade;
varying float vSeed;
varying float vAlpha;

${BAYER_4X4_GLSL}

void main() {
	if (vAlpha <= 0.01) discard;

	vec2 uv = gl_PointCoord * 2.0 - 1.0;
	float r = length(uv);
	if (r > 1.0) discard;

	float bayer = bayer4x4(gl_FragCoord.xy);
	float stipple = fract(bayer + vSeed * 0.41);

	// Crisp splatted circular dot mask
	float dotRadius = mix(0.95, 0.68, vSheath * (0.4 + 0.5 * uShadowDither));
	float dotMask = 1.0 - smoothstep(dotRadius - 0.24, dotRadius, r);
	if (dotMask <= 0.02) discard;

	// Keep the primary laminar Vortex Wall solid & unbroken; apply Bayer stipple dropout only to outer sheath
	if (vSheath > 0.25) {
		float sheathCut = vAlpha * mix(1.0, 0.64, vSheath);
		if (sheathCut < stipple * (0.24 + uShadowDither * 0.34)) discard;
	}

	float ditheredShade = clamp(vShade + (bayer - 0.5) * (0.16 + uShadowDither * 0.28), 0.0, 1.0);
	float stepped = mix(ditheredShade, floor(ditheredShade * 4.0 + 0.5) / 4.0, uShadowDither * 0.4);
	float radialGrad = 1.0 - r * r * 0.18;

	float lum;
	if (uDarkMode < 0.5) {
		// LIGHT MODE (#f2f2f0): charcoal/ink black core with mid-grey outer sheath
		float coreLum = mix(0.04, 0.28, stepped);
		float sheathLum = mix(0.24, 0.58, stepped);
		lum = mix(coreLum, sheathLum, vSheath);
		// Subtle lightening at top wall-cloud collar to blend into cloud undersides
		lum = mix(lum, lum + 0.12, smoothstep(0.75, 1.0, vHeight));
		lum = clamp(lum, 0.03, 0.72);
	} else {
		// DARK MODE (#0a0a0a): crisp white/silver core with graphite outer sheath
		float coreLum = mix(0.68, 0.98, stepped) * radialGrad;
		float sheathLum = mix(0.26, 0.56, stepped) * radialGrad;
		lum = mix(coreLum, sheathLum, vSheath);
		lum = clamp(lum, 0.18, 1.0);
	}

	float alpha = dotMask * vAlpha * mix(0.94, 0.72, vSheath);
	gl_FragColor = vec4(vec3(lum), alpha);
}
`;

// -----------------------------------------------------------------------------
// 2. Swirling Ground Debris & Dust Collar Shaders (~600 splat points)
// -----------------------------------------------------------------------------
const DEBRIS_VERTEX_SHADER = /* glsl */ `
attribute float aAngle;
attribute float aRadiusNorm;
attribute float aHeightSeed;
attribute float aSeed;
attribute float aSize;

uniform float uTime;
uniform float uTouchdown;
uniform vec2 uAnchorXZ;
uniform float uSpinSpeed;
uniform float uPointSize;
uniform float uPixelRatio;

varying float vLife;
varying float vSeed;
varying float vRadiusNorm;
varying float vAlpha;

void main() {
	if (uTouchdown <= 0.005) {
		gl_Position = vec4(2.0, 2.0, 2.0, 1.0);
		gl_PointSize = 0.0;
		vLife = 0.0;
		vSeed = 0.0;
		vRadiusNorm = 0.0;
		vAlpha = 0.0;
		return;
	}

	// Corner Flow (Inflow Layer) -> Swirling Ground Debris Collar:
	// Surface friction pulls air radially INWARD across the ground before turning sharply UPWARD
	// into the vortex wall, lifting a compact swirling debris cloud at ground contact.
	float cycleSpeed = (0.28 + aSeed * 0.26) * (0.72 + 0.38 * uSpinSpeed);
	float life = fract(aHeightSeed + uTime * cycleSpeed);

	// Phase 1 (0..0.42): low radial inflow rushing inward from r = 2.8 -> 0.42 along the ground
	// Phase 2 (0.42..1.0): sharp upward turn & helical lift into the swirling debris collar
	float inflowT = smoothstep(0.0, 0.45, life);
	float liftT = smoothstep(0.38, 1.0, life);

	float outerStartR = 1.65 + aRadiusNorm * 1.55;
	float cornerR = 0.32 + aRadiusNorm * 0.45;
	float loftEndR = 0.72 + aRadiusNorm * 0.95;
	float r = mix(mix(outerStartR, cornerR, inflowT), loftEndR, liftT);

	float peakY = mix(0.55, 1.45, fract(aSeed * 7.13));
	float y = 0.05 + pow(liftT, 1.15) * peakY * (0.45 + 0.55 * uTouchdown);

	// Smooth cyclonic spin accelerating as air converges into the corner flow
	float omega = mix(4.8, 1.8, aRadiusNorm) * uSpinSpeed;
	float angle = aAngle + uTime * omega + life * 3.6;

	vec3 pos;
	pos.x = uAnchorXZ.x + cos(angle) * r;
	pos.y = max(0.05, y);
	pos.z = uAnchorXZ.y + sin(angle) * r;

	float lifeFade = smoothstep(0.0, 0.14, life) * (1.0 - smoothstep(0.72, 1.0, life));

	vLife = life;
	vSeed = aSeed;
	vRadiusNorm = aRadiusNorm;
	vAlpha = lifeFade * uTouchdown;

	vec4 mvPosition = modelViewMatrix * vec4(pos, 1.0);
	gl_Position = projectionMatrix * mvPosition;

	float dist = max(-mvPosition.z, 1.5);
	float sizePx = uPointSize * aSize * uPixelRatio * (28.0 / dist);
	gl_PointSize = clamp(sizePx * (0.65 + 0.35 * lifeFade), 1.4, 18.0);
}
`;

const DEBRIS_FRAGMENT_SHADER = /* glsl */ `
uniform float uDarkMode;
uniform float uShadowDither;

varying float vLife;
varying float vSeed;
varying float vRadiusNorm;
varying float vAlpha;

${BAYER_4X4_GLSL}

void main() {
	if (vAlpha <= 0.01) discard;

	vec2 uv = gl_PointCoord * 2.0 - 1.0;
	float r = length(uv);
	if (r > 1.0) discard;

	float bayer = bayer4x4(gl_FragCoord.xy);
	float stipple = fract(bayer + vSeed * 0.53);
	if (vAlpha < stipple * (0.25 + uShadowDither * 0.35)) discard;

	float dotMask = 1.0 - smoothstep(0.62, 0.96, r);
	if (dotMask <= 0.02) discard;

	// Crisp monochrome contrast: darker/brighter flecks near inner base, softer dust further out
	float tone = mix(0.0, 1.0, vRadiusNorm * 0.65 + vLife * 0.35);
	float lum;
	if (uDarkMode < 0.5) {
		lum = mix(0.05, 0.42, clamp(tone, 0.0, 1.0));
	} else {
		lum = mix(0.92, 0.36, clamp(tone, 0.0, 1.0));
	}

	float alpha = dotMask * vAlpha * 0.88;
	gl_FragColor = vec4(vec3(lum), alpha);
}
`;

// -----------------------------------------------------------------------------
// 3. Concentric Halftone Vortex Footprint Shaders (y = 0.045, radius ~4.2)
// -----------------------------------------------------------------------------
const FOOTPRINT_VERTEX_SHADER = /* glsl */ `
varying vec2 vUv;

void main() {
	vUv = uv * 2.0 - 1.0;
	vec4 worldPos = modelMatrix * vec4(position, 1.0);
	gl_Position = projectionMatrix * viewMatrix * worldPos;
}
`;

const FOOTPRINT_FRAGMENT_SHADER = /* glsl */ `
uniform float uTime;
uniform float uTouchdown;
uniform float uDarkMode;
uniform float uShadowDither;
uniform float uSpinSpeed;

varying vec2 vUv;

${BAYER_4X4_GLSL}

void main() {
	if (uTouchdown <= 0.005) discard;

	float r = length(vUv);
	if (r > 1.0) discard;

	// Rotate local coordinates cyclonically for swirling inflow spiral
	float rot = -uTime * 1.45 * uSpinSpeed;
	float c = cos(rot);
	float s = sin(rot);
	vec2 p = vec2(vUv.x * c - vUv.y * s, vUv.x * s + vUv.y * c);

	float angle = atan(p.y, p.x);

	// 3-arm logarithmic spiral vortex bands + concentric inflow rings
	float logSpiral = sin(angle * 3.0 + log(r + 0.11) * 6.4 - uTime * 2.2 * uSpinSpeed) * 0.5 + 0.5;
	float innerSpiral = sin(angle * 5.0 + r * 14.0 - uTime * 3.6 * uSpinSpeed) * 0.5 + 0.5;
	float rings = sin(r * 22.0 + uTime * 4.2 * uSpinSpeed) * 0.5 + 0.5;

	// Dense central touchdown eye + tapered outer swirl footprint
	float eyeCore = 1.0 - smoothstep(0.06, 0.24, r);
	float radialEnv = (1.0 - smoothstep(0.18, 1.0, r));
	float spiralField = mix(logSpiral * 0.72 + rings * 0.28, innerSpiral, smoothstep(0.45, 0.08, r));

	float field = max(eyeCore * 0.92, spiralField * radialEnv * 1.15);
	field *= uTouchdown;
	if (field <= 0.03) discard;

	// Procedural halftone dot-gradient grid on the rotated ground footprint
	float cellCount = 26.0;
	vec2 gridPos = p * cellCount;
	vec2 cellId = floor(gridPos);
	vec2 cellUv = fract(gridPos) - 0.5;
	float dotDist = length(cellUv) * 2.0;

	float bayer = bayer4x4(gl_FragCoord.xy);
	float cellHash = fract(sin(dot(cellId, vec2(127.1, 311.7))) * 43758.5453);
	float ditherMod = mix(0.0, (bayer - 0.5) * 0.22 + (cellHash - 0.5) * 0.12, uShadowDither);

	// Dot radius scales smoothly with vortex spiral field strength
	float targetRadius = clamp(pow(field, 0.74) * 1.06 + ditherMod, 0.0, 0.94);
	float dotMask = 1.0 - smoothstep(targetRadius - 0.16, targetRadius + 0.14, dotDist);
	if (dotMask <= 0.01) discard;

	// Outer Bayer dissolve
	if (field < bayer * 0.22) discard;

	float lum;
	float alpha;
	if (uDarkMode < 0.5) {
		// LIGHT MODE: crisp charcoal/ink halftone spiral dots on #f2f2f0
		lum = mix(0.32, 0.04, clamp(field, 0.0, 1.0));
		alpha = dotMask * smoothstep(0.03, 0.22, field) * 0.82;
	} else {
		// DARK MODE: crisp silver-white halftone spiral dots on #0a0a0a
		lum = mix(0.36, 0.95, clamp(field, 0.0, 1.0));
		alpha = dotMask * smoothstep(0.03, 0.22, field) * 0.78;
	}

	gl_FragColor = vec4(vec3(lum), alpha);
}
`;

export function createTornadoSystem(scene: THREE.Scene): TornadoSystem {
	const rand = createPrng(0x544f524e); // 'TORN'
	const group = new THREE.Group();
	group.name = 'MonochromeTornadoSystem';
	group.visible = false;

	const initialPixelRatio =
		typeof window !== 'undefined' ? Math.min(window.devicePixelRatio || 1, 2) : 1;

	// -------------------------------------------------------------------------
	// 1. Swirling 3D Point-Cloud Funnel (~5,500 splatted monochrome dots)
	// -------------------------------------------------------------------------
	const funnelGeo = new THREE.BufferGeometry();
	const funnelPos = new Float32Array(FUNNEL_POINT_COUNT * 3);
	const funnelHeightNorm = new Float32Array(FUNNEL_POINT_COUNT);
	const funnelAngle = new Float32Array(FUNNEL_POINT_COUNT);
	const funnelRadialJitter = new Float32Array(FUNNEL_POINT_COUNT);
	const funnelSeed = new Float32Array(FUNNEL_POINT_COUNT);
	const funnelSize = new Float32Array(FUNNEL_POINT_COUNT);

	for (let i = 0; i < FUNNEL_POINT_COUNT; i++) {
		// Slightly denser point distribution in the tight lower-mid funnel and wall collar
		const u = rand();
		const hNorm = u < 0.72 ? Math.pow(rand(), 0.92) : 0.62 + rand() * 0.38;
		funnelHeightNorm[i] = clamp(hNorm, 0, 1);
		funnelAngle[i] = rand() * Math.PI * 2;
		// ~65% dense inner vortex core ([-1..0]), ~35% wispy outer sheath ([0..1])
		funnelRadialJitter[i] = rand() < 0.65 ? -rand() : rand();
		funnelSeed[i] = rand();
		funnelSize[i] = 0.65 + rand() * 0.95;
	}

	funnelGeo.setAttribute('position', new THREE.BufferAttribute(funnelPos, 3));
	funnelGeo.setAttribute('aHeightNorm', new THREE.BufferAttribute(funnelHeightNorm, 1));
	funnelGeo.setAttribute('aAngle', new THREE.BufferAttribute(funnelAngle, 1));
	funnelGeo.setAttribute('aRadialJitter', new THREE.BufferAttribute(funnelRadialJitter, 1));
	funnelGeo.setAttribute('aSeed', new THREE.BufferAttribute(funnelSeed, 1));
	funnelGeo.setAttribute('aSize', new THREE.BufferAttribute(funnelSize, 1));

	const funnelUniforms = {
		uTime: { value: 0 },
		uFunnelStrength: { value: 0 },
		uTopY: { value: 3.4 },
		uAnchorXZ: { value: new THREE.Vector2(0, 0) },
		uWindDir: { value: new THREE.Vector2(1, 0) },
		uWindLean: { value: 0.35 },
		uSpinSpeed: { value: 1.0 },
		uSwayAmp: { value: 0.34 },
		uPointSize: { value: 1.75 },
		uPixelRatio: { value: initialPixelRatio },
		uDarkMode: { value: 0 },
		uShadowDither: { value: 0.65 }
	};

	const funnelMat = new THREE.ShaderMaterial({
		uniforms: funnelUniforms,
		vertexShader: FUNNEL_VERTEX_SHADER,
		fragmentShader: FUNNEL_FRAGMENT_SHADER,
		transparent: true,
		depthWrite: false,
		depthTest: true
	});

	const funnelPoints = new THREE.Points(funnelGeo, funnelMat);
	funnelPoints.frustumCulled = false;
	funnelPoints.renderOrder = 4;
	group.add(funnelPoints);

	// -------------------------------------------------------------------------
	// 2. Swirling Ground Debris Collar (~600 splat points, y = 0.05..1.5)
	// -------------------------------------------------------------------------
	const debrisGeo = new THREE.BufferGeometry();
	const debrisPos = new Float32Array(DEBRIS_POINT_COUNT * 3);
	const debrisAngle = new Float32Array(DEBRIS_POINT_COUNT);
	const debrisRadiusNorm = new Float32Array(DEBRIS_POINT_COUNT);
	const debrisHeightSeed = new Float32Array(DEBRIS_POINT_COUNT);
	const debrisSeed = new Float32Array(DEBRIS_POINT_COUNT);
	const debrisSize = new Float32Array(DEBRIS_POINT_COUNT);

	for (let i = 0; i < DEBRIS_POINT_COUNT; i++) {
		debrisAngle[i] = rand() * Math.PI * 2;
		debrisRadiusNorm[i] = Math.pow(rand(), 0.85);
		debrisHeightSeed[i] = rand();
		debrisSeed[i] = rand();
		debrisSize[i] = 0.62 + rand() * 0.98;
	}

	debrisGeo.setAttribute('position', new THREE.BufferAttribute(debrisPos, 3));
	debrisGeo.setAttribute('aAngle', new THREE.BufferAttribute(debrisAngle, 1));
	debrisGeo.setAttribute('aRadiusNorm', new THREE.BufferAttribute(debrisRadiusNorm, 1));
	debrisGeo.setAttribute('aHeightSeed', new THREE.BufferAttribute(debrisHeightSeed, 1));
	debrisGeo.setAttribute('aSeed', new THREE.BufferAttribute(debrisSeed, 1));
	debrisGeo.setAttribute('aSize', new THREE.BufferAttribute(debrisSize, 1));

	const debrisUniforms = {
		uTime: { value: 0 },
		uTouchdown: { value: 0 },
		uAnchorXZ: { value: new THREE.Vector2(0, 0) },
		uSpinSpeed: { value: 1.0 },
		uPointSize: { value: 1.65 },
		uPixelRatio: { value: initialPixelRatio },
		uDarkMode: { value: 0 },
		uShadowDither: { value: 0.65 }
	};

	const debrisMat = new THREE.ShaderMaterial({
		uniforms: debrisUniforms,
		vertexShader: DEBRIS_VERTEX_SHADER,
		fragmentShader: DEBRIS_FRAGMENT_SHADER,
		transparent: true,
		depthWrite: false,
		depthTest: true
	});

	const debrisPoints = new THREE.Points(debrisGeo, debrisMat);
	debrisPoints.frustumCulled = false;
	debrisPoints.renderOrder = 5;
	group.add(debrisPoints);

	// -------------------------------------------------------------------------
	// 3. Concentric Halftone Vortex Footprint (y = 0.045, radius ~4.2)
	// -------------------------------------------------------------------------
	const footprintGeo = new THREE.PlaneGeometry(FOOTPRINT_RADIUS * 2, FOOTPRINT_RADIUS * 2);
	footprintGeo.rotateX(-Math.PI / 2);

	const footprintUniforms = {
		uTime: { value: 0 },
		uTouchdown: { value: 0 },
		uDarkMode: { value: 0 },
		uShadowDither: { value: 0.65 },
		uSpinSpeed: { value: 1.0 }
	};

	const footprintMat = new THREE.ShaderMaterial({
		uniforms: footprintUniforms,
		vertexShader: FOOTPRINT_VERTEX_SHADER,
		fragmentShader: FOOTPRINT_FRAGMENT_SHADER,
		transparent: true,
		depthWrite: false,
		depthTest: true
	});

	const footprintMesh = new THREE.Mesh(footprintGeo, footprintMat);
	footprintMesh.position.set(0, FOOTPRINT_Y, 0);
	footprintMesh.frustumCulled = false;
	footprintMesh.renderOrder = 3;
	group.add(footprintMesh);

	scene.add(group);

	// -------------------------------------------------------------------------
	// Simulation State (zero per-frame allocations)
	// -------------------------------------------------------------------------
	let vortexClock = 0;
	let funnelStrength = 0;
	let hasAnchor = false;
	let ropingOut = false;
	let anchorX = 0;
	let anchorZ = 0;
	let anchorTopY = 3.4;

	function spawnUpwindAnchor(
		sampleRainColumn: RainColumnSampler | undefined,
		fallbackBaseY: number,
		dirX: number,
		dirZ: number,
		preferUpwind: boolean
	): void {
		let foundX = -dirX * (preferUpwind ? 4.8 : 1.2) + (rand() - 0.5) * 3.0;
		let foundZ = -dirZ * (preferUpwind ? 4.8 : 1.2) + (rand() - 0.5) * 3.0;
		let foundY = Math.max(2.6, fallbackBaseY + 0.85);

		if (sampleRainColumn) {
			// Pick the best dense cloud column, preferring upwind half when reforming so it travels cleanly across the sky
			let bestScore = -1e9;
			for (let attempt = 0; attempt < 8; attempt++) {
				const col = sampleRainColumn(rand, 0.26);
				if (!col) break;
				const r = Math.hypot(col.x, col.z);
				if (r > MAX_ANCHOR_RADIUS - 1.0) continue;
				const alongWind = col.x * dirX + col.z * dirZ;
				const upwindScore = preferUpwind ? -alongWind * 0.45 : -Math.abs(alongWind) * 0.25;
				const score = col.density * 2.0 + upwindScore;
				if (score > bestScore) {
					bestScore = score;
					foundX = col.x;
					foundZ = col.z;
					foundY = Math.max(2.6, (col.y || fallbackBaseY) + 0.85);
				}
			}
		}

		anchorX = foundX;
		anchorZ = foundZ;
		anchorTopY = foundY;
		hasAnchor = true;
		ropingOut = false;
	}

	function computeTargetStrength(
		params: CloudParams,
		dominantType: ConcreteCloudType,
		overheadCloudFactor: number,
		clock: number
	): number {
		const mode: TornadoMode =
			(params as CloudParams & { tornadoMode?: TornadoMode }).tornadoMode ?? 'auto';

		if (mode === 'off') {
			return 0;
		}
		if (mode === 'on') {
			return 1.0;
		}

		// 'auto': supercell cumulonimbus with high turbulence, strong wind, and overhead cloud cover
		const stormReady =
			dominantType === 'cumulonimbus' &&
			params.turbulence >= 0.78 &&
			params.windSpeed >= 18 &&
			overheadCloudFactor > 0.14;

		if (!stormReady) {
			return 0;
		}

		const turbBoost = clamp((params.turbulence - 0.78) / 0.22, 0, 1);
		const windBoost = clamp((params.windSpeed - 18) / 22, 0, 1);
		const cloudBoost = smoothstep(0.14, 0.42, overheadCloudFactor);

		// Gentle organic supercell intensity breathing
		const pulse = 0.9 + 0.1 * Math.sin(clock * 0.27 + 0.8);
		return clamp((0.76 + 0.14 * turbBoost + 0.1 * windBoost) * cloudBoost * pulse, 0, 1);
	}

	const statusOut: TornadoStatus = {
		tornadoActive: false,
		funnelStrength: 0,
		anchorX: 0,
		anchorZ: 0,
		topY: 3.5
	};

	function update(
		dt: number,
		_elapsed: number,
		params: CloudParams,
		cloudBaseY: number,
		dominantType: ConcreteCloudType,
		sampleRainColumn?: RainColumnSampler,
		overheadCloudFactor = 1.0
	): TornadoStatus {
		const mode: TornadoMode =
			(params as CloudParams & { tornadoMode?: TornadoMode }).tornadoMode ?? 'auto';
		const simDt = params.paused ? 0 : dt * params.timeSpeed;
		vortexClock += simDt;

		const windRad = (params.windDirection * Math.PI) / 180;
		const dirX = Math.sin(windRad);
		const dirZ = -Math.cos(windRad);
		const safeBaseY = Math.max(3.6, cloudBaseY + 0.55);

		const rawTargetStrength = computeTargetStrength(
			params,
			dominantType,
			overheadCloudFactor,
			vortexClock
		);

		// When a tornado drifts past the downwind rim, rope it out smoothly instead of rushing it across the ground!
		if (hasAnchor && Math.hypot(anchorX, anchorZ) > MAX_ANCHOR_RADIUS) {
			ropingOut = true;
		}

		const targetStrength = ropingOut ? 0 : rawTargetStrength;

		// Smooth formation / rope-out lifecycle (instant response when toggled while paused)
		if (params.paused) {
			funnelStrength = rawTargetStrength;
			ropingOut = false;
		} else {
			const forming = targetStrength > funnelStrength;
			const rate = forming ? (mode === 'on' ? 0.95 : 0.65) : 0.72;
			funnelStrength += (targetStrength - funnelStrength) * Math.min(1, dt * rate);
			if (targetStrength === 0 && funnelStrength < 0.008) {
				funnelStrength = 0;
				if (ropingOut) {
					// Fully roped out: silently re-anchor upwind under the next storm cell so a fresh tornado forms
					hasAnchor = false;
					ropingOut = false;
				}
			}
		}

		const tornadoActive = funnelStrength > 0.015;

		if (!tornadoActive && rawTargetStrength <= 0.005) {
			group.visible = false;
			hasAnchor = false;
			ropingOut = false;
			statusOut.tornadoActive = false;
			statusOut.funnelStrength = 0;
			return statusOut;
		}

		group.visible = true;

		// Spawn smoothly under a dense overhead storm core when forming a new tornado
		if (!hasAnchor) {
			spawnUpwindAnchor(sampleRainColumn, safeBaseY, dirX, dirZ, vortexClock > 1.0);
		}

		// Continuous natural ground travel locked smoothly to the parent supercell's wind drift
		// (NEVER rushes or jumps sideways to a new spot mid-life)
		if (simDt > 0) {
			const cloudDriftSpeed = Math.max(0.28, params.windSpeed * 0.052);
			const perpX = -dirZ;
			const perpZ = dirX;
			const lateralSway = Math.sin(vortexClock * 0.34 + 0.7) * 0.24;

			anchorX += (dirX * cloudDriftSpeed + perpX * lateralSway) * simDt;
			anchorZ += (dirZ * cloudDriftSpeed + perpZ * lateralSway) * simDt;
			anchorTopY += (safeBaseY - anchorTopY) * Math.min(1, simDt * 1.2);
		} else {
			anchorTopY = lerp(anchorTopY, safeBaseY, 0.15);
		}

		const isDark = params.surfaceMode === 'dark' ? 1.0 : 0.0;
		const dither = clamp(params.shadowDither ?? 0.65, 0, 1);
		const basePointSize = clamp(params.pointSize ?? 1.6, 0.7, 3.6);
		const spinSpeed = 0.75 + clamp(params.windSpeed / 40, 0, 1) * 0.55 + params.turbulence * 0.35;
		const swayAmp = 0.24 + params.turbulence * 0.28;
		const windLean = clamp(params.windSpeed / 40, 0, 1) * 0.72;
		// Per meteorological visibility principle: ground-contact rotation & debris cloud appear
		// slightly BEFORE the condensation funnel finishes descending to the surface
		const touchdown = smoothstep(0.14, 0.58, funnelStrength);

		// Update funnel uniforms
		funnelUniforms.uTime.value = vortexClock;
		funnelUniforms.uFunnelStrength.value = funnelStrength;
		funnelUniforms.uTopY.value = anchorTopY;
		funnelUniforms.uAnchorXZ.value.set(anchorX, anchorZ);
		funnelUniforms.uWindDir.value.set(dirX, dirZ);
		funnelUniforms.uWindLean.value = windLean;
		funnelUniforms.uSpinSpeed.value = spinSpeed;
		funnelUniforms.uSwayAmp.value = swayAmp;
		funnelUniforms.uPointSize.value = basePointSize * 1.08;
		funnelUniforms.uDarkMode.value = isDark;
		funnelUniforms.uShadowDither.value = dither;

		// Update ground debris collar uniforms
		debrisUniforms.uTime.value = vortexClock;
		debrisUniforms.uTouchdown.value = touchdown;
		debrisUniforms.uAnchorXZ.value.set(anchorX, anchorZ);
		debrisUniforms.uSpinSpeed.value = spinSpeed;
		debrisUniforms.uPointSize.value = basePointSize * 0.96;
		debrisUniforms.uDarkMode.value = isDark;
		debrisUniforms.uShadowDither.value = dither;

		// Update halftone vortex footprint mesh & uniforms
		footprintMesh.position.set(anchorX, FOOTPRINT_Y, anchorZ);
		footprintUniforms.uTime.value = vortexClock;
		footprintUniforms.uTouchdown.value = touchdown;
		footprintUniforms.uDarkMode.value = isDark;
		footprintUniforms.uShadowDither.value = dither;
		footprintUniforms.uSpinSpeed.value = spinSpeed;

		statusOut.tornadoActive = tornadoActive;
		statusOut.funnelStrength = funnelStrength;
		statusOut.anchorX = anchorX;
		statusOut.anchorZ = anchorZ;
		statusOut.topY = anchorTopY;
		return statusOut;
	}

	function dispose(): void {
		scene.remove(group);
		funnelGeo.dispose();
		funnelMat.dispose();
		debrisGeo.dispose();
		debrisMat.dispose();
		footprintGeo.dispose();
		footprintMat.dispose();
	}

	return {
		update,
		dispose
	};
}
