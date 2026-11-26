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

	// Continuous upward helical draft along normalized column height [0..1]
	float riseRate = (0.14 + aSeed * 0.13) * (0.75 + 0.45 * uSpinSpeed);
	float hRaw = fract(aHeightNorm + uTime * riseRate);

	// Smooth descent from wall cloud (h=1) down to ground (h=0) during formation / rope-out
	float reach = smoothstep(0.02, 0.72, uFunnelStrength);
	float h = mix(1.0 - reach, 1.0, hRaw);
	float worldY = h * uTopY;

	// Funnel radius profile:
	// - tight intense vortex tip at ground (r ~ 0.35..0.55)
	// - graceful curved mid-column taper
	// - wide flaring wall-cloud collar at top (r ~ 2.4..3.2)
	float baseTipR = mix(0.34, 0.50, uFunnelStrength);
	float midTaper = pow(h, 1.75) * 0.88;
	float wallCollar = pow(smoothstep(0.56, 1.0, h), 2.15) * 1.82;
	float profileR = baseTipR + midTaper + wallCollar;

	// Narrow into a sinuous rope during formation and dissipation
	float ropeScale = mix(0.28, 1.0, smoothstep(0.06, 0.82, uFunnelStrength));
	profileR *= ropeScale;

	// Split particles into dense inner vortex core (aRadialJitter <= 0) & wispy outer sheath (> 0)
	float isSheath = smoothstep(0.0, 1.0, aRadialJitter);
	float coreThickness = mix(0.14, 0.45, h);
	float sheathOffset = isSheath * mix(0.24, 0.78, h) * ropeScale;
	float r = max(0.06, profileR + aRadialJitter * coreThickness + sheathOffset);

	// Fast cyclonic rotation: intense angular velocity near tight ground tip, slower at wide collar
	float omega = mix(8.6, 1.55, pow(h, 0.58)) * uSpinSpeed;
	// Braided helical twist along height + subtle suction sub-vortex ripples
	float angle = aAngle + uTime * omega + h * 5.2;
	float subVortex = sin(angle * 2.0 - uTime * 3.4 + h * 6.0) * 0.075 * (1.0 - h * 0.5);
	r += subVortex * ropeScale;

	// Organic sinuous S-curve sway along the funnel spine
	float swayEnv = sin(h * 3.14159265) * (0.65 + 0.35 * (1.0 - h));
	float ropeWhip = mix(1.45, 1.0, smoothstep(0.15, 0.75, uFunnelStrength));
	float amp = uSwayAmp * swayEnv * ropeWhip;
	float swayX = sin(h * 3.2 - uTime * 1.4) * amp + cos(h * 5.4 + uTime * 0.95) * amp * 0.32;
	float swayZ = cos(h * 2.9 - uTime * 1.25 + 1.15) * amp + sin(h * 4.8 - uTime * 1.05) * amp * 0.32;

	// Slight downwind tilt with altitude
	vec2 lean = uWindDir * (pow(h, 1.35) * uWindLean);

	vec3 pos;
	pos.x = uAnchorXZ.x + lean.x + swayX + cos(angle) * r;
	pos.y = worldY;
	pos.z = uAnchorXZ.y + lean.y + swayZ + sin(angle) * r;

	// Directional cylindrical shading so the monochrome column reads with sculptural 3D volume
	vec3 radialNorm = normalize(vec3(cos(angle), 0.25 * h, sin(angle)));
	vec3 keyDir = normalize(vec3(0.55, 0.52, 0.65));
	float cylShade = dot(radialNorm, keyDir) * 0.5 + 0.5;
	// Dense lower-mid core is deeper/higher contrast; top wall collar blends smoothly into cloud base
	float coreContrast = mix(0.22, 0.88, cylShade) * mix(0.82, 1.08, h);

	// Softly fade particles at the very bottom wrap and top wall-cloud merge
	float endFade = smoothstep(0.0, 0.04, hRaw) * (1.0 - smoothstep(0.93, 1.0, hRaw));
	float strengthFade = smoothstep(0.01, 0.28, uFunnelStrength);

	vHeight = h;
	vSheath = isSheath;
	vShade = clamp(coreContrast, 0.0, 1.0);
	vSeed = aSeed;
	vAlpha = endFade * strengthFade * mix(0.96, 0.62, isSheath);

	vec4 mvPosition = modelViewMatrix * vec4(pos, 1.0);
	gl_Position = projectionMatrix * mvPosition;

	float dist = max(-mvPosition.z, 1.5);
	float sizeScale = mix(0.76, 1.28, pow(h, 1.2)) * mix(1.0, 1.18, isSheath);
	float sizePx = uPointSize * aSize * sizeScale * uPixelRatio * (33.0 / dist);
	gl_PointSize = clamp(sizePx * (0.55 + 0.45 * strengthFade), 1.2, 24.0);
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

	// Ordered Bayer stipple dropout on airy outer sheath & upper collar edge
	float sheathCut = vAlpha * mix(1.0, 0.68, vSheath);
	if (sheathCut < stipple * (0.28 + uShadowDither * 0.38)) discard;

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

	// Continuous outward & upward whipping debris cycle
	float cycleSpeed = (0.38 + aSeed * 0.44) * (0.7 + 0.45 * uSpinSpeed);
	float life = fract(aHeightSeed + uTime * cycleSpeed);

	// Whip outward from tight touchdown base (0.35..0.85) to wide swirling collar (1.4..3.1)
	float startR = 0.34 + aRadiusNorm * 0.52;
	float endR = 1.35 + aRadiusNorm * 1.85;
	float r = mix(startR, endR, pow(life, 0.72));

	// Rise rapidly from y = 0.05 up to ~1.5 then arc slightly outward/downward
	float arc = sin(life * 3.14159265);
	float peakY = mix(0.45, 1.52, fract(aSeed * 7.13));
	float y = 0.05 + pow(arc, 0.85) * peakY * (0.45 + 0.55 * uTouchdown);

	// High-speed cyclonic spin near touchdown, slowing slightly as debris flings outward
	float omega = mix(9.4, 3.2, aRadiusNorm) * uSpinSpeed;
	float angle = aAngle + uTime * omega + life * 4.5;

	// Subtle turbulent vertical/radial jitter
	float flutter = sin(uTime * 5.2 + aSeed * 25.0) * 0.08;
	vec3 pos;
	pos.x = uAnchorXZ.x + cos(angle) * (r + flutter);
	pos.y = max(0.05, y + cos(uTime * 4.7 + aSeed * 19.0) * 0.04);
	pos.z = uAnchorXZ.y + sin(angle) * (r - flutter);

	float lifeFade = smoothstep(0.0, 0.14, life) * (1.0 - smoothstep(0.68, 1.0, life));

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
	let anchorX = 0;
	let anchorZ = 0;
	let targetAnchorX = 0;
	let targetAnchorZ = 0;
	let anchorTopY = 3.4;
	let reanchorCooldown = 0;

	function sampleAnchor(
		sampleRainColumn: RainColumnSampler | undefined,
		fallbackBaseY: number,
		snapImmediately: boolean
	): void {
		let foundX = 0;
		let foundZ = 0;
		let foundY = Math.max(2.3, fallbackBaseY + 0.3);
		let found = false;

		if (sampleRainColumn) {
			// Try up to 4 samples to prefer a dense overhead cloud column well inside the rim
			for (let attempt = 0; attempt < 4; attempt++) {
				const col = sampleRainColumn(rand, 0.25);
				if (!col) break;
				const r = Math.hypot(col.x, col.z);
				if (r <= MAX_ANCHOR_RADIUS - 0.8 || attempt === 3) {
					const scale = r > MAX_ANCHOR_RADIUS - 0.5 ? (MAX_ANCHOR_RADIUS - 0.5) / r : 1.0;
					foundX = col.x * scale;
					foundZ = col.z * scale;
					foundY = Math.max(2.3, (col.y || fallbackBaseY) + 0.35);
					found = true;
					break;
				}
			}
		}

		if (!found) {
			const theta = rand() * Math.PI * 2;
			const r = rand() * 2.4;
			foundX = Math.cos(theta) * r;
			foundZ = Math.sin(theta) * r;
		}

		targetAnchorX = foundX;
		targetAnchorZ = foundZ;
		if (snapImmediately) {
			anchorX = foundX;
			anchorZ = foundZ;
			anchorTopY = foundY;
		} else {
			anchorTopY = lerp(anchorTopY, foundY, 0.25);
		}
		hasAnchor = true;
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
		funnelStrength: 0
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

		const targetStrength = computeTargetStrength(
			params,
			dominantType,
			overheadCloudFactor,
			vortexClock
		);

		// Smooth descent / rope-out lifecycle (instant response when toggled while paused)
		if (params.paused) {
			funnelStrength = targetStrength;
		} else {
			const forming = targetStrength > funnelStrength;
			const rate = forming ? (mode === 'on' ? 1.85 : 0.72) : 0.62;
			funnelStrength += (targetStrength - funnelStrength) * Math.min(1, dt * rate);
			if (targetStrength === 0 && funnelStrength < 0.004) {
				funnelStrength = 0;
			}
		}

		const tornadoActive = funnelStrength > 0.015;

		if (!tornadoActive && targetStrength <= 0.005) {
			group.visible = false;
			hasAnchor = false;
			statusOut.tornadoActive = false;
			statusOut.funnelStrength = 0;
			return statusOut;
		}

		group.visible = true;

		// Anchor underneath a dense overhead cloud column on initial formation
		const safeBaseY = Math.max(2.2, cloudBaseY + 0.35);
		if (!hasAnchor) {
			sampleAnchor(sampleRainColumn, safeBaseY, true);
			reanchorCooldown = 10.0;
		}

		// Gently wander with the wind & smoothly re-anchor if drifting near the circular rim
		const windRad = (params.windDirection * Math.PI) / 180;
		const dirX = Math.sin(windRad);
		const dirZ = -Math.cos(windRad);

		if (simDt > 0) {
			const wanderSpeed = Math.max(0.14, params.windSpeed * 0.028);
			const meanderX = Math.sin(vortexClock * 0.29 + 0.5) * 0.16;
			const meanderZ = Math.cos(vortexClock * 0.23 - 0.9) * 0.16;

			targetAnchorX += (dirX * wanderSpeed + meanderX) * simDt;
			targetAnchorZ += (dirZ * wanderSpeed + meanderZ) * simDt;

			reanchorCooldown -= simDt;
			const distFromCenter = Math.hypot(targetAnchorX, targetAnchorZ);
			if (distFromCenter > MAX_ANCHOR_RADIUS || reanchorCooldown <= 0) {
				if (distFromCenter > MAX_ANCHOR_RADIUS) {
					sampleAnchor(sampleRainColumn, safeBaseY, false);
				} else if (sampleRainColumn) {
					// Subtle periodic pull toward an active dense cloud core
					const col = sampleRainColumn(rand, 0.25);
					if (col && Math.hypot(col.x, col.z) < MAX_ANCHOR_RADIUS - 0.8) {
						targetAnchorX = lerp(targetAnchorX, col.x, 0.35);
						targetAnchorZ = lerp(targetAnchorZ, col.z, 0.35);
					}
				}
				reanchorCooldown = 9.0 + rand() * 5.0;
			}

			// Smoothly steer live anchor toward target anchor (never teleports)
			const followAlpha = Math.min(1, simDt * 0.65);
			anchorX += (targetAnchorX - anchorX) * followAlpha;
			anchorZ += (targetAnchorZ - anchorZ) * followAlpha;
			anchorTopY += (safeBaseY - anchorTopY) * Math.min(1, simDt * 1.4);
		} else {
			anchorTopY = lerp(anchorTopY, safeBaseY, 0.15);
		}

		const isDark = params.surfaceMode === 'dark' ? 1.0 : 0.0;
		const dither = clamp(params.shadowDither ?? 0.65, 0, 1);
		const basePointSize = clamp(params.pointSize ?? 1.6, 0.7, 3.6);
		const spinSpeed = 0.75 + clamp(params.windSpeed / 40, 0, 1) * 0.55 + params.turbulence * 0.35;
		const swayAmp = 0.24 + params.turbulence * 0.28;
		const windLean = clamp(params.windSpeed / 40, 0, 1) * 0.72;
		const touchdown = smoothstep(0.32, 0.8, funnelStrength);

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
