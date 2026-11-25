/**
 * Pure monochrome (black, grey, white) GLSL shaders for:
 * 1. Splatted dot-gradient 3D cloud point field with real-time 3D volumetric sun self-shadowing,
 *    ordered Bayer dithering & continuous wind advection.
 * 2. Real-time 3D raymarched splatted dot-gradient halftone ground shadow synchronized 1:1
 *    with solar position, 3D cloud formation, and live wind drift.
 */

import * as THREE from 'three';

const VOLUME_SAMPLE_GLSL = /* glsl */ `
// Sample the 52x28x52 3D density volume packed as a 7x4 slice atlas (364x208)
// with half-texel tile inset (zero cross-tile bleed) and smooth trilinear interpolation.
float sampleSlice2D(sampler2D atlas, vec2 xzUv, float sliceIdx) {
	float col = mod(sliceIdx, 7.0);
	float row = floor(sliceIdx / 7.0);
	// Clamp inside tile by 0.5 texel (0.5 / 52.0) to prevent bilinear bleed across slices
	vec2 insetUv = clamp(xzUv, vec2(0.009615), vec2(0.990385));
	vec2 atlasUv = vec2((col + insetUv.x) / 7.0, (row + insetUv.y) / 4.0);
	return texture2D(atlas, atlasUv).r;
}

float sampleVolume3D(sampler2D atlas, vec3 uvw) {
	if (uvw.y < 0.0 || uvw.y > 1.0) return 0.0;
	float fy = clamp(uvw.y, 0.0, 0.9999) * 27.0;
	float s0 = floor(fy);
	float s1 = min(s0 + 1.0, 27.0);
	float fracY = fy - s0;
	float d0 = sampleSlice2D(atlas, uvw.xz, s0);
	float d1 = sampleSlice2D(atlas, uvw.xz, s1);
	return mix(d0, d1, fracY);
}
`;

const CLOUD_VERTEX_SHADER = /* glsl */ `
attribute float aDensity;
attribute float aHeightNorm;
attribute float aSeed;
attribute float aSize;
attribute float aShade;

uniform sampler2D uVolumeAtlas;
uniform float uTime;
uniform vec2 uWindOffset;
uniform float uTurbulence;
uniform float uHalfExtentXZ;
uniform float uPointSize;
uniform float uPixelRatio;
uniform vec3 uSunDir;
uniform float uSunElevNorm;
uniform float uBaseY;
uniform float uHeightY;

varying float vLight;
varying float vDensity;
varying float vSeed;
varying float vEdgeFade;

${VOLUME_SAMPLE_GLSL}

void main() {
	vec3 pos = position;

	// Continuous wind advection with periodic domain wrap across [-uHalfExtentXZ, +uHalfExtentXZ]
	float span = uHalfExtentXZ * 2.0;
	vec2 shiftedXZ = pos.xz + uWindOffset;
	pos.xz = mod(shiftedXZ + uHalfExtentXZ, span) - uHalfExtentXZ;

	// Organic convective & turbulent micro-evolution
	float phase = aSeed * 6.2831853 + uTime * (0.45 + uTurbulence * 0.9);
	float swirlAmp = (0.06 + uTurbulence * 0.24) * (0.35 + aHeightNorm * 0.65);
	pos.x += sin(phase + pos.z * 0.35) * swirlAmp;
	pos.y += cos(phase * 0.85 + pos.x * 0.3) * swirlAmp * 0.65;
	pos.z += cos(phase * 1.15 - pos.y * 0.4) * swirlAmp;

	// Smooth radial boundary fade so wrapped points never pop at domain edges
	float radial = length(pos.xz) / uHalfExtentXZ;
	vEdgeFade = 1.0 - smoothstep(0.82, 1.02, radial);

	// Real-time 3D volumetric raymarch toward live uSunDir through the advected cloud volume
	vec3 sunDir = normalize(uSunDir);
	float stepWorld = max(uHeightY, 2.0) * 0.22;
	float opticalDepth = 0.0;
	for (int i = 1; i <= 6; i++) {
		vec3 samplePos = pos + sunDir * (float(i) * stepWorld);
		float sNy = (samplePos.y - uBaseY) / max(uHeightY, 0.01);
		if (sNy < 0.0 || sNy > 1.0) continue;
		float sRadial = length(samplePos.xz) / uHalfExtentXZ;
		if (sRadial >= 1.02) continue;
		float sEdge = 1.0 - smoothstep(0.82, 1.02, sRadial);
		vec2 unwrapped = mod(samplePos.xz - uWindOffset + uHalfExtentXZ, span) / span;
		float d = sampleVolume3D(uVolumeAtlas, vec3(unwrapped.x, sNy, unwrapped.y));
		opticalDepth += d * sEdge * (i <= 2 ? 0.58 : 0.40);
	}

	float liveTransmittance = exp(-opticalDepth * 1.45);
	float heightLift = 0.22 + 0.78 * pow(clamp(aHeightNorm, 0.0, 1.0), 0.72);
	float sunLift = mix(0.62, 1.06, uSunElevNorm);

	// Combine real-time directional transmittance with subtle ambient occlusion from aShade
	vLight = clamp((liveTransmittance * 0.72 + heightLift * 0.18 + aShade * 0.10) * sunLift, 0.02, 1.0);
	vDensity = aDensity;
	vSeed = aSeed;

	vec4 mvPosition = modelViewMatrix * vec4(pos, 1.0);
	gl_Position = projectionMatrix * mvPosition;

	// Perspective-attenuated splatted dot diameter
	float dist = max(-mvPosition.z, 1.5);
	float sizePx = uPointSize * aSize * uPixelRatio * (34.0 / dist);
	gl_PointSize = clamp(sizePx * (0.75 + 0.25 * vEdgeFade), 1.0, 26.0);
}
`;

const CLOUD_FRAGMENT_SHADER = /* glsl */ `
uniform float uShadowDither;
uniform float uDarkMode;

varying float vLight;
varying float vDensity;
varying float vSeed;
varying float vEdgeFade;

// 4x4 Bayer ordered dithering threshold in [0, 1]
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
	if (vEdgeFade <= 0.01) discard;

	vec2 uv = gl_PointCoord * 2.0 - 1.0;
	float r = length(uv);
	if (r > 1.0) discard;

	float bayer = bayer4x4(gl_FragCoord.xy);
	float stipple = fract(bayer + vSeed * 0.35);

	// Ordered dot-gradient stipple dropout on wispy edges & shadowed transitions
	float ditherStrength = uShadowDither;
	float shadowFactor = 1.0 - vLight;

	// Splatted dot radius shrinks crisply in stippled shadow-edge transitions
	float dotRadius = mix(0.96, 0.56 + 0.42 * vDensity, ditherStrength * 0.55);
	float alphaMask = 1.0 - smoothstep(dotRadius - 0.22, dotRadius, r);
	if (alphaMask <= 0.02) discard;

	// Stochastic ordered dither cut for airy low-density wisps
	float visibility = vEdgeFade * (0.28 + 0.72 * vDensity);
	if (visibility < stipple * (0.35 + ditherStrength * 0.42)) {
		discard;
	}

	// Quantized monochrome dot-gradient luminance (strictly black, grey, white)
	float ditheredLight = clamp(vLight + (bayer - 0.5) * (0.14 + ditherStrength * 0.34), 0.0, 1.0);
	// Subtle radial splat core-to-rim shading for tactile ink-dot depth
	float radialGrad = 1.0 - r * r * (0.16 + ditherStrength * 0.14);

	float lum;
	if (uDarkMode < 0.5) {
		// LIGHT MODE (#f2f2f0 paper):
		// Deep undersides -> crisp ink black/charcoal; midtones -> grey; crowns -> bright white with subtle rim
		float tone = smoothstep(0.08, 0.94, ditheredLight);
		// Quantize lightly into halftone steps when shadowDither is high
		float stepped = mix(tone, floor(tone * 5.0 + 0.5) / 5.0, ditherStrength * 0.42);
		lum = mix(0.06, 0.98, stepped);
		// Give bright crown dots a whisper of silver-grey rim so they sculpt cleanly against #f2f2f0
		float rimDarken = smoothstep(0.55, 0.95, r) * smoothstep(0.7, 1.0, lum) * 0.16;
		lum = clamp(lum - rimDarken, 0.04, 0.99);
	} else {
		// DARK MODE (#0a0a0a obsidian):
		// Shadowed undersides -> dark charcoal; midtones -> silver-grey; sunlit crowns -> crisp pure white
		float tone = smoothstep(0.06, 0.92, ditheredLight);
		float stepped = mix(tone, floor(tone * 5.0 + 0.5) / 5.0, ditherStrength * 0.42);
		lum = mix(0.14, 0.98, stepped) * mix(0.88, 1.0, radialGrad);
		lum = clamp(lum, 0.10, 1.0);
	}

	// Crisp dot alpha
	float alpha = alphaMask * vEdgeFade * mix(0.76, 0.98, clamp(vDensity + shadowFactor * 0.35, 0.0, 1.0));
	gl_FragColor = vec4(vec3(lum), alpha);
}
`;

const GROUND_SHADOW_VERTEX_SHADER = /* glsl */ `
varying vec2 vWorldXZ;

void main() {
	vec4 worldPos = modelMatrix * vec4(position, 1.0);
	vWorldXZ = worldPos.xz;
	gl_Position = projectionMatrix * viewMatrix * worldPos;
}
`;

const GROUND_SHADOW_FRAGMENT_SHADER = /* glsl */ `
uniform sampler2D uVolumeAtlas;
uniform vec2 uWindOffset;
uniform float uHalfExtentXZ;
uniform vec3 uSunDir;
uniform float uBaseY;
uniform float uHeightY;
uniform float uShadowDither;
uniform float uDarkMode;
uniform float uSunElevNorm;
uniform float uSunShadowStrength;

varying vec2 vWorldXZ;

${VOLUME_SAMPLE_GLSL}

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
	if (uSunShadowStrength <= 0.005) discard;

	// Large circular horizon plate fade (never clips into a square)
	float plateR = length(vWorldXZ) / 36.0;
	float plateMask = 1.0 - smoothstep(0.76, 1.0, plateR);
	if (plateMask <= 0.005) discard;

	vec3 sunDir = normalize(uSunDir);
	float sunY = max(sunDir.y, 0.12);

	// Analytical intersection of the 3D sun ray from ground (y=0) with cloud slab [uBaseY, uBaseY + uHeightY]
	float tEnter = uBaseY / sunY;
	float tExit = (uBaseY + uHeightY) / sunY;
	float tSpan = max(tExit - tEnter, 0.1);

	const int STEPS = 18;
	float stepT = tSpan / float(STEPS);
	float spanXZ = uHalfExtentXZ * 2.0;

	// Ordered Bayer ray-start jitter eliminates slicing bands while keeping crisp halftone texture
	float bayer = bayer4x4(gl_FragCoord.xy);
	float tau = 0.0;

	for (int i = 0; i < STEPS; i++) {
		float t = tEnter + (float(i) + bayer) * stepT;
		vec3 p = vec3(vWorldXZ.x, 0.0, vWorldXZ.y) + sunDir * t;

		// Exact circular cloud boundary match (zero square clipping!)
		float radial = length(p.xz) / uHalfExtentXZ;
		if (radial >= 1.02) continue;

		float edgeFade = 1.0 - smoothstep(0.82, 1.02, radial);
		float ny = (p.y - uBaseY) / max(uHeightY, 0.01);

		// Undo live wind advection with exact periodic wrap so shadow tracks moving clouds 1:1
		vec2 unwrapped = mod(p.xz - uWindOffset + uHalfExtentXZ, spanXZ) / spanXZ;
		float d = sampleVolume3D(uVolumeAtlas, vec3(unwrapped.x, ny, unwrapped.y));
		tau += d * edgeFade;
	}

	// Scale optical depth by normalized path thickness through the cloud layer
	float thicknessFactor = clamp(uHeightY / 4.2, 0.55, 2.2);
	float opticalDepth = (tau / float(STEPS)) * 3.8 * thicknessFactor;
	float rawShadow = (1.0 - exp(-opticalDepth)) * plateMask;

	// Modulate shadow intensity smoothly by solar elevation & twilight/night fade
	float shadow = clamp(rawShadow * mix(0.62, 1.06, uSunElevNorm) * uSunShadowStrength, 0.0, 1.0);
	if (shadow < 0.022) discard;

	// Procedural splatted dot-gradient halftone lattice on the ground plane
	float spacing = 0.36;
	vec2 gridPos = vWorldXZ / spacing;
	vec2 cellId = floor(gridPos);
	vec2 cellUv = fract(gridPos) - 0.5;
	float dist = length(cellUv) * 2.0;

	// Subtle per-cell micro jitter & ordered Bayer modulation
	float cellHash = fract(sin(dot(cellId, vec2(127.1, 311.7))) * 43758.5453);

	// Dot radius expands smoothly with shadow optical depth
	float ditherMod = mix(0.0, (bayer - 0.5) * 0.24 + (cellHash - 0.5) * 0.14, uShadowDither);
	float targetRadius = clamp(pow(shadow, 0.72) * 1.08 + ditherMod, 0.0, 0.96);

	float aa = 0.14;
	float dotMask = 1.0 - smoothstep(targetRadius - aa, targetRadius + aa, dist);
	if (dotMask <= 0.01) discard;

	if (uDarkMode < 0.5) {
		// LIGHT MODE: crisp monochrome charcoal/black halftone dots over #f2f2f0
		float inkLum = mix(0.28, 0.05, shadow);
		float alpha = dotMask * smoothstep(0.02, 0.16, shadow) * 0.78;
		gl_FragColor = vec4(vec3(inkLum), alpha);
	} else {
		// DARK MODE: subtle monochrome silver-graphite halftone footprint on #0a0a0a
		float rim = smoothstep(0.05, 0.45, shadow) * (1.0 - smoothstep(0.55, 0.95, shadow));
		float lum = mix(0.26, 0.14, shadow) + rim * 0.16;
		float alpha = dotMask * smoothstep(0.03, 0.18, shadow) * 0.72;
		gl_FragColor = vec4(vec3(lum), alpha);
	}
}
`;

export interface CloudShaderUniforms {
	[uniform: string]: THREE.IUniform<unknown>;
	uVolumeAtlas: THREE.IUniform<THREE.DataTexture>;
	uTime: THREE.IUniform<number>;
	uWindOffset: THREE.IUniform<THREE.Vector2>;
	uTurbulence: THREE.IUniform<number>;
	uHalfExtentXZ: THREE.IUniform<number>;
	uPointSize: THREE.IUniform<number>;
	uPixelRatio: THREE.IUniform<number>;
	uSunDir: THREE.IUniform<THREE.Vector3>;
	uSunElevNorm: THREE.IUniform<number>;
	uBaseY: THREE.IUniform<number>;
	uHeightY: THREE.IUniform<number>;
	uShadowDither: THREE.IUniform<number>;
	uDarkMode: THREE.IUniform<number>;
}

export interface GroundShadowUniforms {
	[uniform: string]: THREE.IUniform<unknown>;
	uVolumeAtlas: THREE.IUniform<THREE.DataTexture>;
	uWindOffset: THREE.IUniform<THREE.Vector2>;
	uHalfExtentXZ: THREE.IUniform<number>;
	uSunDir: THREE.IUniform<THREE.Vector3>;
	uBaseY: THREE.IUniform<number>;
	uHeightY: THREE.IUniform<number>;
	uShadowDither: THREE.IUniform<number>;
	uDarkMode: THREE.IUniform<number>;
	uSunElevNorm: THREE.IUniform<number>;
	uSunShadowStrength: THREE.IUniform<number>;
}

export function createCloudPointMaterial(
	volumeAtlas: THREE.DataTexture,
	pixelRatio: number
): { material: THREE.ShaderMaterial; uniforms: CloudShaderUniforms } {
	const uniforms: CloudShaderUniforms = {
		uVolumeAtlas: { value: volumeAtlas },
		uTime: { value: 0 },
		uWindOffset: { value: new THREE.Vector2(0, 0) },
		uTurbulence: { value: 0.35 },
		uHalfExtentXZ: { value: 12.5 },
		uPointSize: { value: 1.6 },
		uPixelRatio: { value: pixelRatio },
		uSunDir: { value: new THREE.Vector3(0.4, 0.8, 0.4).normalize() },
		uSunElevNorm: { value: 0.7 },
		uBaseY: { value: 3.0 },
		uHeightY: { value: 4.0 },
		uShadowDither: { value: 0.65 },
		uDarkMode: { value: 0 }
	};

	const material = new THREE.ShaderMaterial({
		uniforms,
		vertexShader: CLOUD_VERTEX_SHADER,
		fragmentShader: CLOUD_FRAGMENT_SHADER,
		transparent: true,
		depthWrite: true,
		depthTest: true
	});

	return { material, uniforms };
}

export function createGroundShadowMaterial(
	volumeAtlas: THREE.DataTexture
): { material: THREE.ShaderMaterial; uniforms: GroundShadowUniforms } {
	const uniforms: GroundShadowUniforms = {
		uVolumeAtlas: { value: volumeAtlas },
		uWindOffset: { value: new THREE.Vector2(0, 0) },
		uHalfExtentXZ: { value: 12.5 },
		uSunDir: { value: new THREE.Vector3(0.4, 0.8, 0.4).normalize() },
		uBaseY: { value: 3.0 },
		uHeightY: { value: 4.0 },
		uShadowDither: { value: 0.65 },
		uDarkMode: { value: 0 },
		uSunElevNorm: { value: 0.7 },
		uSunShadowStrength: { value: 1.0 }
	};

	const material = new THREE.ShaderMaterial({
		uniforms,
		vertexShader: GROUND_SHADOW_VERTEX_SHADER,
		fragmentShader: GROUND_SHADOW_FRAGMENT_SHADER,
		transparent: true,
		depthWrite: false,
		depthTest: true
	});

	return { material, uniforms };
}
