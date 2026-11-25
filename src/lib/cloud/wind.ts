/**
 * 3D monochrome wind current streamline & directional tracer visualization.
 * Renders curved multi-segment flow trails deflected by 3D curl turbulence,
 * complete with leading directional chevrons and stipple flow beads.
 */

import * as THREE from 'three';
import type { CloudParams } from '$lib/types';
import { clamp, createSeededNoise, lerp, smoothstep } from './noise';

export interface WindSystem {
	update(dt: number, elapsed: number, params: CloudParams): void;
	dispose(): void;
}

const MAX_STREAMLINES = 68;
const SEGMENTS_PER_LINE = 28;
const TRAIL_POINTS = SEGMENTS_PER_LINE + 1;
const VERTS_PER_LINE = SEGMENTS_PER_LINE * 2;
const TOTAL_LINE_VERTS = MAX_STREAMLINES * VERTS_PER_LINE;

// 2 short angled segments (4 vertices) forming a minimal directional chevron at each streamline head
const CHEVRON_VERTS_PER_LINE = 4;
const TOTAL_CHEVRON_VERTS = MAX_STREAMLINES * CHEVRON_VERTS_PER_LINE;

// 2 subtle stipple tracer beads riding along each streamline
const BEADS_PER_LINE = 2;
const TOTAL_BEADS = MAX_STREAMLINES * BEADS_PER_LINE;

const DOMAIN_HALF = 14.2;
const DOMAIN_SPAN = DOMAIN_HALF * 2;
const MIN_ALT_Y = 0.65;
const MAX_ALT_Y = 12.6;

interface StreamlineState {
	headX: number;
	headY: number;
	headZ: number;
	baseY: number;
	speedFactor: number;
	lengthFactor: number;
	phase: number;
	opacityWeight: number;
	// Snake-style static world-space path history (index 0 = oldest tail, TRAIL_POINTS-1 = newest behind head)
	histX: Float32Array;
	histY: Float32Array;
	histZ: Float32Array;
	/** Number of valid recorded history nodes (grows from 1..TRAIL_POINTS after a wrap/spawn) */
	filledCount: number;
	/** Distance head has traveled since the last recorded history node */
	distAccum: number;
	tangentX: number;
	tangentY: number;
	tangentZ: number;
}

const STREAMLINE_VERTEX_SHADER = /* glsl */ `
attribute float aProgress;
attribute float aAlpha;
attribute float aPhase;

uniform float uTime;
uniform float uSpeedNorm;

varying float vProgress;
varying float vAlpha;
varying float vPulse;

void main() {
	vProgress = aProgress;
	vAlpha = aAlpha;

	// Subtle traveling flow pulse from tail (0) to head (1)
	float waveFreq = 1.6 + uSpeedNorm * 1.4;
	float wave = sin((aProgress * 6.2831853 * 1.35) - uTime * waveFreq + aPhase);
	vPulse = 0.78 + 0.22 * wave;

	gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
}
`;

const STREAMLINE_FRAGMENT_SHADER = /* glsl */ `
uniform float uDarkMode;
uniform float uSpeedNorm;

varying float vProgress;
varying float vAlpha;
varying float vPulse;

void main() {
	if (vAlpha <= 0.005) discard;

	// Sleek head-to-tail alpha gradient: feathered tail -> crisp leading head
	float tailFade = smoothstep(0.0, 0.72, vProgress);
	float headBoost = 0.65 + 0.35 * smoothstep(0.68, 1.0, vProgress);
	float alpha = vAlpha * tailFade * headBoost * vPulse * mix(0.46, 0.76, uSpeedNorm);

	if (alpha <= 0.008) discard;

	// Strictly monochrome palette matching light (#f2f2f0) and dark (#0a0a0a) scenes
	float lum;
	if (uDarkMode < 0.5) {
		lum = mix(0.42, 0.06, smoothstep(0.1, 0.96, vProgress));
	} else {
		lum = mix(0.38, 0.96, smoothstep(0.1, 0.96, vProgress));
	}

	gl_FragColor = vec4(vec3(lum), alpha);
}
`;

const BEAD_VERTEX_SHADER = /* glsl */ `
attribute float aAlpha;
attribute float aSize;

uniform float uPixelRatio;

varying float vAlpha;

void main() {
	vAlpha = aAlpha;
	vec4 mvPosition = modelViewMatrix * vec4(position, 1.0);
	gl_Position = projectionMatrix * mvPosition;
	float dist = max(-mvPosition.z, 2.0);
	gl_PointSize = clamp(aSize * uPixelRatio * (30.0 / dist), 1.5, 9.0);
}
`;

const BEAD_FRAGMENT_SHADER = /* glsl */ `
uniform float uDarkMode;

varying float vAlpha;

void main() {
	if (vAlpha <= 0.01) discard;
	vec2 uv = gl_PointCoord * 2.0 - 1.0;
	float r = length(uv);
	if (r > 1.0) discard;

	float mask = 1.0 - smoothstep(0.55, 1.0, r);
	float lum = uDarkMode < 0.5 ? 0.08 : 0.94;
	gl_FragColor = vec4(vec3(lum), mask * vAlpha * 0.82);
}
`;

function wrapCoord(v: number): number {
	const wrapped = ((v + DOMAIN_HALF) % DOMAIN_SPAN + DOMAIN_SPAN) % DOMAIN_SPAN;
	return wrapped - DOMAIN_HALF;
}

export function createWindSystem(scene: THREE.Scene): WindSystem {
	const noise = createSeededNoise(0x57494e44);
	const group = new THREE.Group();
	group.name = 'WindCurrentSystem';

	// Deterministic streamline seeds stratified across altitude bands and horizontal plane
	const streamlines: StreamlineState[] = [];
	for (let i = 0; i < MAX_STREAMLINES; i++) {
		const layerNorm = (i + 0.5) / MAX_STREAMLINES;
		// Stratified altitude with mild jitter: denser in lower-mid troposphere where clouds sit
		const shapedAlt = Math.pow(layerNorm, 0.92);
		const baseY = clamp(
			lerp(MIN_ALT_Y, MAX_ALT_Y, shapedAlt) + (noise.rand() - 0.5) * 0.9,
			MIN_ALT_Y,
			MAX_ALT_Y
		);
		// Altitude wind shear: higher currents move slightly faster
		const altShear = lerp(0.76, 1.28, (baseY - MIN_ALT_Y) / (MAX_ALT_Y - MIN_ALT_Y));

		const hx = (noise.rand() * 2 - 1) * DOMAIN_HALF;
		const hz = (noise.rand() * 2 - 1) * DOMAIN_HALF;
		const histX = new Float32Array(TRAIL_POINTS);
		const histY = new Float32Array(TRAIL_POINTS);
		const histZ = new Float32Array(TRAIL_POINTS);
		histX.fill(hx);
		histY.fill(baseY);
		histZ.fill(hz);

		streamlines.push({
			headX: hx,
			headY: baseY,
			headZ: hz,
			baseY,
			speedFactor: altShear * (0.85 + noise.rand() * 0.3),
			lengthFactor: 0.78 + noise.rand() * 0.44,
			phase: noise.rand() * Math.PI * 2,
			opacityWeight: 0.65 + noise.rand() * 0.35,
			histX,
			histY,
			histZ,
			filledCount: 1,
			distAccum: 0,
			tangentX: 1,
			tangentY: 0,
			tangentZ: 0
		});
	}

	// Streamline LineSegments buffers
	const linePositions = new Float32Array(TOTAL_LINE_VERTS * 3);
	const lineProgress = new Float32Array(TOTAL_LINE_VERTS);
	const lineAlpha = new Float32Array(TOTAL_LINE_VERTS);
	const linePhase = new Float32Array(TOTAL_LINE_VERTS);

	// Pre-populate static progress and phase attributes
	for (let i = 0; i < MAX_STREAMLINES; i++) {
		const baseVert = i * VERTS_PER_LINE;
		const phase = streamlines[i].phase;
		for (let s = 0; s < SEGMENTS_PER_LINE; s++) {
			const p0 = s / SEGMENTS_PER_LINE;
			const p1 = (s + 1) / SEGMENTS_PER_LINE;
			const v0 = baseVert + s * 2;
			const v1 = v0 + 1;
			lineProgress[v0] = p0;
			lineProgress[v1] = p1;
			linePhase[v0] = phase;
			linePhase[v1] = phase;
		}
	}

	const lineGeo = new THREE.BufferGeometry();
	const linePosAttr = new THREE.BufferAttribute(linePositions, 3);
	linePosAttr.setUsage(THREE.DynamicDrawUsage);
	const lineAlphaAttr = new THREE.BufferAttribute(lineAlpha, 1);
	lineAlphaAttr.setUsage(THREE.DynamicDrawUsage);

	lineGeo.setAttribute('position', linePosAttr);
	lineGeo.setAttribute('aProgress', new THREE.BufferAttribute(lineProgress, 1));
	lineGeo.setAttribute('aAlpha', lineAlphaAttr);
	lineGeo.setAttribute('aPhase', new THREE.BufferAttribute(linePhase, 1));

	const sharedUniforms = {
		uTime: { value: 0 },
		uSpeedNorm: { value: 0.3 },
		uDarkMode: { value: 0 }
	};

	const lineMat = new THREE.ShaderMaterial({
		uniforms: sharedUniforms,
		vertexShader: STREAMLINE_VERTEX_SHADER,
		fragmentShader: STREAMLINE_FRAGMENT_SHADER,
		transparent: true,
		depthWrite: false,
		depthTest: true
	});

	const lineMesh = new THREE.LineSegments(lineGeo, lineMat);
	lineMesh.frustumCulled = false;
	group.add(lineMesh);

	// Directional chevron ticks at each streamline head
	const chevronPositions = new Float32Array(TOTAL_CHEVRON_VERTS * 3);
	const chevronProgress = new Float32Array(TOTAL_CHEVRON_VERTS);
	const chevronAlpha = new Float32Array(TOTAL_CHEVRON_VERTS);
	const chevronPhase = new Float32Array(TOTAL_CHEVRON_VERTS);

	for (let i = 0; i < MAX_STREAMLINES; i++) {
		const base = i * CHEVRON_VERTS_PER_LINE;
		const phase = streamlines[i].phase;
		// Tip vertices at progress 1.0, wing tips at 0.88
		chevronProgress[base] = 1.0;
		chevronProgress[base + 1] = 0.86;
		chevronProgress[base + 2] = 1.0;
		chevronProgress[base + 3] = 0.86;
		for (let k = 0; k < 4; k++) {
			chevronPhase[base + k] = phase;
		}
	}

	const chevronGeo = new THREE.BufferGeometry();
	const chevronPosAttr = new THREE.BufferAttribute(chevronPositions, 3);
	chevronPosAttr.setUsage(THREE.DynamicDrawUsage);
	const chevronAlphaAttr = new THREE.BufferAttribute(chevronAlpha, 1);
	chevronAlphaAttr.setUsage(THREE.DynamicDrawUsage);

	chevronGeo.setAttribute('position', chevronPosAttr);
	chevronGeo.setAttribute('aProgress', new THREE.BufferAttribute(chevronProgress, 1));
	chevronGeo.setAttribute('aAlpha', chevronAlphaAttr);
	chevronGeo.setAttribute('aPhase', new THREE.BufferAttribute(chevronPhase, 1));

	const chevronMesh = new THREE.LineSegments(chevronGeo, lineMat);
	chevronMesh.frustumCulled = false;
	group.add(chevronMesh);

	// Stipple tracer beads riding along the streamlines
	const beadPositions = new Float32Array(TOTAL_BEADS * 3);
	const beadAlpha = new Float32Array(TOTAL_BEADS);
	const beadSize = new Float32Array(TOTAL_BEADS);
	for (let i = 0; i < MAX_STREAMLINES; i++) {
		beadSize[i * BEADS_PER_LINE] = 2.6;
		beadSize[i * BEADS_PER_LINE + 1] = 1.9;
	}

	const beadGeo = new THREE.BufferGeometry();
	const beadPosAttr = new THREE.BufferAttribute(beadPositions, 3);
	beadPosAttr.setUsage(THREE.DynamicDrawUsage);
	const beadAlphaAttr = new THREE.BufferAttribute(beadAlpha, 1);
	beadAlphaAttr.setUsage(THREE.DynamicDrawUsage);

	beadGeo.setAttribute('position', beadPosAttr);
	beadGeo.setAttribute('aAlpha', beadAlphaAttr);
	beadGeo.setAttribute('aSize', new THREE.BufferAttribute(beadSize, 1));

	const beadUniforms = {
		uDarkMode: { value: 0 },
		uPixelRatio: { value: typeof window !== 'undefined' ? Math.min(window.devicePixelRatio || 1, 2) : 1 }
	};

	const beadMat = new THREE.ShaderMaterial({
		uniforms: beadUniforms,
		vertexShader: BEAD_VERTEX_SHADER,
		fragmentShader: BEAD_FRAGMENT_SHADER,
		transparent: true,
		depthWrite: false,
		depthTest: true
	});

	const beadPoints = new THREE.Points(beadGeo, beadMat);
	beadPoints.frustumCulled = false;
	group.add(beadPoints);

	scene.add(group);

	// Scratch buffers (zero per-frame allocations)
	const curlOut: [number, number, number] = [0, 0, 0];
	const renderX = new Float32Array(TRAIL_POINTS);
	const renderY = new Float32Array(TRAIL_POINTS);
	const renderZ = new Float32Array(TRAIL_POINTS);

	let simTime = 0;
	let initialized = false;
	let lastWindDensity = Number.NaN;

	function evaluateFlow(
		x: number,
		y: number,
		z: number,
		dirX: number,
		dirZ: number,
		turbAmp: number,
		timeShift: number,
		out: [number, number, number]
	): void {
		const freq = 0.125;
		noise.curl3D(x * freq + timeShift, y * freq * 1.15, z * freq - timeShift * 0.7, out);
		// Combine mean wind vector with divergence-free curl deflection
		const vx = dirX + out[0] * turbAmp * 0.52;
		const vy = out[1] * turbAmp * 0.28;
		const vz = dirZ + out[2] * turbAmp * 0.52;
		const invLen = 1 / Math.max(Math.hypot(vx, vy, vz), 1e-4);
		out[0] = vx * invLen;
		out[1] = vy * invLen;
		out[2] = vz * invLen;
	}

	function resetHistoryToHead(s: StreamlineState): void {
		s.histX.fill(s.headX);
		s.histY.fill(s.headY);
		s.histZ.fill(s.headZ);
		s.filledCount = 1;
		s.distAccum = 0;
	}

	function seedInitialSnakeHistory(
		dirX: number,
		dirZ: number,
		turbAmp: number,
		baseSpacing: number
	): void {
		for (let i = 0; i < MAX_STREAMLINES; i++) {
			const s = streamlines[i];
			const spacing = baseSpacing * s.lengthFactor;
			let px = s.headX;
			let py = s.headY;
			let pz = s.headZ;

			s.histX[TRAIL_POINTS - 1] = px;
			s.histY[TRAIL_POINTS - 1] = py;
			s.histZ[TRAIL_POINTS - 1] = pz;

			for (let k = TRAIL_POINTS - 2; k >= 0; k--) {
				evaluateFlow(px, py, pz, dirX, dirZ, turbAmp, 0, curlOut);
				if (k === TRAIL_POINTS - 2) {
					s.tangentX = curlOut[0];
					s.tangentY = curlOut[1];
					s.tangentZ = curlOut[2];
				}
				px -= curlOut[0] * spacing;
				py = clamp(py - curlOut[1] * spacing * 0.65, 0.35, MAX_ALT_Y + 0.8);
				pz -= curlOut[2] * spacing;
				s.histX[k] = px;
				s.histY[k] = py;
				s.histZ[k] = pz;
			}
			s.filledCount = TRAIL_POINTS;
			s.distAccum = 0;
		}
	}

	function pushHistoryNode(s: StreamlineState, x: number, y: number, z: number): void {
		// Shift static body nodes back by 1 slot (snake game: tail follows exact path of head)
		s.histX.copyWithin(0, 1);
		s.histY.copyWithin(0, 1);
		s.histZ.copyWithin(0, 1);
		const last = TRAIL_POINTS - 1;
		s.histX[last] = x;
		s.histY[last] = y;
		s.histZ[last] = z;
		if (s.filledCount < TRAIL_POINTS) {
			s.filledCount++;
		}
	}

	function update(dt: number, _elapsed: number, params: CloudParams): void {
		const isDark = params.surfaceMode === 'dark';
		const darkVal = isDark ? 1.0 : 0.0;
		sharedUniforms.uDarkMode.value = darkVal;
		beadUniforms.uDarkMode.value = darkVal;

		const showWind = params.showWind ?? true;
		group.visible = showWind;
		if (!showWind) {
			return;
		}

		const windDensity = clamp(params.windDensity ?? 0.65, 0.2, 1.0);
		const activeCount = Math.max(12, Math.round(MAX_STREAMLINES * windDensity));
		const speedNorm = clamp(params.windSpeed / 40, 0, 1);
		sharedUniforms.uSpeedNorm.value = speedNorm;

		const windRad = (params.windDirection * Math.PI) / 180;
		const dirX = Math.sin(windRad);
		const dirZ = -Math.cos(windRad);

		// Base advection speed and fixed world-space spacing between snake history nodes
		const advectBase = 0.85 + params.windSpeed * 0.22;
		const turbAmp = 0.08 + params.turbulence * 1.35;
		const baseSpacing = 0.21;

		if (!initialized) {
			seedInitialSnakeHistory(dirX, dirZ, turbAmp, baseSpacing);
		}

		if (params.paused && initialized && lastWindDensity === windDensity) {
			return;
		}

		const timeStep = params.paused ? 0 : dt * params.timeSpeed;
		simTime += timeStep;
		sharedUniforms.uTime.value = simTime;

		const timeShift = simTime * (0.14 + speedNorm * 0.16);

		for (let i = 0; i < activeCount; i++) {
			const s = streamlines[i];
			const spacing = baseSpacing * s.lengthFactor;

			// Advance ONLY the head along the live 3D flow field; body nodes remain static in world space
			if (timeStep > 0) {
				evaluateFlow(s.headX, s.headY, s.headZ, dirX, dirZ, turbAmp, timeShift, curlOut);
				s.tangentX = curlOut[0];
				s.tangentY = curlOut[1];
				s.tangentZ = curlOut[2];

				const stepDist = advectBase * s.speedFactor * timeStep;
				const nextRawX = s.headX + curlOut[0] * stepDist;
				const nextRawZ = s.headZ + curlOut[2] * stepDist;
				const nextY = clamp(
					s.headY +
						curlOut[1] * stepDist * 0.65 +
						(s.baseY - s.headY) * Math.min(1, timeStep * 0.8),
					MIN_ALT_Y,
					MAX_ALT_Y
				);

				const wrappedX = wrapCoord(nextRawX);
				const wrappedZ = wrapCoord(nextRawZ);
				const didWrap =
					Math.abs(wrappedX - nextRawX) > 1.0 || Math.abs(wrappedZ - nextRawZ) > 1.0;

				s.headX = wrappedX;
				s.headY = nextY;
				s.headZ = wrappedZ;

				if (didWrap) {
					// Cleanly re-seed snake at the opposite boundary so it grows out without stretching
					resetHistoryToHead(s);
				} else {
					s.distAccum += stepDist;
					// Record static breadcrumbs at uniform arc spacing so the tail snakes along the exact path
					let guard = 0;
					while (s.distAccum >= spacing && guard < 6) {
						s.distAccum -= spacing;
						const backDist = s.distAccum;
						pushHistoryNode(
							s,
							s.headX - s.tangentX * backDist,
							clamp(s.headY - s.tangentY * backDist * 0.65, 0.35, MAX_ALT_Y + 0.8),
							s.headZ - s.tangentZ * backDist
						);
						guard++;
					}
					if (s.distAccum >= spacing) {
						s.distAccum = 0;
					}
				}
			}

			// Build smooth snake polyline from static history nodes + live head + smoothly gliding tail tip
			const frac = clamp(s.distAccum / spacing, 0, 1);
			const lastIdx = TRAIL_POINTS - 1;
			renderX[lastIdx] = s.headX;
			renderY[lastIdx] = s.headY;
			renderZ[lastIdx] = s.headZ;

			for (let k = 1; k < lastIdx; k++) {
				// All interior nodes sit at their exact static recorded world positions
				renderX[k] = s.histX[k + 1];
				renderY[k] = s.histY[k + 1];
				renderZ[k] = s.histZ[k + 1];
			}

			// Smoothly glide the tail tip from hist[1] -> hist[2] as the head advances by `frac`
			// (continuous across history shifts: old hist[2] becomes new hist[1])
			renderX[0] = lerp(s.histX[1], s.histX[2], frac);
			renderY[0] = lerp(s.histY[1], s.histY[2], frac);
			renderZ[0] = lerp(s.histZ[1], s.histZ[2], frac);

			// Write line segments with smooth radial boundary fade
			const baseVert = i * VERTS_PER_LINE;
			for (let seg = 0; seg < SEGMENTS_PER_LINE; seg++) {
				const v0 = baseVert + seg * 2;
				const v1 = v0 + 1;
				const p0 = v0 * 3;
				const p1 = v1 * 3;

				const x0 = renderX[seg];
				const y0 = renderY[seg];
				const z0 = renderZ[seg];
				const x1 = renderX[seg + 1];
				const y1 = renderY[seg + 1];
				const z1 = renderZ[seg + 1];

				linePositions[p0] = x0;
				linePositions[p0 + 1] = y0;
				linePositions[p0 + 2] = z0;
				linePositions[p1] = x1;
				linePositions[p1 + 1] = y1;
				linePositions[p1 + 2] = z1;

				const r0 = Math.hypot(x0, z0) / DOMAIN_HALF;
				const r1 = Math.hypot(x1, z1) / DOMAIN_HALF;
				lineAlpha[v0] = (1.0 - smoothstep(0.76, 1.0, r0)) * s.opacityWeight;
				lineAlpha[v1] = (1.0 - smoothstep(0.76, 1.0, r1)) * s.opacityWeight;
			}

			// Leading directional chevron at streamline head
			const headTangentX = s.tangentX;
			const headTangentY = s.tangentY;
			const headTangentZ = s.tangentZ;
			const headR = Math.hypot(s.headX, s.headZ) / DOMAIN_HALF;
			const headAlpha = (1.0 - smoothstep(0.76, 0.98, headR)) * s.opacityWeight;
			const chevLen = lerp(0.16, 0.34, speedNorm);
			const wingSpread = chevLen * 0.48;

			const hLen = Math.max(Math.hypot(headTangentX, headTangentZ), 1e-4);
			const sideX = -headTangentZ / hLen;
			const sideZ = headTangentX / hLen;

			const backX = s.headX - headTangentX * chevLen;
			const backY = s.headY - headTangentY * chevLen;
			const backZ = s.headZ - headTangentZ * chevLen;

			const cb = i * CHEVRON_VERTS_PER_LINE;
			const cp = cb * 3;
			chevronPositions[cp] = s.headX;
			chevronPositions[cp + 1] = s.headY;
			chevronPositions[cp + 2] = s.headZ;
			chevronPositions[cp + 3] = backX + sideX * wingSpread;
			chevronPositions[cp + 4] = backY;
			chevronPositions[cp + 5] = backZ + sideZ * wingSpread;
			chevronPositions[cp + 6] = s.headX;
			chevronPositions[cp + 7] = s.headY;
			chevronPositions[cp + 8] = s.headZ;
			chevronPositions[cp + 9] = backX - sideX * wingSpread;
			chevronPositions[cp + 10] = backY;
			chevronPositions[cp + 11] = backZ - sideZ * wingSpread;

			chevronAlpha[cb] = headAlpha;
			chevronAlpha[cb + 1] = headAlpha * 0.85;
			chevronAlpha[cb + 2] = headAlpha;
			chevronAlpha[cb + 3] = headAlpha * 0.85;

			// Stipple tracer beads at head and mid-body
			const b0 = i * BEADS_PER_LINE;
			const b1 = b0 + 1;
			beadPositions[b0 * 3] = s.headX;
			beadPositions[b0 * 3 + 1] = s.headY;
			beadPositions[b0 * 3 + 2] = s.headZ;
			beadAlpha[b0] = headAlpha * (0.55 + 0.35 * speedNorm);

			const midIdx = Math.floor(SEGMENTS_PER_LINE * 0.62);
			const mx = renderX[midIdx];
			const my = renderY[midIdx];
			const mz = renderZ[midIdx];
			const midR = Math.hypot(mx, mz) / DOMAIN_HALF;
			beadPositions[b1 * 3] = mx;
			beadPositions[b1 * 3 + 1] = my;
			beadPositions[b1 * 3 + 2] = mz;
			beadAlpha[b1] = (1.0 - smoothstep(0.76, 0.98, midR)) * s.opacityWeight * 0.45;
		}

		lineGeo.setDrawRange(0, activeCount * VERTS_PER_LINE);
		chevronGeo.setDrawRange(0, activeCount * CHEVRON_VERTS_PER_LINE);
		beadGeo.setDrawRange(0, activeCount * BEADS_PER_LINE);

		linePosAttr.needsUpdate = true;
		lineAlphaAttr.needsUpdate = true;
		chevronPosAttr.needsUpdate = true;
		chevronAlphaAttr.needsUpdate = true;
		beadPosAttr.needsUpdate = true;
		beadAlphaAttr.needsUpdate = true;

		initialized = true;
		lastWindDensity = windDensity;
	}

	function dispose(): void {
		scene.remove(group);
		lineGeo.dispose();
		chevronGeo.dispose();
		beadGeo.dispose();
		lineMat.dispose();
		beadMat.dispose();
	}

	return {
		update,
		dispose
	};
}
