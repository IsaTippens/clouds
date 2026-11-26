/**
 * Three.js monochrome point-cloud viewport controller.
 * Manages WebGLRenderer, OrbitControls, stencil coordinate ground plane,
 * splatted dot-gradient ground shadow, procedural cloud point geometry,
 * camera presets, real-time wind advection, and throttled telemetry.
 */

import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import type { CloudParams, CloudTelemetry } from '$lib/types';
import { createStreamingCloudField, type StreamingCloudField } from './generator';
import { computeMeteorology, type MeteorologyState } from './meteorology';
import { clamp } from './noise';
import {
	createCloudPointMaterial,
	createGroundShadowMaterial,
	type CloudShaderUniforms,
	type GroundShadowUniforms
} from './shaders';
import { createWindSystem } from './wind';
import { createRainSystem, type RainStatus } from './rain';
import { createAirTrafficSystem, type TrafficStatus } from './traffic';

export type CameraPreset = 'iso' | 'top' | 'side' | 'low';

export interface CloudViewportController {
	updateParams(params: CloudParams): void;
	resize(): void;
	resetCamera(): void;
	setCameraPreset(preset: CameraPreset): void;
	dispose(): void;
}

const LIGHT_BG = 0xf2f2f0;
const DARK_BG = 0x0a0a0a;

const CAMERA_PRESETS: Record<
	CameraPreset,
	{ pos: [number, number, number]; target: [number, number, number] }
> = {
	iso: { pos: [21.5, 14.5, 22.5], target: [0, 4.4, 0] },
	top: { pos: [0.01, 35.0, 0.01], target: [0, 3.8, 0] },
	side: { pos: [28.0, 5.8, 0.01], target: [0, 5.0, 0] },
	low: { pos: [17.5, 1.45, 17.5], target: [0, 6.2, 0] }
};

/**
 * Build minimalist architectural monochrome stencil grid, compass ring,
 * crosshair ticks, and vertical altitude ruler at y = 0.01.
 */
function createStencilGrid(): {
	group: THREE.Group;
	minorMat: THREE.LineBasicMaterial;
	majorMat: THREE.LineBasicMaterial;
	sunMarker: THREE.LineSegments;
	dispose: () => void;
} {
	const group = new THREE.Group();
	const minorPts: number[] = [];
	const majorPts: number[] = [];

	const extent = 16;
	const step = 2;
	const y = 0.015;

	// Subtle grid hairlines & intersection crosshairs
	for (let i = -extent; i <= extent; i += step) {
		const isAxis = i === 0;
		const target = isAxis ? majorPts : minorPts;
		target.push(-extent, y, i, extent, y, i);
		target.push(i, y, -extent, i, y, extent);
	}

	// Outer framing border ticks
	const tick = 0.35;
	for (let i = -extent; i <= extent; i += 4) {
		majorPts.push(i, y, -extent - tick, i, y, -extent + tick);
		majorPts.push(i, y, extent - tick, i, y, extent + tick);
		majorPts.push(-extent - tick, y, i, -extent + tick, y, i);
		majorPts.push(extent - tick, y, i, extent + tick, y, i);
	}

	// Compass reference ring (radius 14.5) + cardinal N/E/S/W notches
	const ringR = 14.5;
	const segments = 96;
	for (let s = 0; s < segments; s++) {
		const a0 = (s / segments) * Math.PI * 2;
		const a1 = ((s + 1) / segments) * Math.PI * 2;
		minorPts.push(
			Math.cos(a0) * ringR,
			y,
			Math.sin(a0) * ringR,
			Math.cos(a1) * ringR,
			y,
			Math.sin(a1) * ringR
		);
	}

	// Cardinal & 30-degree compass ticks
	for (let deg = 0; deg < 360; deg += 30) {
		const rad = (deg * Math.PI) / 180;
		const isCardinal = deg % 90 === 0;
		const inner = isCardinal ? ringR - 0.75 : ringR - 0.35;
		const outer = isCardinal ? ringR + 0.55 : ringR + 0.25;
		const target = isCardinal ? majorPts : minorPts;
		target.push(
			Math.sin(rad) * inner,
			y,
			-Math.cos(rad) * inner,
			Math.sin(rad) * outer,
			y,
			-Math.cos(rad) * outer
		);
	}

	// North pointer chevron at (0, y, -ringR - 0.8)
	majorPts.push(0, y, -ringR - 1.35, -0.35, y, -ringR - 0.65);
	majorPts.push(0, y, -ringR - 1.35, 0.35, y, -ringR - 0.65);

	// Minimal vertical altitude scale mast at back-left corner (-15.5, 0..12, -15.5)
	const mastX = -15.5;
	const mastZ = -15.5;
	majorPts.push(mastX, 0, mastZ, mastX, 12, mastZ);
	for (let h = 0; h <= 12; h += 2) {
		const tickLen = h % 4 === 0 ? 0.55 : 0.28;
		majorPts.push(mastX, h, mastZ, mastX + tickLen, h, mastZ);
		majorPts.push(mastX, h, mastZ, mastX, h, mastZ + tickLen);
	}

	const minorGeo = new THREE.BufferGeometry();
	minorGeo.setAttribute('position', new THREE.Float32BufferAttribute(minorPts, 3));
	const minorMat = new THREE.LineBasicMaterial({
		color: 0xd6d6d2,
		transparent: true,
		opacity: 0.7
	});

	const majorGeo = new THREE.BufferGeometry();
	majorGeo.setAttribute('position', new THREE.Float32BufferAttribute(majorPts, 3));
	const majorMat = new THREE.LineBasicMaterial({
		color: 0x8e8e8a,
		transparent: true,
		opacity: 0.85
	});

	const minorLines = new THREE.LineSegments(minorGeo, minorMat);
	const majorLines = new THREE.LineSegments(majorGeo, majorMat);
	group.add(minorLines, majorLines);

	// Dynamic sun azimuth indicator tick on the compass ring
	const sunGeo = new THREE.BufferGeometry();
	sunGeo.setAttribute(
		'position',
		new THREE.Float32BufferAttribute(
			[
				0, y + 0.01, -ringR + 0.4, 0, y + 0.01, -ringR - 1.15,
				-0.25, y + 0.01, -ringR - 0.75, 0.25, y + 0.01, -ringR - 0.75
			],
			3
		)
	);
	const sunMarker = new THREE.LineSegments(sunGeo, majorMat);
	group.add(sunMarker);

	return {
		group,
		minorMat,
		majorMat,
		sunMarker,
		dispose: () => {
			minorGeo.dispose();
			majorGeo.dispose();
			sunGeo.dispose();
			minorMat.dispose();
			majorMat.dispose();
		}
	};
}

/**
 * Determine if parameter changes require regenerating the 3D point cloud geometry.
 */
function needsGeometryRebuild(prev: CloudParams, next: CloudParams): boolean {
	return (
		prev.regionSeed !== next.regionSeed ||
		prev.latitude !== next.latitude ||
		prev.longitude !== next.longitude ||
		prev.altitude !== next.altitude ||
		prev.humidity !== next.humidity ||
		prev.pressure !== next.pressure ||
		prev.temperature !== next.temperature ||
		prev.windDirection !== next.windDirection ||
		prev.turbulence !== next.turbulence ||
		prev.coverage !== next.coverage ||
		prev.cloudType !== next.cloudType ||
		prev.pointCount !== next.pointCount ||
		Math.abs(prev.windSpeed - next.windSpeed) >= 2.0
	);
}

export function createCloudViewport(
	canvas: HTMLCanvasElement,
	initialParams: CloudParams,
	onTelemetry?: (telemetry: CloudTelemetry) => void
): CloudViewportController {
	let currentParams: CloudParams = { ...initialParams };
	let meteo: MeteorologyState = computeMeteorology(currentParams);
	let disposed = false;

	// Renderer
	const renderer = new THREE.WebGLRenderer({
		canvas,
		antialias: true,
		alpha: false,
		powerPreference: 'high-performance'
	});
	const getPixelRatio = () => Math.min(window.devicePixelRatio || 1, 2);
	renderer.setPixelRatio(getPixelRatio());
	renderer.setClearColor(currentParams.surfaceMode === 'dark' ? DARK_BG : LIGHT_BG, 1);

	// Scene & Camera
	const scene = new THREE.Scene();
	const camera = new THREE.PerspectiveCamera(42, 1, 0.2, 200);
	camera.position.set(...CAMERA_PRESETS.iso.pos);

	// OrbitControls
	const controls = new OrbitControls(camera, canvas);
	controls.enableDamping = true;
	controls.dampingFactor = 0.065;
	controls.minDistance = 8;
	controls.maxDistance = 68;
	controls.maxPolarAngle = Math.PI * 0.56;
	controls.touches = {
		ONE: THREE.TOUCH.ROTATE,
		TWO: THREE.TOUCH.DOLLY_PAN
	};
	controls.target.set(...CAMERA_PRESETS.iso.target);
	controls.update();

	// Smooth camera transition state
	let camTransition: {
		fromPos: THREE.Vector3;
		toPos: THREE.Vector3;
		fromTarget: THREE.Vector3;
		toTarget: THREE.Vector3;
		elapsed: number;
		duration: number;
	} | null = null;

	const cancelCamTransition = () => {
		camTransition = null;
	};
	controls.addEventListener('start', cancelCamTransition);

	// Minimalist monochrome stencil grid
	const stencilGrid = createStencilGrid();
	scene.add(stencilGrid.group);

	// Initial infinite streaming cloud field ("inf-dev")
	let totalScrollX = 0;
	let totalScrollZ = 0;
	let cloudField: StreamingCloudField = createStreamingCloudField(
		currentParams,
		meteo,
		totalScrollX,
		totalScrollZ
	);

	// 3D Volume density atlas DataTexture (shared by real-time 3D cloud & ground shadow shaders)
	const volumeAtlasBuffer = new Uint8Array(cloudField.volumeAtlasData);
	const volumeAtlasTexture = new THREE.DataTexture(
		volumeAtlasBuffer,
		cloudField.volumeAtlasWidth,
		cloudField.volumeAtlasHeight,
		THREE.RGBAFormat
	);
	volumeAtlasTexture.magFilter = THREE.LinearFilter;
	volumeAtlasTexture.minFilter = THREE.LinearFilter;
	volumeAtlasTexture.wrapS = THREE.ClampToEdgeWrapping;
	volumeAtlasTexture.wrapT = THREE.ClampToEdgeWrapping;
	volumeAtlasTexture.flipY = false;
	volumeAtlasTexture.needsUpdate = true;

	// Cloud points mesh
	const cloudGeometry = new THREE.BufferGeometry();
	const { material: cloudMaterial, uniforms: cloudUniforms } = createCloudPointMaterial(
		volumeAtlasTexture,
		getPixelRatio()
	);
	const cloudPoints = new THREE.Points(cloudGeometry, cloudMaterial);
	cloudPoints.frustumCulled = false;
	scene.add(cloudPoints);

	// Expanded ground shadow plane (76x76) so low-sun shadows stretch freely without square clipping
	const groundGeo = new THREE.PlaneGeometry(76, 76);
	groundGeo.rotateX(-Math.PI / 2);
	const { material: groundMaterial, uniforms: groundUniforms } =
		createGroundShadowMaterial(volumeAtlasTexture);
	const groundMesh = new THREE.Mesh(groundGeo, groundMaterial);
	groundMesh.position.y = 0.005;
	scene.add(groundMesh);

	let posAttr: THREE.BufferAttribute;
	let densityAttr: THREE.BufferAttribute;
	let heightNormAttr: THREE.BufferAttribute;
	let seedAttr: THREE.BufferAttribute;
	let sizeAttr: THREE.BufferAttribute;
	let shadeAttr: THREE.BufferAttribute;

	function bindCloudFieldToGeometry(field: StreamingCloudField): void {
		posAttr = new THREE.BufferAttribute(field.positions, 3);
		densityAttr = new THREE.BufferAttribute(field.aDensity, 1);
		heightNormAttr = new THREE.BufferAttribute(field.aHeightNorm, 1);
		seedAttr = new THREE.BufferAttribute(field.aSeed, 1);
		sizeAttr = new THREE.BufferAttribute(field.aSize, 1);
		shadeAttr = new THREE.BufferAttribute(field.aShade, 1);

		posAttr.setUsage(THREE.DynamicDrawUsage);
		densityAttr.setUsage(THREE.DynamicDrawUsage);
		heightNormAttr.setUsage(THREE.DynamicDrawUsage);
		seedAttr.setUsage(THREE.DynamicDrawUsage);
		sizeAttr.setUsage(THREE.DynamicDrawUsage);
		shadeAttr.setUsage(THREE.DynamicDrawUsage);

		cloudGeometry.setAttribute('position', posAttr);
		cloudGeometry.setAttribute('aDensity', densityAttr);
		cloudGeometry.setAttribute('aHeightNorm', heightNormAttr);
		cloudGeometry.setAttribute('aSeed', seedAttr);
		cloudGeometry.setAttribute('aSize', sizeAttr);
		cloudGeometry.setAttribute('aShade', shadeAttr);
		cloudGeometry.computeBoundingSphere();

		// Update 3D volume atlas texture in place
		volumeAtlasBuffer.set(field.volumeAtlasData);
		volumeAtlasTexture.needsUpdate = true;

		cloudUniforms.uHalfExtentXZ.value = field.bounds.halfExtentXZ;
		cloudUniforms.uBaseY.value = field.bounds.baseY;
		cloudUniforms.uHeightY.value = field.bounds.heightY;
		cloudUniforms.uWindOffset.value.set(field.subOffsetX, field.subOffsetZ);

		groundUniforms.uHalfExtentXZ.value = field.bounds.halfExtentXZ;
		groundUniforms.uBaseY.value = field.bounds.baseY;
		groundUniforms.uHeightY.value = field.bounds.heightY;
		groundUniforms.uWindOffset.value.set(field.subOffsetX, field.subOffsetZ);
	}

	function uploadStreamedSlice(): void {
		posAttr.needsUpdate = true;
		densityAttr.needsUpdate = true;
		heightNormAttr.needsUpdate = true;
		seedAttr.needsUpdate = true;
		sizeAttr.needsUpdate = true;
		shadeAttr.needsUpdate = true;

		volumeAtlasBuffer.set(cloudField.volumeAtlasData);
		volumeAtlasTexture.needsUpdate = true;
	}

	function syncUniforms(
		params: CloudParams,
		m: MeteorologyState,
		cUniforms: CloudShaderUniforms,
		gUniforms: GroundShadowUniforms
	): void {
		const isDark = params.surfaceMode === 'dark';
		renderer.setClearColor(isDark ? DARK_BG : LIGHT_BG, 1);

		stencilGrid.minorMat.color.setHex(isDark ? 0x1e1e1e : 0xd6d6d2);
		stencilGrid.majorMat.color.setHex(isDark ? 0x4d4d4d : 0x8a8a86);
		stencilGrid.sunMarker.rotation.y = (-m.sunAzimuthDeg * Math.PI) / 180;

		const sunElevNorm = clamp((m.sunElevationDeg + 6) / 76, 0, 1);
		// Smoothly fade sun-cast ground shadows out below horizon (nighttime)
		const sunShadowStrength = clamp((m.sunElevationDeg + 3) / 12, 0, 1);

		cUniforms.uTurbulence.value = params.turbulence;
		cUniforms.uPointSize.value = params.pointSize;
		cUniforms.uShadowDither.value = params.shadowDither;
		cUniforms.uDarkMode.value = isDark ? 1.0 : 0.0;
		cUniforms.uSunDir.value.set(m.sunDir[0], m.sunDir[1], m.sunDir[2]);
		cUniforms.uSunElevNorm.value = sunElevNorm;

		gUniforms.uShadowDither.value = params.shadowDither;
		gUniforms.uDarkMode.value = isDark ? 1.0 : 0.0;
		gUniforms.uSunDir.value.set(m.sunDir[0], m.sunDir[1], m.sunDir[2]);
		gUniforms.uSunElevNorm.value = sunElevNorm;
		gUniforms.uSunShadowStrength.value = sunShadowStrength;
	}

	bindCloudFieldToGeometry(cloudField);
	syncUniforms(currentParams, meteo, cloudUniforms, groundUniforms);

	// Atmospheric & air-traffic subsystems (wind streamlines, rain curtains/splashes, aircraft & birds)
	const windSystem = createWindSystem(scene);
	const rainSystem = createRainSystem(scene);
	const airTrafficSystem = createAirTrafficSystem(scene);
	let rainStatus: RainStatus = { rainActive: false, rainRateMmHr: 0 };
	let trafficStatus: TrafficStatus = { aircraftCount: 0, birdCount: 0 };

	// Sizing & ResizeObserver
	function resize(): void {
		if (disposed) return;
		const parent = canvas.parentElement;
		const width = Math.max(1, parent ? parent.clientWidth : canvas.clientWidth || window.innerWidth);
		const height = Math.max(1, parent ? parent.clientHeight : canvas.clientHeight || window.innerHeight);
		const dpr = getPixelRatio();
		const aspect = width / height;

		renderer.setPixelRatio(dpr);
		renderer.setSize(width, height, false);
		camera.aspect = aspect;
		// Gently widen vertical FOV on portrait mobile screens so wide cloud formations stay framed
		camera.fov = aspect < 0.95 ? clamp(42 / Math.pow(aspect, 0.52), 42, 66) : 42;
		camera.updateProjectionMatrix();
		cloudUniforms.uPixelRatio.value = dpr;
	}

	resize();

	let resizeObserver: ResizeObserver | null = null;
	if (typeof ResizeObserver !== 'undefined') {
		resizeObserver = new ResizeObserver(() => resize());
		resizeObserver.observe(canvas.parentElement ?? canvas);
	}

	// Simulation, debounced rebuild, & telemetry state
	let simTime = 0;
	let geometryDirty = false;
	let lastRebuildAt = performance.now();

	let lastFrameTime = performance.now();
	let fpsAccum = 0;
	let fpsFrames = 0;
	let measuredFps = 60;
	let lastTelemetryAt = 0;
	let rafId = 0;

	function emitTelemetry(): void {
		if (!onTelemetry) return;
		onTelemetry({
			dominantType: meteo.dominantType,
			weights: { ...meteo.weights },
			activePoints: cloudField.activePoints,
			baseAltitudeKm: meteo.baseAltitudeKm,
			topAltitudeKm: meteo.topAltitudeKm,
			sunElevationDeg: meteo.sunElevationDeg,
			sunAzimuthDeg: meteo.sunAzimuthDeg,
			dewPointC: meteo.dewPointC,
			fps: Math.round(measuredFps),
			rainActive: rainStatus.rainActive,
			rainRateMmHr: rainStatus.rainRateMmHr,
			aircraftCount: trafficStatus.aircraftCount,
			birdCount: trafficStatus.birdCount
		});
	}

	// Prime atmospheric & air-traffic systems and emit initial telemetry immediately
	{
		const cloudBaseY = cloudField.bounds.baseY;
		const cloudTopY = cloudField.bounds.baseY + cloudField.bounds.heightY;
		windSystem.update(0, 0, currentParams);
		rainStatus = rainSystem.update(
			0,
			0,
			currentParams,
			cloudBaseY,
			meteo.dominantType,
			cloudField.sampleRainColumn,
			cloudField.getOverheadCloudFactor()
		);
		trafficStatus = airTrafficSystem.update(0, 0, currentParams, cloudBaseY, cloudTopY);
	}
	emitTelemetry();

	function animate(now: number): void {
		if (disposed) return;
		rafId = requestAnimationFrame(animate);

		const dt = Math.min(Math.max((now - lastFrameTime) / 1000, 0), 0.1);
		lastFrameTime = now;

		// Rolling FPS measurement
		fpsAccum += dt;
		fpsFrames++;
		if (fpsAccum >= 0.4) {
			measuredFps = fpsFrames / fpsAccum;
			fpsAccum = 0;
			fpsFrames = 0;
		}

		// Debounced structural cloud regeneration (preserves current infinite scroll position)
		if (geometryDirty && now - lastRebuildAt >= 75) {
			cloudField = createStreamingCloudField(currentParams, meteo, totalScrollX, totalScrollZ);
			bindCloudFieldToGeometry(cloudField);
			geometryDirty = false;
			lastRebuildAt = now;
			emitTelemetry();
		}

		// Continuous infinite non-repeating wind streaming
		if (!currentParams.paused) {
			const speedMult = currentParams.timeSpeed;
			simTime += dt * speedMult;

			const windRad = (currentParams.windDirection * Math.PI) / 180;
			const dirX = Math.sin(windRad);
			const dirZ = -Math.cos(windRad);
			const drift = currentParams.windSpeed * 0.055 * speedMult * dt;
			const driftX = dirX * drift;
			const driftZ = dirZ * drift;
			totalScrollX += driftX;
			totalScrollZ += driftZ;

			const sliceShifted = cloudField.advance(driftX, driftZ);
			if (sliceShifted) {
				uploadStreamedSlice();
			}

			cloudUniforms.uTime.value = simTime;
			cloudUniforms.uWindOffset.value.set(cloudField.subOffsetX, cloudField.subOffsetZ);
			groundUniforms.uWindOffset.value.set(cloudField.subOffsetX, cloudField.subOffsetZ);
		}

		// Update wind currents, localized rain events under clouds, and air traffic every frame
		const cloudBaseY = cloudField.bounds.baseY;
		const cloudTopY = cloudField.bounds.baseY + cloudField.bounds.heightY;
		windSystem.update(dt, simTime, currentParams);
		rainStatus = rainSystem.update(
			dt,
			simTime,
			currentParams,
			cloudBaseY,
			meteo.dominantType,
			cloudField.sampleRainColumn,
			cloudField.getOverheadCloudFactor()
		);
		trafficStatus = airTrafficSystem.update(dt, simTime, currentParams, cloudBaseY, cloudTopY);

		// Smooth camera preset animation
		if (camTransition) {
			camTransition.elapsed += dt;
			const t = clamp(camTransition.elapsed / camTransition.duration, 0, 1);
			// Cubic ease-out
			const ease = 1 - Math.pow(1 - t, 3);
			camera.position.lerpVectors(camTransition.fromPos, camTransition.toPos, ease);
			controls.target.lerpVectors(camTransition.fromTarget, camTransition.toTarget, ease);
			if (t >= 1) {
				camTransition = null;
			}
		}

		controls.update();
		renderer.render(scene, camera);

		// Throttled telemetry (~7 Hz)
		if (now - lastTelemetryAt >= 145) {
			lastTelemetryAt = now;
			emitTelemetry();
		}
	}

	rafId = requestAnimationFrame(animate);

	function setCameraPreset(preset: CameraPreset): void {
		const spec = CAMERA_PRESETS[preset];
		camTransition = {
			fromPos: camera.position.clone(),
			toPos: new THREE.Vector3(...spec.pos),
			fromTarget: controls.target.clone(),
			toTarget: new THREE.Vector3(...spec.target),
			elapsed: 0,
			duration: 0.48
		};
	}

	function resetCamera(): void {
		setCameraPreset('iso');
	}

	function updateParams(nextParams: CloudParams): void {
		if (disposed) return;
		const structuralChanged = needsGeometryRebuild(currentParams, nextParams);
		currentParams = { ...nextParams };
		meteo = computeMeteorology(currentParams);
		syncUniforms(currentParams, meteo, cloudUniforms, groundUniforms);

		if (structuralChanged) {
			geometryDirty = true;
		}

		emitTelemetry();
	}

	function dispose(): void {
		if (disposed) return;
		disposed = true;
		cancelAnimationFrame(rafId);
		resizeObserver?.disconnect();
		controls.removeEventListener('start', cancelCamTransition);
		controls.dispose();
		cloudGeometry.dispose();
		cloudMaterial.dispose();
		groundGeo.dispose();
		groundMaterial.dispose();
		volumeAtlasTexture.dispose();
		stencilGrid.dispose();
		windSystem.dispose();
		rainSystem.dispose();
		airTrafficSystem.dispose();
		renderer.dispose();
	}

	return {
		updateParams,
		resize,
		resetCamera,
		setCameraPreset,
		dispose
	};
}
