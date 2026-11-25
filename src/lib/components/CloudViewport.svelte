<script lang="ts">
	import { onMount } from 'svelte';
	import { createCloudViewport, type CameraPreset, type CloudViewportController } from '$lib/cloud';
	import { appState } from '$lib/state.svelte';
	import type { CloudTelemetry } from '$lib/types';

	let containerEl: HTMLDivElement | undefined = $state();
	let canvasEl: HTMLCanvasElement | undefined = $state();
	let controller: CloudViewportController | null = null;

	function handleTelemetry(telemetry: CloudTelemetry) {
		appState.telemetry.dominantType = telemetry.dominantType;
		appState.telemetry.weights = { ...telemetry.weights };
		appState.telemetry.activePoints = telemetry.activePoints;
		appState.telemetry.baseAltitudeKm = telemetry.baseAltitudeKm;
		appState.telemetry.topAltitudeKm = telemetry.topAltitudeKm;
		appState.telemetry.sunElevationDeg = telemetry.sunElevationDeg;
		appState.telemetry.sunAzimuthDeg = telemetry.sunAzimuthDeg;
		appState.telemetry.dewPointC = telemetry.dewPointC;
		appState.telemetry.fps = telemetry.fps;
		appState.telemetry.rainActive = telemetry.rainActive;
		appState.telemetry.rainRateMmHr = telemetry.rainRateMmHr;
		appState.telemetry.aircraftCount = telemetry.aircraftCount;
		appState.telemetry.birdCount = telemetry.birdCount;
	}

	onMount(() => {
		if (!canvasEl || !containerEl) return;

		controller = createCloudViewport(
			canvasEl,
			$state.snapshot(appState.params),
			handleTelemetry
		);

		const resizeObserver = new ResizeObserver(() => {
			controller?.resize();
		});
		resizeObserver.observe(containerEl);

		const handleWindowResize = () => controller?.resize();
		window.addEventListener('resize', handleWindowResize);

		return () => {
			window.removeEventListener('resize', handleWindowResize);
			resizeObserver.disconnect();
			controller?.dispose();
			controller = null;
		};
	});

	$effect(() => {
		const snapshot = $state.snapshot(appState.params);
		controller?.updateParams(snapshot);
	});

	export function setCameraPreset(preset: CameraPreset): void {
		controller?.setCameraPreset(preset);
	}

	export function resetCamera(): void {
		controller?.resetCamera();
	}

	export function resize(): void {
		controller?.resize();
	}
</script>

<div class="viewport-root" bind:this={containerEl}>
	<canvas bind:this={canvasEl} class="viewport-canvas"></canvas>
</div>

<style>
	.viewport-root {
		position: fixed;
		inset: 0;
		width: 100vw;
		height: 100vh;
		height: 100dvh;
		overflow: hidden;
		background: var(--bg-paper);
		z-index: 1;
	}

	.viewport-canvas {
		display: block;
		width: 100%;
		height: 100%;
		cursor: grab;
		touch-action: none;
	}

	.viewport-canvas:active {
		cursor: grabbing;
	}
</style>
