<script lang="ts">
	import { onMount } from 'svelte';
	import type { CameraPreset } from '$lib/cloud';
	import CloudViewport from '$lib/components/CloudViewport.svelte';
	import HudOverlay from '$lib/components/HudOverlay.svelte';
	import Sidebar from '$lib/components/Sidebar.svelte';
	import { appState, initThemePreference } from '$lib/state.svelte';

	let viewportRef: CloudViewport | undefined = $state();

	onMount(() => {
		// Start with the control drawer collapsed on mobile/narrow screens so the 3D cloud is immediately visible
		if (window.innerWidth < 820) {
			appState.menuOpen = false;
		}
		return initThemePreference();
	});

	function handleKeydown(event: KeyboardEvent) {
		const target = event.target as HTMLElement | null;
		if (
			target &&
			(target.tagName === 'INPUT' ||
				target.tagName === 'TEXTAREA' ||
				target.tagName === 'SELECT' ||
				target.isContentEditable)
		) {
			return;
		}

		if (event.metaKey || event.ctrlKey || event.altKey) return;

		if (event.key === 'm' || event.key === 'M' || event.key === '[') {
			event.preventDefault();
			appState.menuOpen = !appState.menuOpen;
		}
	}

	function handleCameraPreset(preset: CameraPreset) {
		viewportRef?.setCameraPreset(preset);
	}

	function handleResetCamera() {
		viewportRef?.resetCamera();
	}
</script>

<svelte:window onkeydown={handleKeydown} />

<div class="app-shell" data-theme={appState.params.surfaceMode}>
	<CloudViewport bind:this={viewportRef} />
	<Sidebar />
	<HudOverlay onCameraPreset={handleCameraPreset} onResetCamera={handleResetCamera} />
</div>

<style>
	.app-shell {
		position: relative;
		width: 100vw;
		height: 100vh;
		height: 100dvh;
		overflow: hidden;
		background: var(--bg-paper);
		color: var(--ink);
	}
</style>
