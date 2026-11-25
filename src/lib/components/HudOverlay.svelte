<script lang="ts">
	import type { CameraPreset } from '$lib/cloud';
	import { appState, cycleThemePreference } from '$lib/state.svelte';
	import type { ConcreteCloudType } from '$lib/types';

	interface Props {
		onCameraPreset: (preset: CameraPreset) => void;
		onResetCamera: () => void;
	}

	let { onCameraPreset, onResetCamera }: Props = $props();

	let activeCamera = $state<CameraPreset | null>('iso');
	let telemetryExpanded = $state(false);

	const CAMERA_PRESETS: { id: CameraPreset; label: string }[] = [
		{ id: 'iso', label: 'ISO' },
		{ id: 'top', label: 'TOP' },
		{ id: 'side', label: 'SIDE' },
		{ id: 'low', label: 'LOW' }
	];

	const WEIGHT_CODES: { key: ConcreteCloudType; code: string }[] = [
		{ key: 'cumulus', code: 'CU' },
		{ key: 'stratus', code: 'ST' },
		{ key: 'cirrus', code: 'CI' },
		{ key: 'cumulonimbus', code: 'CB' },
		{ key: 'altocumulus', code: 'AC' }
	];

	function handleCamera(preset: CameraPreset) {
		activeCamera = preset;
		onCameraPreset(preset);
	}

	function handleReset() {
		activeCamera = 'iso';
		onResetCamera();
	}

	function toggleSurface() {
		cycleThemePreference();
	}

	let surfaceLabelDesktop = $derived.by(() => {
		const icon = appState.params.surfaceMode === 'light' ? '◐' : '◑';
		if (appState.themePreference === 'auto') return `${icon} AUTO`;
		return appState.themePreference === 'light' ? '◐ LIGHT' : '◑ DARK';
	});

	let surfaceLabelMobile = $derived.by(() => {
		if (appState.themePreference === 'auto') return 'A';
		return appState.params.surfaceMode === 'light' ? '◐' : '◑';
	});

	function formatCompactPoints(n: number): string {
		if (n >= 1000) return `${(n / 1000).toFixed(1)}K`;
		return `${Math.round(n)}`;
	}
</script>

<div class="hud-layer" class:sidebar-open={appState.menuOpen}>
	<!-- TOP-LEFT: Sidebar Toggle + Title Stamp -->
	<div class="hud-top-left">
		<button
			type="button"
			class="stencil-btn toggle-btn splat-shadow-sm"
			onclick={() => (appState.menuOpen = !appState.menuOpen)}
			title="Toggle control drawer (M or [)"
			aria-label={appState.menuOpen ? 'Hide controls' : 'Show controls'}
		>
			<span class="label-desktop">{appState.menuOpen ? '[ × HIDE ]' : '[ ≡ CONTROLS ]'}</span>
			<span class="label-mobile">{appState.menuOpen ? '[ × ]' : '[ ≡ ]'}</span>
		</button>

		<div class="title-stamp stencil-panel-thin splat-shadow-sm">
			<span class="stamp-title">STRATA // 01</span>
			<span class="stamp-sub">ATMOSPHERIC POINT CLOUD</span>
		</div>
	</div>

	<!-- TOP-RIGHT: Camera Preset Bar + Quick Light/Dark Toggle -->
	<div class="hud-top-right">
		<div class="camera-bar stencil-panel splat-shadow" role="toolbar" aria-label="Camera presets">
			{#each CAMERA_PRESETS as cam}
				<button
					type="button"
					class="bar-btn"
					class:is-active={activeCamera === cam.id}
					onclick={() => handleCamera(cam.id)}
				>
					{cam.label}
				</button>
			{/each}
			<span class="bar-divider" aria-hidden="true"></span>
			<button type="button" class="bar-btn" onclick={handleReset}>RESET</button>
		</div>

		<button
			type="button"
			class="stencil-btn surface-btn splat-shadow-sm"
			onclick={toggleSurface}
			title="Cycle theme: Auto (System) / Light / Dark"
			aria-label="Cycle monochrome theme preference"
		>
			<span class="label-desktop">{surfaceLabelDesktop}</span>
			<span class="label-mobile">{surfaceLabelMobile}</span>
		</button>
	</div>

	<!-- BOTTOM-LEFT: Compact Live Meteorological Telemetry Stamp -->
	<div class="hud-bottom-left">
		<div
			class="telemetry stencil-panel stencil-brackets splat-shadow"
			class:mobile-expanded={telemetryExpanded}
		>
			<button
				type="button"
				class="tele-header"
				onclick={() => (telemetryExpanded = !telemetryExpanded)}
				aria-expanded={telemetryExpanded}
				aria-label="Toggle telemetry details"
			>
				<div class="tele-dominant">
					<span class="tele-kicker">CLASSIFICATION</span>
					<span class="tele-type">{appState.telemetry.dominantType.toUpperCase()}</span>
				</div>
				<div class="tele-header-right">
					<span class="tele-compact-summary tabular">
						{appState.telemetry.baseAltitudeKm.toFixed(1)}–{appState.telemetry.topAltitudeKm.toFixed(1)}km · {Math.round(appState.telemetry.sunElevationDeg)}°
					</span>
					<span class="tele-mode">
						{appState.params.cloudType === 'auto' ? 'AUTO' : 'LOCKED'}
					</span>
					<span class="tele-chevron" aria-hidden="true">{telemetryExpanded ? '−' : '+'}</span>
				</div>
			</button>

			<div class="tele-details">
				<div class="weights-row" aria-label="Cloud type classification weights">
					{#each WEIGHT_CODES as item}
						{@const w = appState.telemetry.weights[item.key] ?? 0}
						<div class="weight-cell" class:is-dominant={appState.telemetry.dominantType === item.key}>
							<div class="weight-bar-track">
								<div class="weight-bar-fill" style="width: {Math.round(w * 100)}%"></div>
							</div>
							<div class="weight-meta tabular">
								<span>{item.code}</span>
								<span>{Math.round(w * 100)}</span>
							</div>
						</div>
					{/each}
				</div>

				<div class="tele-grid tabular">
					<div class="tele-stat">
						<span class="stat-label">ALT</span>
						<span class="stat-val">
							{appState.telemetry.baseAltitudeKm.toFixed(1)}–{appState.telemetry.topAltitudeKm.toFixed(1)} km
						</span>
					</div>
					<div class="tele-stat">
						<span class="stat-label">SUN</span>
						<span class="stat-val">{Math.round(appState.telemetry.sunElevationDeg)}°</span>
					</div>
					<div class="tele-stat">
						<span class="stat-label">DEW</span>
						<span class="stat-val">{appState.telemetry.dewPointC.toFixed(1)}°C</span>
					</div>
					<div class="tele-stat">
						<span class="stat-label">PTS</span>
						<span class="stat-val">{formatCompactPoints(appState.telemetry.activePoints)}</span>
					</div>
					<div class="tele-stat">
						<span class="stat-label">FPS</span>
						<span class="stat-val">{Math.round(appState.telemetry.fps)}</span>
					</div>
				</div>

				<div class="tele-grid-extras tabular">
					<div class="tele-stat">
						<span class="stat-label">PRECIP</span>
						<span class="stat-val" class:is-precip-active={appState.telemetry.rainActive}>
							{appState.telemetry.rainActive
								? `${appState.telemetry.rainRateMmHr.toFixed(1)} mm/h`
								: 'DRY'}
						</span>
					</div>
					<div class="tele-stat">
						<span class="stat-label">AC</span>
						<span class="stat-val">{appState.telemetry.aircraftCount}</span>
					</div>
					<div class="tele-stat">
						<span class="stat-label">BIRDS</span>
						<span class="stat-val">{appState.telemetry.birdCount}</span>
					</div>
				</div>
			</div>
		</div>
	</div>

	<!-- BOTTOM-RIGHT: Interaction Hint -->
	<div class="hud-bottom-right">
		<div class="hint-stamp">
			DRAG ORBIT · SCROLL ZOOM · RIGHT-DRAG PAN · [M] MENU
		</div>
	</div>
</div>

<style>
	.hud-layer {
		position: fixed;
		inset: 0;
		z-index: 20;
		pointer-events: none;
	}

	.hud-top-left,
	.hud-top-right,
	.hud-bottom-left,
	.hud-bottom-right {
		position: fixed;
		pointer-events: auto;
	}

	.hud-top-left {
		top: max(16px, env(safe-area-inset-top));
		left: max(16px, env(safe-area-inset-left));
		display: flex;
		align-items: stretch;
		gap: 0.55rem;
		transition: transform var(--transition-drawer);
	}

	.hud-bottom-left {
		bottom: max(16px, env(safe-area-inset-bottom));
		left: max(16px, env(safe-area-inset-left));
		transition: transform var(--transition-drawer);
	}

	/* Shift left-aligned HUD islands smoothly when the left sidebar is open */
	.hud-layer.sidebar-open .hud-top-left,
	.hud-layer.sidebar-open .hud-bottom-left {
		transform: translateX(var(--sidebar-width));
	}

	.hud-top-right {
		top: max(16px, env(safe-area-inset-top));
		right: max(16px, env(safe-area-inset-right));
		display: flex;
		align-items: stretch;
		gap: 0.55rem;
	}

	.hud-bottom-right {
		bottom: max(16px, env(safe-area-inset-bottom));
		right: max(16px, env(safe-area-inset-right));
		pointer-events: none;
	}

	.label-mobile {
		display: none;
	}

	.toggle-btn {
		padding: 0.45rem 0.72rem;
		border-width: 1.5px;
		font-size: 0.68rem;
	}

	.title-stamp {
		display: flex;
		flex-direction: column;
		justify-content: center;
		padding: 0.28rem 0.65rem;
		line-height: 1.15;
	}

	.stamp-title {
		font-size: 0.74rem;
		font-weight: 700;
		letter-spacing: 0.14em;
		color: var(--ink-strong);
	}

	.stamp-sub {
		font-size: 0.54rem;
		font-weight: 500;
		letter-spacing: 0.12em;
		color: var(--ink-soft);
	}

	/* Camera preset bar */
	.camera-bar {
		display: inline-flex;
		align-items: stretch;
		padding: 0;
	}

	.bar-btn {
		appearance: none;
		border: none;
		background: transparent;
		color: var(--ink);
		padding: 0.44rem 0.62rem;
		font-size: 0.66rem;
		font-weight: 600;
		letter-spacing: 0.1em;
		text-transform: uppercase;
		cursor: pointer;
		transition:
			background-color 90ms ease,
			color 90ms ease;
	}

	.bar-btn:hover,
	.bar-btn.is-active {
		background: var(--surface-invert);
		color: var(--ink-invert);
	}

	.bar-divider {
		width: 1px;
		background: var(--border);
		margin: 4px 0;
	}

	.surface-btn {
		border-width: 1.5px;
		padding: 0.44rem 0.68rem;
		font-size: 0.66rem;
	}

	/* Telemetry stamp */
	.telemetry {
		width: 312px;
		padding: 0.62rem 0.72rem;
		display: flex;
		flex-direction: column;
		gap: 0.52rem;
	}

	.tele-header {
		appearance: none;
		width: 100%;
		border: none;
		background: transparent;
		color: inherit;
		padding: 0 0 0.42rem;
		text-align: left;
		display: flex;
		align-items: flex-start;
		justify-content: space-between;
		gap: 0.5rem;
		border-bottom: 1px solid var(--border-subtle);
		cursor: default;
	}

	.tele-header-right {
		display: flex;
		align-items: center;
		gap: 0.4rem;
		flex-shrink: 0;
	}

	.tele-compact-summary,
	.tele-chevron {
		display: none;
	}

	.tele-details {
		display: flex;
		flex-direction: column;
		gap: 0.52rem;
	}

	.tele-dominant {
		display: flex;
		flex-direction: column;
		gap: 0.1rem;
	}

	.tele-kicker {
		font-size: 0.54rem;
		font-weight: 600;
		letter-spacing: 0.13em;
		color: var(--ink-faint);
	}

	.tele-type {
		font-size: 0.8rem;
		font-weight: 700;
		letter-spacing: 0.1em;
		color: var(--ink-strong);
	}

	.tele-mode {
		font-size: 0.58rem;
		font-weight: 600;
		letter-spacing: 0.1em;
		padding: 0.14rem 0.36rem;
		border: 1px solid var(--border);
		color: var(--ink);
	}

	.weights-row {
		display: grid;
		grid-template-columns: repeat(5, 1fr);
		gap: 0.35rem;
	}

	.weight-cell {
		display: flex;
		flex-direction: column;
		gap: 0.18rem;
		opacity: 0.58;
	}

	.weight-cell.is-dominant {
		opacity: 1;
	}

	.weight-bar-track {
		height: 4px;
		background: var(--border-subtle);
		width: 100%;
	}

	.weight-bar-fill {
		height: 100%;
		background: var(--ink);
		transition: width 160ms ease;
	}

	.weight-meta {
		display: flex;
		justify-content: space-between;
		font-size: 0.56rem;
		font-weight: 600;
		letter-spacing: 0.04em;
		color: var(--ink-soft);
	}

	.weight-cell.is-dominant .weight-meta {
		color: var(--ink-strong);
	}

	.tele-grid {
		display: grid;
		grid-template-columns: repeat(5, auto);
		justify-content: space-between;
		gap: 0.45rem;
		padding-top: 0.38rem;
		border-top: 1px solid var(--border-subtle);
	}

	.tele-grid-extras {
		display: grid;
		grid-template-columns: 1.4fr 0.8fr 0.8fr;
		gap: 0.45rem;
		padding-top: 0.36rem;
		border-top: 1px solid var(--border-subtle);
	}

	.stat-val.is-precip-active {
		display: inline-block;
		padding: 0 0.28rem;
		background: var(--surface-invert);
		color: var(--ink-invert);
		width: fit-content;
	}

	.tele-stat {
		display: flex;
		flex-direction: column;
		gap: 0.08rem;
	}

	.stat-label {
		font-size: 0.52rem;
		font-weight: 600;
		letter-spacing: 0.11em;
		color: var(--ink-faint);
	}

	.stat-val {
		font-size: 0.66rem;
		font-weight: 600;
		color: var(--ink-strong);
		white-space: nowrap;
	}

	/* Interaction hint */
	.hint-stamp {
		padding: 0.28rem 0.55rem;
		background: var(--bg-panel);
		border: 1px solid var(--border-subtle);
		font-size: 0.58rem;
		font-weight: 500;
		letter-spacing: 0.11em;
		color: var(--ink-soft);
	}

	@media (max-width: 820px) {
		.title-stamp,
		.hint-stamp {
			display: none;
		}

		.hud-layer.sidebar-open .hud-top-left,
		.hud-layer.sidebar-open .hud-bottom-left {
			transform: translateX(0);
		}
	}

	@media (max-width: 680px) {
		.hud-top-left {
			top: max(10px, env(safe-area-inset-top));
			left: max(10px, env(safe-area-inset-left));
		}

		.hud-top-right {
			top: max(10px, env(safe-area-inset-top));
			right: max(10px, env(safe-area-inset-right));
			gap: 0.35rem;
		}

		.hud-bottom-left {
			bottom: max(10px, env(safe-area-inset-bottom));
			left: max(10px, env(safe-area-inset-left));
			right: max(10px, env(safe-area-inset-right));
		}

		.label-desktop {
			display: none;
		}

		.label-mobile {
			display: inline;
		}

		.toggle-btn {
			min-height: 36px;
			padding: 0.4rem 0.58rem;
			font-size: 0.7rem;
		}

		.bar-btn {
			min-height: 36px;
			padding: 0.38rem 0.42rem;
			font-size: 0.6rem;
			letter-spacing: 0.06em;
		}

		.surface-btn {
			min-height: 36px;
			min-width: 36px;
			padding: 0.38rem 0.5rem;
			font-size: 0.8rem;
		}

		.telemetry {
			width: 100%;
			max-width: 100%;
			padding: 0.5rem 0.65rem;
			gap: 0;
		}

		.telemetry.mobile-expanded {
			gap: 0.5rem;
		}

		.tele-header {
			cursor: pointer;
			align-items: center;
			padding-bottom: 0;
			border-bottom: none;
		}

		.telemetry.mobile-expanded .tele-header {
			padding-bottom: 0.42rem;
			border-bottom: 1px solid var(--border-subtle);
		}

		.tele-kicker {
			display: none;
		}

		.tele-type {
			font-size: 0.72rem;
		}

		.tele-compact-summary {
			display: inline;
			font-size: 0.6rem;
			font-weight: 600;
			color: var(--ink-soft);
			white-space: nowrap;
		}

		.telemetry.mobile-expanded .tele-compact-summary {
			display: none;
		}

		.tele-chevron {
			display: inline-flex;
			align-items: center;
			justify-content: center;
			width: 18px;
			height: 18px;
			border: 1px solid var(--border-subtle);
			font-size: 0.68rem;
			font-weight: 700;
			color: var(--ink-soft);
		}

		.tele-details {
			display: none;
		}

		.telemetry.mobile-expanded .tele-details {
			display: flex;
		}
	}
</style>
