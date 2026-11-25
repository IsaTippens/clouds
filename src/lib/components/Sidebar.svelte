<script lang="ts">
	import {
		appState,
		applyPreset,
		PRESETS,
		randomizeLocation,
		resetParams
	} from '$lib/state.svelte';
	import type { CloudType, RainMode, SurfaceMode } from '$lib/types';
	import StencilSlider from './StencilSlider.svelte';

	const CLOUD_TYPES: { id: CloudType; label: string; code: string }[] = [
		{ id: 'auto', label: 'Auto', code: 'MET' },
		{ id: 'cumulus', label: 'Cumulus', code: 'CU' },
		{ id: 'stratus', label: 'Stratus', code: 'ST' },
		{ id: 'cirrus', label: 'Cirrus', code: 'CI' },
		{ id: 'cumulonimbus', label: 'Cumulonimbus', code: 'CB' },
		{ id: 'altocumulus', label: 'Altocumulus', code: 'AC' }
	];

	const RAIN_MODES: { id: RainMode; label: string }[] = [
		{ id: 'auto', label: 'AUTO' },
		{ id: 'on', label: 'ON' },
		{ id: 'off', label: 'OFF' }
	];

	const COMPASS_POINTS = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW'] as const;

	let sections = $state({
		cloud: true,
		time: true,
		position: true,
		weather: true,
		atmosphere: true,
		render: true
	});

	let activePreset = $derived(
		PRESETS.find((entry) => entry.id === appState.activePresetId) ?? null
	);

	function toggleSection(key: keyof typeof sections) {
		sections[key] = !sections[key];
	}

	function markCustom() {
		appState.activePresetId = null;
	}

	function selectCloudType(type: CloudType) {
		appState.params.cloudType = type;
		markCustom();
	}

	function setSurfaceMode(mode: SurfaceMode) {
		appState.params.surfaceMode = mode;
	}

	function setRainMode(mode: RainMode) {
		appState.params.rainMode = mode;
	}

	function handleSeedInput(event: Event) {
		const target = event.currentTarget as HTMLInputElement;
		const parsed = parseInt(target.value, 10);
		if (!Number.isNaN(parsed)) {
			appState.params.regionSeed = Math.max(0, Math.min(999999, parsed));
		}
	}

	function formatTime(hours: number): string {
		const normalized = ((hours % 24) + 24) % 24;
		const wholeHours = Math.floor(normalized);
		const minutes = Math.floor((normalized - wholeHours) * 60);
		return `${String(wholeHours).padStart(2, '0')}:${String(minutes).padStart(2, '0')}`;
	}

	function formatLat(deg: number): string {
		const hemi = deg >= 0 ? 'N' : 'S';
		return `${Math.abs(deg).toFixed(1)}°${hemi}`;
	}

	function formatLon(deg: number): string {
		const hemi = deg >= 0 ? 'E' : 'W';
		return `${Math.abs(deg).toFixed(1)}°${hemi}`;
	}

	function formatPercent(val: number): string {
		return `${Math.round(val * 100)}%`;
	}

	function formatWindDir(deg: number): string {
		const norm = ((deg % 360) + 360) % 360;
		const idx = Math.round(norm / 45) % 8;
		return `${Math.round(norm)}° ${COMPASS_POINTS[idx]}`;
	}

	function formatPoints(count: number): string {
		return Math.round(count).toLocaleString('en-US');
	}
</script>

<!-- Mobile tap-outside backdrop to dismiss the control drawer -->
{#if appState.menuOpen}
	<button
		type="button"
		class="sidebar-backdrop"
		aria-label="Close control drawer"
		onclick={() => (appState.menuOpen = false)}
	></button>
{/if}

<aside
	class="sidebar"
	class:is-open={appState.menuOpen}
	aria-label="Atmosphere and render controls"
	aria-hidden={!appState.menuOpen}
>
	<!-- Splatted halftone dot-gradient shadow cast onto the main 3D viewport -->
	<div class="splat-edge-right" aria-hidden="true"></div>

	<div class="sidebar-header splat-strip">
		<div class="brand">
			<span class="brand-code">STRATA // 01</span>
			<span class="brand-sub">CONTROL MATRIX</span>
		</div>
		<button
			type="button"
			class="stencil-btn close-btn"
			onclick={() => (appState.menuOpen = false)}
			title="Hide controls (M or [)"
			aria-label="Collapse sidebar"
		>
			[ × ]
		</button>
	</div>

	<div class="sidebar-body stencil-scroll">
		<!-- 01. CLOUD SYSTEM -->
		<section class="accordion">
			<button
				type="button"
				class="accordion-header"
				onclick={() => toggleSection('cloud')}
				aria-expanded={sections.cloud}
			>
				<span class="mono-micro">01 // CLOUD SYSTEM</span>
				<span class="accordion-sign">{sections.cloud ? '[ − ]' : '[ + ]'}</span>
			</button>

			{#if sections.cloud}
				<div class="accordion-body">
					<div class="sub-label">MORPHOLOGY</div>
					<div class="type-grid">
						{#each CLOUD_TYPES as item}
							<button
								type="button"
								class="stencil-btn type-chip"
								class:is-active={appState.params.cloudType === item.id}
								class:splat-shadow-sm={appState.params.cloudType === item.id}
								onclick={() => selectCloudType(item.id)}
							>
								<span class="chip-code">{item.code}</span>
								<span class="chip-name">{item.label}</span>
							</button>
						{/each}
					</div>

					<div class="sub-label">PRESETS</div>
					<div class="preset-list">
						{#each PRESETS as preset}
							<button
								type="button"
								class="stencil-btn preset-chip"
								class:is-active={appState.activePresetId === preset.id}
								class:splat-shadow-sm={appState.activePresetId === preset.id}
								onclick={() => applyPreset(preset.id)}
							>
								<span class="preset-code">{preset.code}</span>
								<span class="preset-name">{preset.name}</span>
							</button>
						{/each}
					</div>

					{#if activePreset}
						<p class="preset-desc">{activePreset.description}</p>
					{/if}
				</div>
			{/if}
		</section>

		<!-- 02. TIME -->
		<section class="accordion">
			<button
				type="button"
				class="accordion-header"
				onclick={() => toggleSection('time')}
				aria-expanded={sections.time}
			>
				<span class="mono-micro">02 // TIME</span>
				<span class="accordion-sign">{sections.time ? '[ − ]' : '[ + ]'}</span>
			</button>

			{#if sections.time}
				<div class="accordion-body">
					<div class="inline-row">
						<button
							type="button"
							class="stencil-btn play-btn"
							class:is-active={appState.params.paused}
							class:splat-shadow-sm={appState.params.paused}
							onclick={() => (appState.params.paused = !appState.params.paused)}
						>
							{appState.params.paused ? '[ ▶ RESUME ]' : '[ ❚❚ PAUSE ]'}
						</button>
						<span class="time-readout tabular">SOLAR {formatTime(appState.params.timeOfDay)}</span>
					</div>

					<StencilSlider
						label="Time of Day"
						bind:value={appState.params.timeOfDay}
						min={0}
						max={24}
						step={0.05}
						ticks={9}
						formatValue={formatTime}
					/>

					<StencilSlider
						label="Simulation Speed"
						bind:value={appState.params.timeSpeed}
						min={0}
						max={3}
						step={0.05}
						ticks={7}
						formatValue={(v) => `${v.toFixed(2)}×`}
					/>
				</div>
			{/if}
		</section>

		<!-- 03. POSITION -->
		<section class="accordion">
			<button
				type="button"
				class="accordion-header"
				onclick={() => toggleSection('position')}
				aria-expanded={sections.position}
			>
				<span class="mono-micro">03 // POSITION</span>
				<span class="accordion-sign">{sections.position ? '[ − ]' : '[ + ]'}</span>
			</button>

			{#if sections.position}
				<div class="accordion-body">
					<StencilSlider
						label="Latitude"
						bind:value={appState.params.latitude}
						min={-90}
						max={90}
						step={0.1}
						ticks={9}
						formatValue={formatLat}
						onChange={markCustom}
					/>

					<StencilSlider
						label="Longitude"
						bind:value={appState.params.longitude}
						min={-180}
						max={180}
						step={0.1}
						ticks={9}
						formatValue={formatLon}
						onChange={markCustom}
					/>

					<StencilSlider
						label="Layer Altitude"
						bind:value={appState.params.altitude}
						min={0.5}
						max={12}
						step={0.1}
						unit="km"
						ticks={9}
						onChange={markCustom}
					/>

					<div class="seed-row">
						<label class="seed-field">
							<span class="seed-label">SEED</span>
							<input
								type="number"
								class="seed-input tabular"
								min={0}
								max={999999}
								value={appState.params.regionSeed}
								onchange={handleSeedInput}
							/>
						</label>
						<button
							type="button"
							class="stencil-btn splat-shadow-sm"
							onclick={() => {
								randomizeLocation();
								markCustom();
							}}
						>
							RANDOMIZE
						</button>
					</div>
				</div>
			{/if}
		</section>

		<!-- 04. WEATHER -->
		<section class="accordion">
			<button
				type="button"
				class="accordion-header"
				onclick={() => toggleSection('weather')}
				aria-expanded={sections.weather}
			>
				<span class="mono-micro">04 // WEATHER</span>
				<span class="accordion-sign">{sections.weather ? '[ − ]' : '[ + ]'}</span>
			</button>

			{#if sections.weather}
				<div class="accordion-body">
					<StencilSlider
						label="Humidity"
						bind:value={appState.params.humidity}
						min={0}
						max={1}
						step={0.01}
						ticks={9}
						formatValue={formatPercent}
						onChange={markCustom}
					/>

					<StencilSlider
						label="Pressure"
						bind:value={appState.params.pressure}
						min={960}
						max={1040}
						step={1}
						unit="hPa"
						ticks={9}
						onChange={markCustom}
					/>

					<StencilSlider
						label="Temperature"
						bind:value={appState.params.temperature}
						min={-30}
						max={40}
						step={0.5}
						unit="°C"
						ticks={9}
						onChange={markCustom}
					/>

					<StencilSlider
						label="Wind Speed"
						bind:value={appState.params.windSpeed}
						min={0}
						max={40}
						step={0.5}
						unit="m/s"
						ticks={9}
						onChange={markCustom}
					/>

					<StencilSlider
						label="Wind Direction"
						bind:value={appState.params.windDirection}
						min={0}
						max={360}
						step={1}
						ticks={9}
						formatValue={formatWindDir}
						onChange={markCustom}
					/>

					<StencilSlider
						label="Turbulence"
						bind:value={appState.params.turbulence}
						min={0}
						max={1}
						step={0.01}
						ticks={9}
						formatValue={formatPercent}
						onChange={markCustom}
					/>

					<StencilSlider
						label="Coverage"
						bind:value={appState.params.coverage}
						min={0}
						max={1}
						step={0.01}
						ticks={9}
						formatValue={formatPercent}
						onChange={markCustom}
					/>
				</div>
			{/if}
		</section>

		<!-- 05. ATMOSPHERE & TRAFFIC -->
		<section class="accordion">
			<button
				type="button"
				class="accordion-header"
				onclick={() => toggleSection('atmosphere')}
				aria-expanded={sections.atmosphere}
			>
				<span class="mono-micro">05 // ATMOSPHERE & TRAFFIC</span>
				<span class="accordion-sign">{sections.atmosphere ? '[ − ]' : '[ + ]'}</span>
			</button>

			{#if sections.atmosphere}
				<div class="accordion-body">
					<div class="sub-label">WIND FLOW</div>
					<button
						type="button"
						class="stencil-btn toggle-row"
						class:is-active={appState.params.showWind}
						class:splat-shadow-sm={appState.params.showWind}
						aria-pressed={appState.params.showWind}
						onclick={() => (appState.params.showWind = !appState.params.showWind)}
					>
						<span>WIND CURRENTS</span>
						<span class="toggle-state">{appState.params.showWind ? '[ ON ]' : '[ OFF ]'}</span>
					</button>

					{#if appState.params.showWind}
						<StencilSlider
							label="Wind Streamline Density"
							bind:value={appState.params.windDensity}
							min={0.2}
							max={1.0}
							step={0.05}
							ticks={9}
							formatValue={formatPercent}
						/>
					{/if}

					<div class="sub-label">RAIN EVENTS</div>
					<div class="segmented segmented-3" role="group" aria-label="Rain event mode">
						{#each RAIN_MODES as mode}
							<button
								type="button"
								class="stencil-btn seg-btn"
								class:is-active={appState.params.rainMode === mode.id}
								onclick={() => setRainMode(mode.id)}
							>
								{mode.label}
							</button>
						{/each}
					</div>

					<StencilSlider
						label="Rain Intensity"
						bind:value={appState.params.rainIntensity}
						min={0}
						max={1}
						step={0.01}
						ticks={9}
						formatValue={formatPercent}
					/>

					<div class="sub-label">AIR TRAFFIC</div>
					<div class="toggle-list">
						<button
							type="button"
							class="stencil-btn toggle-row"
							class:is-active={appState.params.showPlanes}
							class:splat-shadow-sm={appState.params.showPlanes}
							aria-pressed={appState.params.showPlanes}
							onclick={() => (appState.params.showPlanes = !appState.params.showPlanes)}
						>
							<span>PLANES</span>
							<span class="toggle-state">{appState.params.showPlanes ? '[ ON ]' : '[ OFF ]'}</span>
						</button>

						<button
							type="button"
							class="stencil-btn toggle-row"
							class:is-active={appState.params.showHelicopters}
							class:splat-shadow-sm={appState.params.showHelicopters}
							aria-pressed={appState.params.showHelicopters}
							onclick={() => (appState.params.showHelicopters = !appState.params.showHelicopters)}
						>
							<span>HELICOPTERS</span>
							<span class="toggle-state">
								{appState.params.showHelicopters ? '[ ON ]' : '[ OFF ]'}
							</span>
						</button>

						<button
							type="button"
							class="stencil-btn toggle-row"
							class:is-active={appState.params.showBirds}
							class:splat-shadow-sm={appState.params.showBirds}
							aria-pressed={appState.params.showBirds}
							onclick={() => (appState.params.showBirds = !appState.params.showBirds)}
						>
							<span>BIRDS</span>
							<span class="toggle-state">{appState.params.showBirds ? '[ ON ]' : '[ OFF ]'}</span>
						</button>
					</div>
				</div>
			{/if}
		</section>

		<!-- 06. RENDER / SPLAT -->
		<section class="accordion">
			<button
				type="button"
				class="accordion-header"
				onclick={() => toggleSection('render')}
				aria-expanded={sections.render}
			>
				<span class="mono-micro">06 // RENDER / SPLAT</span>
				<span class="accordion-sign">{sections.render ? '[ − ]' : '[ + ]'}</span>
			</button>

			{#if sections.render}
				<div class="accordion-body">
					<StencilSlider
						label="Point Count"
						bind:value={appState.params.pointCount}
						min={15000}
						max={150000}
						step={2500}
						ticks={9}
						formatValue={formatPoints}
					/>

					<StencilSlider
						label="Point Size"
						bind:value={appState.params.pointSize}
						min={0.5}
						max={4.0}
						step={0.1}
						unit="px"
						ticks={8}
					/>

					<StencilSlider
						label="Shadow Dot Dither"
						bind:value={appState.params.shadowDither}
						min={0}
						max={1}
						step={0.01}
						ticks={9}
						formatValue={formatPercent}
					/>

					<div class="sub-label">MONOCHROME SURFACE</div>
					<div class="segmented">
						<button
							type="button"
							class="stencil-btn seg-btn"
							class:is-active={appState.params.surfaceMode === 'light'}
							onclick={() => setSurfaceMode('light')}
						>
							LIGHT PAPER
						</button>
						<button
							type="button"
							class="stencil-btn seg-btn"
							class:is-active={appState.params.surfaceMode === 'dark'}
							onclick={() => setSurfaceMode('dark')}
						>
							DARK INK
						</button>
					</div>

					<button
						type="button"
						class="stencil-btn reset-btn splat-shadow-sm"
						onclick={resetParams}
					>
						RESET DEFAULTS
					</button>
				</div>
			{/if}
		</section>
	</div>
</aside>

<style>
	.sidebar-backdrop {
		display: none;
	}

	.sidebar {
		position: fixed;
		top: 0;
		left: 0;
		bottom: 0;
		width: var(--sidebar-width);
		max-width: 88vw;
		z-index: 30;
		display: flex;
		flex-direction: column;
		background: var(--bg-panel);
		border-right: 1.5px solid var(--border);
		transform: translateX(-100%);
		transition: transform var(--transition-drawer);
		pointer-events: none;
		padding-top: env(safe-area-inset-top);
		padding-left: env(safe-area-inset-left);
	}

	.sidebar.is-open {
		transform: translateX(0);
		pointer-events: auto;
	}

	.sidebar :global(.splat-edge-right) {
		opacity: 0;
		transition: opacity var(--transition-drawer);
	}

	.sidebar.is-open :global(.splat-edge-right) {
		opacity: 1;
	}

	.sidebar-header {
		display: flex;
		align-items: center;
		justify-content: space-between;
		padding: 0.78rem 0.9rem;
		border-bottom: 1.5px solid var(--border);
		background-color: var(--bg-panel);
		flex-shrink: 0;
	}

	.brand {
		display: flex;
		flex-direction: column;
		gap: 0.12rem;
	}

	.brand-code {
		font-size: 0.82rem;
		font-weight: 700;
		letter-spacing: 0.14em;
		color: var(--ink-strong);
	}

	.brand-sub {
		font-size: 0.58rem;
		font-weight: 500;
		letter-spacing: 0.14em;
		color: var(--ink-soft);
	}

	.close-btn {
		padding: 0.28rem 0.5rem;
		font-size: 0.68rem;
	}

	.sidebar-body {
		flex: 1;
		overflow-y: auto;
		overflow-x: hidden;
		-webkit-overflow-scrolling: touch;
		overscroll-behavior: contain;
		touch-action: pan-y;
		padding-bottom: max(1.5rem, env(safe-area-inset-bottom));
	}

	.accordion {
		border-bottom: 1px solid var(--border-subtle);
	}

	.accordion-header {
		width: 100%;
		display: flex;
		align-items: center;
		justify-content: space-between;
		padding: 0.65rem 0.9rem;
		border: none;
		background: var(--bg-panel-muted);
		color: var(--ink-strong);
		cursor: pointer;
		text-align: left;
		transition: background-color 90ms ease;
	}

	.accordion-header:hover {
		background: var(--bg-hover);
	}

	.accordion-sign {
		font-size: 0.65rem;
		font-weight: 600;
		letter-spacing: 0.06em;
		color: var(--ink-soft);
	}

	.accordion-body {
		position: relative;
		z-index: 0;
		display: flex;
		flex-direction: column;
		gap: 0.72rem;
		padding: 0.8rem 0.9rem 0.95rem;
		background: var(--bg-panel);
	}

	.sub-label {
		font-size: 0.6rem;
		font-weight: 600;
		letter-spacing: 0.12em;
		color: var(--ink-faint);
		margin-bottom: -0.25rem;
	}

	.type-grid {
		display: grid;
		grid-template-columns: repeat(2, minmax(0, 1fr));
		gap: 0.42rem;
	}

	.type-chip {
		justify-content: flex-start;
		padding: 0.38rem 0.5rem;
		font-size: 0.64rem;
		letter-spacing: 0.05em;
	}

	.chip-code {
		font-weight: 700;
		opacity: 0.7;
		min-width: 1.75rem;
	}

	.chip-name {
		overflow: hidden;
		text-overflow: ellipsis;
		white-space: nowrap;
	}

	.preset-list {
		display: grid;
		grid-template-columns: 1fr;
		gap: 0.36rem;
	}

	.preset-chip {
		justify-content: flex-start;
		padding: 0.4rem 0.55rem;
		font-size: 0.66rem;
		letter-spacing: 0.06em;
	}

	.preset-code {
		font-weight: 700;
		opacity: 0.68;
		min-width: 3.1rem;
		text-align: left;
	}

	.preset-name {
		text-align: left;
	}

	.preset-desc {
		margin: 0.1rem 0 0;
		padding: 0.45rem 0.55rem;
		border-left: 2px solid var(--border);
		background: var(--bg-panel-muted);
		font-size: 0.64rem;
		line-height: 1.45;
		color: var(--ink-soft);
	}

	.inline-row {
		display: flex;
		align-items: center;
		justify-content: space-between;
		gap: 0.6rem;
	}

	.play-btn {
		flex: 1;
	}

	.time-readout {
		font-size: 0.68rem;
		font-weight: 600;
		letter-spacing: 0.08em;
		color: var(--ink-soft);
	}

	.seed-row {
		display: flex;
		align-items: stretch;
		gap: 0.45rem;
		margin-top: 0.15rem;
	}

	.seed-field {
		flex: 1;
		display: flex;
		align-items: center;
		border: 1px solid var(--border);
		background: var(--bg-panel);
		padding: 0 0.45rem;
		gap: 0.4rem;
	}

	.seed-label {
		font-size: 0.6rem;
		font-weight: 600;
		letter-spacing: 0.1em;
		color: var(--ink-faint);
	}

	.seed-input {
		width: 100%;
		border: none;
		background: transparent;
		font-size: 0.72rem;
		font-weight: 600;
		color: var(--ink-strong);
		padding: 0.32rem 0;
		outline: none;
	}

	.segmented {
		display: grid;
		grid-template-columns: repeat(2, 1fr);
		gap: 0;
	}

	.segmented-3 {
		grid-template-columns: repeat(3, 1fr);
	}

	.toggle-list {
		display: grid;
		grid-template-columns: 1fr;
		gap: 0.36rem;
	}

	.toggle-row {
		width: 100%;
		justify-content: space-between;
		padding: 0.38rem 0.55rem;
		font-size: 0.64rem;
		letter-spacing: 0.08em;
	}

	.toggle-state {
		font-weight: 700;
		letter-spacing: 0.06em;
	}

	.seg-btn {
		width: 100%;
	}

	.seg-btn + .seg-btn {
		border-left: none;
	}

	.reset-btn {
		width: 100%;
		margin-top: 0.25rem;
		padding: 0.5rem 0.7rem;
	}

	@media (max-width: 820px) {
		.sidebar-backdrop {
			display: block;
			position: fixed;
			inset: 0;
			z-index: 25;
			border: none;
			padding: 0;
			background: rgba(10, 10, 10, 0.28);
			cursor: pointer;
		}
	}

	@media (pointer: coarse), (max-width: 680px) {
		.accordion-header {
			min-height: 44px;
			padding: 0.75rem 0.9rem;
		}

		.type-chip,
		.preset-chip,
		.toggle-row,
		.seg-btn,
		.play-btn,
		.reset-btn,
		.close-btn {
			min-height: 38px;
		}
	}
</style>
