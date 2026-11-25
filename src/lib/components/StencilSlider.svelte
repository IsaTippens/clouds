<script lang="ts">
	interface Props {
		label: string;
		value: number;
		min: number;
		max: number;
		step?: number;
		unit?: string;
		ticks?: number;
		formatValue?: (value: number) => string;
		onChange?: (value: number) => void;
	}

	let {
		label,
		value = $bindable(),
		min,
		max,
		step = 1,
		unit = '',
		ticks = 9,
		formatValue,
		onChange
	}: Props = $props();

	let percent = $derived.by(() => {
		if (max <= min) return 0;
		const clamped = Math.min(max, Math.max(min, value));
		return ((clamped - min) / (max - min)) * 100;
	});

	let displayValue = $derived.by(() => {
		if (formatValue) return formatValue(value);
		const formatted = Number.isInteger(step) ? Math.round(value).toString() : value.toFixed(1);
		return unit ? `${formatted} ${unit}` : formatted;
	});

	let tickIndices = $derived(Array.from({ length: Math.max(2, ticks) }, (_, i) => i));

	function handleInput(event: Event) {
		const target = event.currentTarget as HTMLInputElement;
		const next = parseFloat(target.value);
		if (!Number.isNaN(next)) {
			value = next;
			onChange?.(next);
		}
	}
</script>

<div class="stencil-slider">
	<div class="slider-header">
		<span class="slider-label">{label}</span>
		<span class="slider-value tabular">{displayValue}</span>
	</div>

	<div class="slider-track-wrap">
		<div class="slider-track" aria-hidden="true">
			<div class="slider-fill" style="width: {percent}%"></div>
		</div>

		<div class="slider-ticks" aria-hidden="true">
			{#each tickIndices as idx}
				<span
					class="tick"
					class:tick-major={idx === 0 || idx === tickIndices.length - 1 || idx === Math.floor(tickIndices.length / 2)}
				></span>
			{/each}
		</div>

		<input
			type="range"
			class="slider-input"
			{min}
			{max}
			{step}
			{value}
			aria-label={label}
			oninput={handleInput}
		/>
	</div>
</div>

<style>
	.stencil-slider {
		display: flex;
		flex-direction: column;
		gap: 0.28rem;
	}

	.slider-header {
		display: flex;
		align-items: baseline;
		justify-content: space-between;
		gap: 0.5rem;
	}

	.slider-label {
		font-size: 0.65rem;
		font-weight: 500;
		letter-spacing: 0.08em;
		text-transform: uppercase;
		color: var(--ink-soft);
	}

	.slider-value {
		font-size: 0.72rem;
		font-weight: 600;
		letter-spacing: 0.04em;
		color: var(--ink-strong);
	}

	.slider-track-wrap {
		position: relative;
		height: 22px;
		display: flex;
		flex-direction: column;
		justify-content: center;
	}

	.slider-track {
		position: relative;
		width: 100%;
		height: 3px;
		background: var(--border-subtle);
	}

	.slider-fill {
		position: absolute;
		top: 0;
		left: 0;
		bottom: 0;
		background: var(--ink);
	}

	.slider-ticks {
		display: flex;
		justify-content: space-between;
		align-items: flex-start;
		margin-top: 3px;
		height: 5px;
		pointer-events: none;
	}

	.tick {
		width: 1px;
		height: 3px;
		background: var(--border-subtle);
	}

	.tick-major {
		height: 5px;
		background: var(--ink-faint);
	}

	.slider-input {
		position: absolute;
		inset: 0;
		width: 100%;
		height: 100%;
		margin: 0;
		opacity: 1;
		background: transparent;
		-webkit-appearance: none;
		appearance: none;
		cursor: pointer;
		touch-action: none;
	}

	.slider-input:focus-visible {
		outline: 1.5px solid var(--border);
		outline-offset: 2px;
	}

	.slider-input::-webkit-slider-runnable-track {
		height: 22px;
		background: transparent;
	}

	.slider-input::-webkit-slider-thumb {
		-webkit-appearance: none;
		appearance: none;
		width: 10px;
		height: 14px;
		margin-top: 1px;
		background: var(--surface-invert);
		border: 1.5px solid var(--bg-panel);
		outline: 1px solid var(--border);
		border-radius: 0;
		cursor: ew-resize;
	}

	.slider-input:hover::-webkit-slider-thumb {
		background: var(--bg-panel);
		border-color: var(--border);
		outline-color: var(--border);
	}

	.slider-input::-moz-range-track {
		height: 22px;
		background: transparent;
		border: none;
	}

	.slider-input::-moz-range-thumb {
		width: 10px;
		height: 14px;
		background: var(--surface-invert);
		border: 1.5px solid var(--bg-panel);
		outline: 1px solid var(--border);
		border-radius: 0;
		cursor: ew-resize;
	}

	.slider-input:hover::-moz-range-thumb {
		background: var(--bg-panel);
		border-color: var(--border);
	}

	@media (pointer: coarse), (max-width: 680px) {
		.slider-track-wrap {
			height: 34px;
		}

		.slider-input::-webkit-slider-runnable-track {
			height: 34px;
		}

		.slider-input::-webkit-slider-thumb {
			width: 16px;
			height: 22px;
			margin-top: 2px;
		}

		.slider-input::-moz-range-track {
			height: 34px;
		}

		.slider-input::-moz-range-thumb {
			width: 16px;
			height: 22px;
		}
	}
</style>
