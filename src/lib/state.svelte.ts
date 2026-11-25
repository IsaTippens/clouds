import type { CloudParams, CloudPreset, CloudTelemetry, SurfaceMode, ThemePreference } from './types';

export const DEFAULT_PARAMS: CloudParams = {
	// Time
	timeOfDay: 14.5,
	timeSpeed: 0.4,
	paused: false,

	// Position
	latitude: 46.52,
	longitude: 7.98,
	altitude: 2.4,
	regionSeed: 40921,

	// Weather
	humidity: 0.68,
	pressure: 1014,
	temperature: 17,
	windSpeed: 9,
	windDirection: 235,
	turbulence: 0.38,
	coverage: 0.62,

	// Rendering
	cloudType: 'auto',
	pointCount: 75000,
	pointSize: 1.6,
	shadowDither: 0.65,
	surfaceMode: 'light',

	// Atmosphere & Traffic
	showWind: true,
	windDensity: 0.65,
	rainMode: 'auto',
	rainIntensity: 0.7,
	showPlanes: true,
	showHelicopters: true,
	showBirds: true
};

export const DEFAULT_TELEMETRY: CloudTelemetry = {
	dominantType: 'cumulus',
	weights: {
		cumulus: 1,
		stratus: 0,
		cirrus: 0,
		cumulonimbus: 0,
		altocumulus: 0
	},
	activePoints: 0,
	baseAltitudeKm: 1.8,
	topAltitudeKm: 3.4,
	sunElevationDeg: 42,
	sunAzimuthDeg: 210,
	dewPointC: 11.1,
	fps: 0,
	rainActive: false,
	rainRateMmHr: 0,
	aircraftCount: 0,
	birdCount: 0
};

export const PRESETS: CloudPreset[] = [
	{
		id: 'auto-weather',
		name: 'Auto Weather',
		code: 'MET-00',
		description: 'Dynamic classification driven by live humidity, pressure, altitude, and shear.',
		params: {
			cloudType: 'auto',
			timeOfDay: 14.5,
			altitude: 2.4,
			humidity: 0.68,
			pressure: 1014,
			temperature: 17,
			windSpeed: 9,
			windDirection: 235,
			turbulence: 0.38,
			coverage: 0.62,
			rainMode: 'auto',
			rainIntensity: 0.7
		}
	},
	{
		id: 'fair-cumulus',
		name: 'Fair Cumulus',
		code: 'CU-01',
		description: 'Buoyant afternoon convective towers over warm flat condensation bases.',
		params: {
			cloudType: 'cumulus',
			timeOfDay: 13.75,
			latitude: 38.9,
			longitude: -104.8,
			altitude: 2.1,
			humidity: 0.64,
			pressure: 1018,
			temperature: 22,
			windSpeed: 7,
			windDirection: 210,
			turbulence: 0.42,
			coverage: 0.54
		}
	},
	{
		id: 'marine-stratus',
		name: 'Marine Stratus',
		code: 'ST-02',
		description: 'Low, laterally uniform coastal deck trapped beneath a cool inversion.',
		params: {
			cloudType: 'stratus',
			timeOfDay: 8.25,
			latitude: 37.77,
			longitude: -122.42,
			altitude: 0.9,
			humidity: 0.91,
			pressure: 1016,
			temperature: 12,
			windSpeed: 11,
			windDirection: 295,
			turbulence: 0.14,
			coverage: 0.88,
			rainMode: 'auto',
			rainIntensity: 0.45
		}
	},
	{
		id: 'high-cirrus',
		name: 'High Cirrus',
		code: 'CI-03',
		description: 'Upper-tropospheric ice-crystal filaments drawn out by jet-stream shear.',
		params: {
			cloudType: 'cirrus',
			timeOfDay: 17.5,
			latitude: 52.3,
			longitude: -1.9,
			altitude: 9.8,
			humidity: 0.38,
			pressure: 1006,
			temperature: -24,
			windSpeed: 34,
			windDirection: 265,
			turbulence: 0.28,
			coverage: 0.45
		}
	},
	{
		id: 'supercell-anvil',
		name: 'Supercell Anvil',
		code: 'CB-04',
		description: 'Deep unstable updraft column spreading laterally against the tropopause.',
		params: {
			cloudType: 'cumulonimbus',
			timeOfDay: 16.25,
			latitude: 35.47,
			longitude: -97.52,
			altitude: 5.5,
			humidity: 0.94,
			pressure: 982,
			temperature: 28,
			windSpeed: 26,
			windDirection: 195,
			turbulence: 0.88,
			coverage: 0.84,
			rainMode: 'auto',
			rainIntensity: 0.92
		}
	},
	{
		id: 'alpine-altocumulus',
		name: 'Alpine Altocumulus',
		code: 'AC-05',
		description: 'Mid-level cellular wavelets sculpted by orographic flow across ridges.',
		params: {
			cloudType: 'altocumulus',
			timeOfDay: 10.5,
			latitude: 46.55,
			longitude: 7.98,
			altitude: 4.8,
			humidity: 0.72,
			pressure: 1011,
			temperature: 4,
			windSpeed: 18,
			windDirection: 280,
			turbulence: 0.46,
			coverage: 0.66
		}
	}
];

export const appState = $state<{
	params: CloudParams;
	telemetry: CloudTelemetry;
	menuOpen: boolean;
	activePresetId: string | null;
	themePreference: ThemePreference;
}>({
	params: { ...DEFAULT_PARAMS },
	telemetry: {
		...DEFAULT_TELEMETRY,
		weights: { ...DEFAULT_TELEMETRY.weights }
	},
	menuOpen: true,
	activePresetId: 'auto-weather',
	themePreference: 'auto'
});

const THEME_STORAGE_KEY = 'strata-theme-preference';

function resolveSystemSurfaceMode(): SurfaceMode {
	if (typeof window === 'undefined' || typeof window.matchMedia !== 'function') {
		return 'light';
	}
	return window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
}

export function setThemePreference(pref: ThemePreference): void {
	appState.themePreference = pref;
	appState.params.surfaceMode = pref === 'auto' ? resolveSystemSurfaceMode() : pref;
	if (typeof window !== 'undefined') {
		try {
			window.localStorage.setItem(THEME_STORAGE_KEY, pref);
		} catch {
			// Ignore storage quota / privacy mode errors
		}
	}
}

export function cycleThemePreference(): void {
	const order: ThemePreference[] = ['auto', 'light', 'dark'];
	const idx = order.indexOf(appState.themePreference);
	setThemePreference(order[(idx + 1) % order.length]);
}

export function initThemePreference(): () => void {
	if (typeof window === 'undefined') return () => {};

	let saved: ThemePreference = 'auto';
	try {
		const raw = window.localStorage.getItem(THEME_STORAGE_KEY);
		if (raw === 'auto' || raw === 'light' || raw === 'dark') {
			saved = raw;
		}
	} catch {
		// Ignore storage access errors
	}

	appState.themePreference = saved;
	appState.params.surfaceMode = saved === 'auto' ? resolveSystemSurfaceMode() : saved;

	if (typeof window.matchMedia !== 'function') return () => {};
	const media = window.matchMedia('(prefers-color-scheme: dark)');
	const handleSystemChange = (event: MediaQueryListEvent) => {
		if (appState.themePreference === 'auto') {
			appState.params.surfaceMode = event.matches ? 'dark' : 'light';
		}
	};
	media.addEventListener('change', handleSystemChange);
	return () => media.removeEventListener('change', handleSystemChange);
}

export function applyPreset(id: string): void {
	const preset = PRESETS.find((entry) => entry.id === id);
	if (!preset) return;
	Object.assign(appState.params, preset.params);
	appState.activePresetId = preset.id;
}

export function resetParams(): void {
	const currentSurface = appState.params.surfaceMode;
	Object.assign(appState.params, DEFAULT_PARAMS, { surfaceMode: currentSurface });
	appState.activePresetId = 'auto-weather';
}

export function randomizeLocation(): void {
	const lat = Math.round((Math.random() * 140 - 70) * 100) / 100;
	const lon = Math.round((Math.random() * 360 - 180) * 100) / 100;
	const seed = Math.floor(Math.random() * 900000) + 100000;
	appState.params.latitude = lat;
	appState.params.longitude = lon;
	appState.params.regionSeed = seed;
}
