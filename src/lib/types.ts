export type CloudType =
	| 'auto'
	| 'cumulus'
	| 'stratus'
	| 'cirrus'
	| 'cumulonimbus'
	| 'altocumulus';

export type ConcreteCloudType = Exclude<CloudType, 'auto'>;

export type SurfaceMode = 'light' | 'dark';

export type ThemePreference = 'auto' | SurfaceMode;

export type RainMode = 'auto' | 'on' | 'off';

export type PrecipPhase = 'dry' | 'rain' | 'snow';

export interface CloudParams {
	// Time
	/** Local solar time in hours (0..24) */
	timeOfDay: number;
	/** Time progression multiplier (0..3) */
	timeSpeed: number;
	/** Freeze simulation updates when true */
	paused: boolean;

	// Position
	/** Geographic latitude in degrees (-90..90) */
	latitude: number;
	/** Geographic longitude in degrees (-180..180) */
	longitude: number;
	/** Reference cloud layer altitude in km (0.5..12) */
	altitude: number;
	/** Deterministic spatial seed (integer) */
	regionSeed: number;

	// Weather
	/** Relative humidity (0..1) */
	humidity: number;
	/** Atmospheric pressure in hPa (960..1040) */
	pressure: number;
	/** Ambient temperature in °C (-30..40) */
	temperature: number;
	/** Wind velocity in m/s (0..40) */
	windSpeed: number;
	/** Wind azimuth in degrees (0..360) */
	windDirection: number;
	/** Atmospheric turbulence intensity (0..1) */
	turbulence: number;
	/** Sky cloud coverage fraction (0..1) */
	coverage: number;

	// Rendering
	/** Cloud morphology mode or automatic meteorological classification */
	cloudType: CloudType;
	/** Target point budget (15000..150000) */
	pointCount: number;
	/** Base screen-space point diameter (0.5..4.0) */
	pointSize: number;
	/** Ordered dither shadow strength (0..1) */
	shadowDither: number;
	/** Monochrome viewport theme */
	surfaceMode: SurfaceMode;

	// Atmosphere & Traffic
	/** Render 3D monochrome wind current streamlines */
	showWind: boolean;
	/** Streamline density fraction (0.2..1.0) */
	windDensity: number;
	/** Precipitation simulation mode */
	rainMode: RainMode;
	/** Rain rate intensity scale (0..1) */
	rainIntensity: number;
	/** Thunderstorm lightning simulation mode */
	lightningMode: RainMode;
	/** Tornado vortex simulation mode */
	tornadoMode: RainMode;
	/** Render high-altitude aircraft with contrails */
	showPlanes: boolean;
	/** Render low-altitude helicopters */
	showHelicopters: boolean;
	/** Render flocking bird formations */
	showBirds: boolean;
}

export interface CloudTelemetry {
	dominantType: Exclude<CloudType, 'auto'>;
	weights: Record<Exclude<CloudType, 'auto'>, number>;
	activePoints: number;
	baseAltitudeKm: number;
	topAltitudeKm: number;
	sunElevationDeg: number;
	sunAzimuthDeg: number;
	dewPointC: number;
	fps: number;
	rainActive: boolean;
	rainRateMmHr: number;
	precipPhase: PrecipPhase;
	lightningActive: boolean;
	tornadoActive: boolean;
	aircraftCount: number;
	birdCount: number;
}

export interface CloudPreset {
	id: string;
	name: string;
	code: string;
	description: string;
	params: Partial<CloudParams>;
}
