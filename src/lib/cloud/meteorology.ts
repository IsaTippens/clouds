/**
 * Meteorological calculations and automatic cloud morphology classification.
 */

import type { CloudParams, ConcreteCloudType } from '$lib/types';
import { clamp } from './noise';

export const CONCRETE_CLOUD_TYPES: readonly ConcreteCloudType[] = [
	'cumulus',
	'stratus',
	'cirrus',
	'cumulonimbus',
	'altocumulus'
] as const;

export interface MeteorologyState {
	dominantType: ConcreteCloudType;
	weights: Record<ConcreteCloudType, number>;
	baseAltitudeKm: number;
	topAltitudeKm: number;
	sunElevationDeg: number;
	sunAzimuthDeg: number;
	/** Unit vector pointing FROM origin TOWARD the sun in world coordinates (+Y up, -Z north, +X east) */
	sunDir: [number, number, number];
	dewPointC: number;
}

const DEG_TO_RAD = Math.PI / 180;
const RAD_TO_DEG = 180 / Math.PI;

/**
 * Magnus-Tetens dew point approximation (°C) from ambient temperature (°C) and relative humidity (0..1).
 */
export function computeDewPoint(temperatureC: number, humidity: number): number {
	const rh = clamp(humidity, 0.01, 1.0);
	const a = 17.625;
	const b = 243.04;
	const gamma = Math.log(rh) + (a * temperatureC) / (b + temperatureC);
	return (b * gamma) / (a - gamma);
}

/**
 * Compute solar elevation & azimuth (degrees) and unit sun direction vector.
 * Uses local solar time (`timeOfDay` in hours 0..24), geographic `latitude`,
 * and a subtle seasonal declination modulation derived from `longitude`.
 */
export function computeSunPosition(
	timeOfDay: number,
	latitude: number,
	longitude: number
): {
	sunElevationDeg: number;
	sunAzimuthDeg: number;
	sunDir: [number, number, number];
} {
	const latRad = clamp(latitude, -89.5, 89.5) * DEG_TO_RAD;
	// Gentle seasonal declination (-16° to +16°) seeded deterministically from longitude
	// so changing location subtly shifts sun arc while remaining physically realistic.
	const declDeg = Math.sin(longitude * DEG_TO_RAD * 0.5) * 14.0 + 6.0;
	const declRad = declDeg * DEG_TO_RAD;

	// Hour angle: 0 at solar noon (12:00), -90° at 06:00, +90° at 18:00
	const hourAngleRad = (timeOfDay - 12.0) * 15.0 * DEG_TO_RAD;

	const sinElev =
		Math.sin(latRad) * Math.sin(declRad) +
		Math.cos(latRad) * Math.cos(declRad) * Math.cos(hourAngleRad);
	const elevRad = Math.asin(clamp(sinElev, -1, 1));
	const cosElev = Math.max(Math.cos(elevRad), 1e-4);

	// Azimuth measured clockwise from North (0=N, 90=E, 180=S, 270=W)
	const cosAz = clamp(
		(Math.sin(declRad) - Math.sin(latRad) * sinElev) / (Math.cos(latRad) * cosElev),
		-1,
		1
	);
	let azDeg = Math.acos(cosAz) * RAD_TO_DEG;
	if (Math.sin(hourAngleRad) > 0) {
		azDeg = 360 - azDeg;
	}

	const azRad = azDeg * DEG_TO_RAD;
	// Clamp effective lighting elevation slightly above horizon so shadows remain well-defined
	// even near twilight, while reporting the true physical sunElevationDeg in telemetry.
	const litElevRad = Math.max(elevRad, 8 * DEG_TO_RAD);
	const cosLit = Math.cos(litElevRad);
	const sx = Math.sin(azRad) * cosLit; // +X = East
	const sy = Math.sin(litElevRad); // +Y = Up
	const sz = -Math.cos(azRad) * cosLit; // -Z = North

	const len = Math.hypot(sx, sy, sz) || 1;
	return {
		sunElevationDeg: elevRad * RAD_TO_DEG,
		sunAzimuthDeg: ((azDeg % 360) + 360) % 360,
		sunDir: [sx / len, sy / len, sz / len]
	};
}

/**
 * Gaussian affinity helper in [0, 1].
 */
function gaussian(x: number, mean: number, sigma: number): number {
	const d = (x - mean) / sigma;
	return Math.exp(-0.5 * d * d);
}

/**
 * Classify normalized cloud morphology weights from atmospheric parameters.
 */
export function classifyCloudWeights(
	params: CloudParams
): Record<ConcreteCloudType, number> {
	if (params.cloudType !== 'auto') {
		return {
			cumulus: params.cloudType === 'cumulus' ? 1 : 0,
			stratus: params.cloudType === 'stratus' ? 1 : 0,
			cirrus: params.cloudType === 'cirrus' ? 1 : 0,
			cumulonimbus: params.cloudType === 'cumulonimbus' ? 1 : 0,
			altocumulus: params.cloudType === 'altocumulus' ? 1 : 0
		};
	}

	const { altitude, humidity, pressure, temperature, turbulence, coverage, latitude, windSpeed } =
		params;
	const absLat = Math.abs(latitude);

	// 1. Cumulus: low-to-mid convective layer, warm, moderate humidity/turbulence, broken sky
	const cumulusScore =
		gaussian(altitude, 2.2, 1.5) *
		gaussian(humidity, 0.65, 0.24) *
		gaussian(turbulence, 0.42, 0.25) *
		gaussian(pressure, 1015, 16) *
		(0.55 + 0.45 * gaussian(temperature, 18, 14)) *
		(0.6 + 0.4 * gaussian(coverage, 0.55, 0.28));

	// 2. Stratus: low altitude, high humidity, low turbulence, high coverage, cool/stable
	const stratusScore =
		gaussian(altitude, 1.1, 1.1) *
		Math.pow(clamp(humidity, 0.1, 1), 1.4) *
		gaussian(turbulence, 0.14, 0.22) *
		(0.45 + 0.55 * clamp(coverage, 0, 1)) *
		(0.65 + 0.35 * gaussian(temperature, 10, 14));

	// 3. Cirrus: high altitude (>6.5 km), cold, drier, higher upper-level wind shear
	const cirrusScore =
		gaussian(altitude, 9.6, 2.3) *
		(0.4 + 0.6 * gaussian(temperature, -20, 22)) *
		(0.5 + 0.5 * gaussian(humidity, 0.4, 0.28)) *
		(0.6 + 0.4 * clamp(windSpeed / 30, 0, 1));

	// 4. Cumulonimbus: deep unstable convection, low pressure, high humidity, high turbulence, warm
	const lowPressureBoost = clamp((1012 - pressure) / 36, 0, 1);
	const tropicalConvection = 0.75 + 0.25 * gaussian(absLat, 25, 28);
	const cumulonimbusScore =
		gaussian(altitude, 5.2, 3.0) *
		Math.pow(clamp(humidity, 0.1, 1), 1.2) *
		(0.2 + 0.8 * clamp(turbulence, 0, 1)) *
		(0.25 + 0.75 * lowPressureBoost) *
		(0.55 + 0.45 * clamp((temperature + 5) / 35, 0, 1)) *
		tropicalConvection;

	// 5. Altocumulus: mid-altitude (3..6.5 km), moderate humidity & turbulence, banded coverage
	const altocumulusScore =
		gaussian(altitude, 4.7, 1.6) *
		gaussian(humidity, 0.68, 0.24) *
		gaussian(turbulence, 0.45, 0.25) *
		gaussian(pressure, 1010, 18) *
		(0.6 + 0.4 * gaussian(temperature, 4, 15));

	const raw: Record<ConcreteCloudType, number> = {
		cumulus: Math.max(0.0001, cumulusScore),
		stratus: Math.max(0.0001, stratusScore),
		cirrus: Math.max(0.0001, cirrusScore),
		cumulonimbus: Math.max(0.0001, cumulonimbusScore),
		altocumulus: Math.max(0.0001, altocumulusScore)
	};

	// Sharpen weights slightly so the dominant morphology reads clearly while still blending smoothly
	const power = 1.35;
	let sum = 0;
	for (const key of CONCRETE_CLOUD_TYPES) {
		raw[key] = Math.pow(raw[key], power);
		sum += raw[key];
	}

	const weights: Record<ConcreteCloudType, number> = {
		cumulus: raw.cumulus / sum,
		stratus: raw.stratus / sum,
		cirrus: raw.cirrus / sum,
		cumulonimbus: raw.cumulonimbus / sum,
		altocumulus: raw.altocumulus / sum
	};

	return weights;
}

/**
 * Compute full meteorological telemetry & solar lighting state from CloudParams.
 */
export function computeMeteorology(params: CloudParams): MeteorologyState {
	const weights = classifyCloudWeights(params);

	let dominantType: ConcreteCloudType = 'cumulus';
	let maxWeight = -1;
	for (const key of CONCRETE_CLOUD_TYPES) {
		if (weights[key] > maxWeight) {
			maxWeight = weights[key];
			dominantType = key;
		}
	}

	const dewPointC = computeDewPoint(params.temperature, params.humidity);
	const { sunElevationDeg, sunAzimuthDeg, sunDir } = computeSunPosition(
		params.timeOfDay,
		params.latitude,
		params.longitude
	);

	// Characteristic vertical thickness (km) per morphology
	const thicknessByType: Record<ConcreteCloudType, number> = {
		cumulus: 1.9 + params.turbulence * 1.4,
		stratus: 0.65 + params.humidity * 0.45,
		cirrus: 1.1 + params.windSpeed * 0.025,
		cumulonimbus: 7.4 + params.turbulence * 3.2,
		altocumulus: 1.2 + params.turbulence * 0.7
	};

	let weightedThickness = 0;
	for (const key of CONCRETE_CLOUD_TYPES) {
		weightedThickness += weights[key] * thicknessByType[key];
	}

	// Lifting condensation level influence on base altitude
	const lclOffset = clamp((params.temperature - dewPointC) * 0.035, 0, 1.2);
	const rawBase =
		weights.cumulonimbus > 0.5
			? clamp(params.altitude * 0.36 + lclOffset * 0.5, 0.7, 2.8)
			: clamp(params.altitude - weightedThickness * 0.35 + lclOffset * 0.25, 0.35, 11.2);

	const topAltitudeKm = clamp(rawBase + weightedThickness, rawBase + 0.4, 15.5);

	return {
		dominantType,
		weights,
		baseAltitudeKm: Math.round(rawBase * 100) / 100,
		topAltitudeKm: Math.round(topAltitudeKm * 100) / 100,
		sunElevationDeg: Math.round(sunElevationDeg * 10) / 10,
		sunAzimuthDeg: Math.round(sunAzimuthDeg * 10) / 10,
		sunDir,
		dewPointC: Math.round(dewPointC * 10) / 10
	};
}
