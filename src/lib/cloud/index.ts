/**
 * Public entrypoint for the procedural monochrome Three.js cloud engine.
 */

export {
	createCloudViewport,
	type CameraPreset,
	type CloudViewportController
} from './scene';

export { computeMeteorology, classifyCloudWeights, computeDewPoint, computeSunPosition } from './meteorology';
export type { MeteorologyState } from './meteorology';
