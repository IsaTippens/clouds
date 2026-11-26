/**
 * Procedural monochrome air-traffic & flocking bird system for STRATA.
 *
 * Implements three independent atmospheric traffic layers rendered strictly
 * in black, grey, and white to match the technical stencil / point-cloud aesthetic:
 *  1. High-altitude swept-wing planes with twin procedural point-cloud contrails
 *     and technical radar altitude drop-ticks + ground-projection blips.
 *  2. Mid/low-altitude helicopters with spinning main/tail rotors, rotor disc
 *     rings, forward velocity pitch, and monochrome aviation strobes.
 *  3. Flocking bird formations (migrating V-formation, wheeling thermal flock,
 *     and low skimming echelon) with articulated flapping & gliding V-wings.
 */

import * as THREE from 'three';
import type { CloudParams, SurfaceMode } from '$lib/types';
import type { TornadoStatus } from './tornado';
import { clamp, createPrng, lerp, smoothstep } from './noise';

export interface TrafficStatus {
	aircraftCount: number;
	birdCount: number;
}

export interface AirTrafficSystem {
	update(
		dt: number,
		elapsed: number,
		params: CloudParams,
		cloudBaseY: number,
		cloudTopY: number,
		tornado?: TornadoStatus
	): TrafficStatus;
	dispose(): void;
}

// ---------------------------------------------------------------------------
// Constants & Monochrome Palette
// ---------------------------------------------------------------------------

const MAX_ACTIVE_ENTITIES = 3; // At most 3 spawned entities (planes, helis, bird flocks) at once
const PLANE_POOL_SIZE = 3;
const HELI_POOL_SIZE = 3;
const FLOCK_POOL_SIZE = 3;
const MAX_BIRDS_PER_FLOCK = 11;
const BIRD_POOL_TOTAL = FLOCK_POOL_SIZE * MAX_BIRDS_PER_FLOCK; // 33 max bird slots

const SPAWN_RADIUS = 36.0; // Distant perimeter where entities spawn and despawn

const CONTRAIL_PER_PLANE = 360; // 180 per engine
const CONTRAIL_TOTAL = PLANE_POOL_SIZE * CONTRAIL_PER_PLANE;
const CONTRAIL_LIFETIME = 7.8; // seconds of simTime
const CONTRAIL_EMIT_INTERVAL = 0.042; // seconds between twin-engine emissions

interface MonochromePalette {
	hullPrimary: number;
	hullSecondary: number;
	edgeLine: number;
	rotorDisc: number;
	dropLine: number;
	groundBlip: number;
	strobePeak: number;
	strobeIdle: number;
	birdPrimary: [number, number, number];
	birdSecondary: [number, number, number];
}

const LIGHT_PALETTE: MonochromePalette = {
	hullPrimary: 0x1c1c1c,
	hullSecondary: 0x424242,
	edgeLine: 0x050505,
	rotorDisc: 0x3e3e3e,
	dropLine: 0x6e6e6e,
	groundBlip: 0x222222,
	strobePeak: 0xffffff,
	strobeIdle: 0x2a2a2a,
	birdPrimary: [0.09, 0.09, 0.09],
	birdSecondary: [0.28, 0.28, 0.28]
};

const DARK_PALETTE: MonochromePalette = {
	hullPrimary: 0xe6e6e6,
	hullSecondary: 0xb0b0b0,
	edgeLine: 0xffffff,
	rotorDisc: 0xc8c8c8,
	dropLine: 0x868686,
	groundBlip: 0xdcdcdc,
	strobePeak: 0xffffff,
	strobeIdle: 0x4a4a4a,
	birdPrimary: [0.94, 0.94, 0.94],
	birdSecondary: [0.68, 0.68, 0.68]
};

// ---------------------------------------------------------------------------
// Contrail Point-Cloud Shaders (matches splatted dot-gradient aesthetic)
// ---------------------------------------------------------------------------

const CONTRAIL_VERTEX_SHADER = /* glsl */ `
attribute float aAge;
attribute float aSize;
attribute float aSeed;

uniform float uPixelRatio;

varying float vAge;
varying float vSeed;

void main() {
	vAge = aAge;
	vSeed = aSeed;

	if (aAge < 0.0 || aAge >= 1.0) {
		gl_Position = vec4(2.0, 2.0, 2.0, 1.0);
		gl_PointSize = 0.0;
		return;
	}

	vec4 mvPosition = modelViewMatrix * vec4(position, 1.0);
	gl_Position = projectionMatrix * mvPosition;

	float dist = max(-mvPosition.z, 1.5);
	float expand = mix(0.75, 2.55, pow(aAge, 0.7));
	float sizePx = aSize * expand * uPixelRatio * (32.0 / dist);
	gl_PointSize = clamp(sizePx, 1.2, 18.0);
}
`;

const CONTRAIL_FRAGMENT_SHADER = /* glsl */ `
uniform float uDarkMode;

varying float vAge;
varying float vSeed;

void main() {
	if (vAge < 0.0 || vAge >= 1.0) discard;

	vec2 uv = gl_PointCoord * 2.0 - 1.0;
	float r = length(uv);
	if (r > 1.0) discard;

	// Crisp splatted dot falloff with slight stipple dissipation at tail end
	float core = 1.0 - smoothstep(0.32, 0.96, r);
	float fadeIn = smoothstep(0.0, 0.045, vAge);
	float fadeOut = pow(1.0 - vAge, 1.45);

	// Subtle stochastic stipple breakup on aged contrail wisps
	float grain = fract(sin(dot(gl_FragCoord.xy + vSeed * 41.0, vec2(12.9898, 78.233))) * 43758.5453);
	if (vAge > 0.45 && grain < (vAge - 0.45) * 0.72) {
		discard;
	}

	float alpha = core * fadeIn * fadeOut * mix(0.68, 0.22, vAge);
	if (alpha <= 0.015) discard;

	// Strictly monochrome: charcoal ink on light paper, crisp silver-white on dark obsidian
	float lum = uDarkMode < 0.5
		? mix(0.18, 0.46, vAge)
		: mix(0.96, 0.62, vAge);

	gl_FragColor = vec4(vec3(lum), alpha);
}
`;

// ---------------------------------------------------------------------------
// Reusable Geometry Builders
// ---------------------------------------------------------------------------

/**
 * Helper to create a triangle mesh BufferGeometry from flat vertex coordinates.
 */
function buildTriangleGeometry(coords: number[]): THREE.BufferGeometry {
	const geo = new THREE.BufferGeometry();
	geo.setAttribute('position', new THREE.Float32BufferAttribute(coords, 3));
	geo.computeVertexNormals();
	return geo;
}

interface SharedMaterials {
	hullPrimary: THREE.MeshBasicMaterial;
	hullSecondary: THREE.MeshBasicMaterial;
	edgeLine: THREE.LineBasicMaterial;
	rotorDisc: THREE.MeshBasicMaterial;
	dropLine: THREE.LineBasicMaterial;
	groundBlip: THREE.LineBasicMaterial;
	strobeMats: THREE.MeshBasicMaterial[];
}

/**
 * Build a sleek swept-wing monochrome aircraft oriented along +Z (forward), +Y (up).
 * Total length ~ 1.78, wingspan ~ 1.66.
 */
function createPlaneMesh(
	mats: SharedMaterials,
	geometriesToDispose: THREE.BufferGeometry[]
): {
	group: THREE.Group;
	leftEngineLocal: THREE.Vector3;
	rightEngineLocal: THREE.Vector3;
} {
	const group = new THREE.Group();

	// 1. Sleek faceted fuselage (nose at +0.92, tail cone at -0.86)
	// Built as a clean 3-section diamond-profile loft: nose cone -> mid body -> tail taper
	const fCoords: number[] = [];
	const pushTri = (
		ax: number, ay: number, az: number,
		bx: number, by: number, bz: number,
		cx: number, cy: number, cz: number
	) => {
		fCoords.push(ax, ay, az, bx, by, bz, cx, cy, cz);
	};
	const pushQuad = (
		a: [number, number, number],
		b: [number, number, number],
		c: [number, number, number],
		d: [number, number, number]
	) => {
		fCoords.push(...a, ...b, ...c, ...a, ...c, ...d);
	};

	const nose: [number, number, number] = [0, -0.01, 0.94];
	const tail: [number, number, number] = [0, 0.04, -0.86];

	const r1: [number, number, number][] = [
		[0, 0.085, 0.42],
		[0.085, 0.0, 0.42],
		[0, -0.07, 0.42],
		[-0.085, 0.0, 0.42]
	];
	const r2: [number, number, number][] = [
		[0, 0.09, -0.34],
		[0.082, 0.0, -0.34],
		[0, -0.065, -0.34],
		[-0.082, 0.0, -0.34]
	];

	for (let i = 0; i < 4; i++) {
		const j = (i + 1) % 4;
		// Nose cone
		pushTri(nose[0], nose[1], nose[2], r1[j][0], r1[j][1], r1[j][2], r1[i][0], r1[i][1], r1[i][2]);
		// Mid fuselage
		pushQuad(r1[i], r1[j], r2[j], r2[i]);
		// Tail taper
		pushTri(tail[0], tail[1], tail[2], r2[i][0], r2[i][1], r2[i][2], r2[j][0], r2[j][1], r2[j][2]);
	}

	const fuselageGeo = buildTriangleGeometry(fCoords);
	const fuselageEdges = new THREE.EdgesGeometry(fuselageGeo, 20);
	geometriesToDispose.push(fuselageGeo, fuselageEdges);

	group.add(new THREE.Mesh(fuselageGeo, mats.hullPrimary));
	group.add(new THREE.LineSegments(fuselageEdges, mats.edgeLine));

	// 2. Swept-back main wings (left & right, slight upward dihedral)
	const wCoords: number[] = [];
	const addWing = (side: 1 | -1) => {
		const rootLead: [number, number, number] = [side * 0.07, -0.01, 0.28];
		const rootTrail: [number, number, number] = [side * 0.07, -0.015, -0.22];
		const tipLead: [number, number, number] = [side * 0.84, 0.055, -0.28];
		const tipTrail: [number, number, number] = [side * 0.84, 0.05, -0.44];
		// Subtle winglet fin
		const wingletTop: [number, number, number] = [side * 0.87, 0.17, -0.45];

		if (side === 1) {
			wCoords.push(...rootLead, ...tipLead, ...tipTrail, ...rootLead, ...tipTrail, ...rootTrail);
			wCoords.push(...tipLead, ...wingletTop, ...tipTrail);
		} else {
			wCoords.push(...rootLead, ...tipTrail, ...tipLead, ...rootLead, ...rootTrail, ...tipTrail);
			wCoords.push(...tipLead, ...tipTrail, ...wingletTop);
		}
	};
	addWing(1);
	addWing(-1);

	// Horizontal tail stabilizers
	const addTailplane = (side: 1 | -1) => {
		const rl: [number, number, number] = [side * 0.03, 0.035, -0.58];
		const rt: [number, number, number] = [side * 0.02, 0.035, -0.82];
		const tl: [number, number, number] = [side * 0.34, 0.06, -0.78];
		const tt: [number, number, number] = [side * 0.34, 0.06, -0.89];
		if (side === 1) {
			wCoords.push(...rl, ...tl, ...tt, ...rl, ...tt, ...rt);
		} else {
			wCoords.push(...rl, ...tt, ...tl, ...rl, ...rt, ...tt);
		}
	};
	addTailplane(1);
	addTailplane(-1);

	const wingsGeo = buildTriangleGeometry(wCoords);
	const wingsEdges = new THREE.EdgesGeometry(wingsGeo, 15);
	geometriesToDispose.push(wingsGeo, wingsEdges);

	group.add(new THREE.Mesh(wingsGeo, mats.hullSecondary));
	group.add(new THREE.LineSegments(wingsEdges, mats.edgeLine));

	// 3. Swept vertical tail fin
	const finCoords: number[] = [
		0, 0.05, -0.46,
		0, 0.36, -0.84,
		0, 0.36, -0.94,
		0, 0.05, -0.46,
		0, 0.36, -0.94,
		0, 0.04, -0.84
	];
	const finGeo = buildTriangleGeometry(finCoords);
	const finEdges = new THREE.EdgesGeometry(finGeo, 10);
	geometriesToDispose.push(finGeo, finEdges);

	group.add(new THREE.Mesh(finGeo, mats.hullPrimary));
	group.add(new THREE.LineSegments(finEdges, mats.edgeLine));

	// 4. Twin under-wing engine nacelles
	const nacelleGeo = new THREE.CylinderGeometry(0.038, 0.032, 0.24, 6);
	nacelleGeo.rotateX(Math.PI / 2);
	const nacelleEdges = new THREE.EdgesGeometry(nacelleGeo, 25);
	geometriesToDispose.push(nacelleGeo, nacelleEdges);

	const engineX = 0.32;
	const engineY = -0.045;
	const engineZ = 0.0;
	for (const side of [-1, 1] as const) {
		const pod = new THREE.Mesh(nacelleGeo, mats.hullPrimary);
		pod.position.set(side * engineX, engineY, engineZ);
		const podEdge = new THREE.LineSegments(nacelleEdges, mats.edgeLine);
		podEdge.position.copy(pod.position);
		group.add(pod, podEdge);
	}

	return {
		group,
		leftEngineLocal: new THREE.Vector3(-engineX, engineY, engineZ - 0.14),
		rightEngineLocal: new THREE.Vector3(engineX, engineY, engineZ - 0.14)
	};
}

/**
 * Build a monochrome technical rotorcraft oriented along +Z (forward), +Y (up).
 */
function createHelicopterMesh(
	mats: SharedMaterials,
	strobeMat: THREE.MeshBasicMaterial,
	geometriesToDispose: THREE.BufferGeometry[]
): {
	group: THREE.Group;
	mainRotor: THREE.Group;
	tailRotor: THREE.Group;
} {
	const group = new THREE.Group();

	// 1. Faceted cabin pod (slightly teardrop profile)
	const cabinGeo = new THREE.BoxGeometry(0.32, 0.28, 0.72);
	// Taper front nose vertices slightly for aerodynamic silhouette
	const posAttr = cabinGeo.attributes.position as THREE.BufferAttribute;
	for (let i = 0; i < posAttr.count; i++) {
		const z = posAttr.getZ(i);
		const y = posAttr.getY(i);
		if (z > 0.1) {
			posAttr.setX(i, posAttr.getX(i) * 0.74);
			if (y > 0) posAttr.setY(i, y * 0.68);
		}
	}
	cabinGeo.computeVertexNormals();
	const cabinEdges = new THREE.EdgesGeometry(cabinGeo, 20);
	geometriesToDispose.push(cabinGeo, cabinEdges);

	const cabinMesh = new THREE.Mesh(cabinGeo, mats.hullPrimary);
	cabinMesh.position.set(0, 0.04, 0.12);
	const cabinWire = new THREE.LineSegments(cabinEdges, mats.edgeLine);
	cabinWire.position.copy(cabinMesh.position);
	group.add(cabinMesh, cabinWire);

	// Upper turbine cowling
	const cowlGeo = new THREE.BoxGeometry(0.18, 0.12, 0.38);
	const cowlEdges = new THREE.EdgesGeometry(cowlGeo, 20);
	geometriesToDispose.push(cowlGeo, cowlEdges);
	const cowl = new THREE.Mesh(cowlGeo, mats.hullSecondary);
	cowl.position.set(0, 0.21, 0.02);
	const cowlWire = new THREE.LineSegments(cowlEdges, mats.edgeLine);
	cowlWire.position.copy(cowl.position);
	group.add(cowl, cowlWire);

	// 2. Tapered tail boom
	const boomGeo = new THREE.CylinderGeometry(0.032, 0.072, 0.92, 5);
	boomGeo.rotateX(Math.PI / 2);
	const boomEdges = new THREE.EdgesGeometry(boomGeo, 20);
	geometriesToDispose.push(boomGeo, boomEdges);
	const boom = new THREE.Mesh(boomGeo, mats.hullSecondary);
	boom.position.set(0, 0.1, -0.66);
	const boomWire = new THREE.LineSegments(boomEdges, mats.edgeLine);
	boomWire.position.copy(boom.position);
	group.add(boom, boomWire);

	// Tail fin + small horizontal stabilizer
	const tailFinGeo = buildTriangleGeometry([
		0, 0.04, -0.96,
		0, 0.34, -1.14,
		0, 0.34, -1.22,
		0, 0.04, -0.96,
		0, 0.34, -1.22,
		0, -0.06, -1.12
	]);
	const tailFinEdges = new THREE.EdgesGeometry(tailFinGeo, 10);
	geometriesToDispose.push(tailFinGeo, tailFinEdges);
	group.add(new THREE.Mesh(tailFinGeo, mats.hullPrimary));
	group.add(new THREE.LineSegments(tailFinEdges, mats.edgeLine));

	const hStabGeo = new THREE.BoxGeometry(0.32, 0.02, 0.09);
	geometriesToDispose.push(hStabGeo);
	const hStab = new THREE.Mesh(hStabGeo, mats.hullSecondary);
	hStab.position.set(0, 0.1, -0.88);
	group.add(hStab);

	// 3. Landing skids & struts (crisp stencil linework)
	const skidPts: number[] = [];
	const skidY = -0.22;
	for (const side of [-1, 1] as const) {
		const sx = side * 0.21;
		// Main horizontal skid bar + curved front tip
		skidPts.push(sx, skidY, -0.26, sx, skidY, 0.44);
		skidPts.push(sx, skidY, 0.44, sx, skidY + 0.065, 0.54);
		// Two angled cross-struts to cabin underside
		skidPts.push(sx, skidY, -0.1, side * 0.12, -0.09, -0.08);
		skidPts.push(sx, skidY, 0.26, side * 0.12, -0.09, 0.24);
	}
	const skidGeo = new THREE.BufferGeometry();
	skidGeo.setAttribute('position', new THREE.Float32BufferAttribute(skidPts, 3));
	geometriesToDispose.push(skidGeo);
	group.add(new THREE.LineSegments(skidGeo, mats.edgeLine));

	// 4. Main rotor mast + spinning blades + faint rotor disc ring
	const mastGeo = new THREE.CylinderGeometry(0.022, 0.026, 0.14, 6);
	geometriesToDispose.push(mastGeo);
	const mast = new THREE.Mesh(mastGeo, mats.hullPrimary);
	mast.position.set(0, 0.31, 0.06);
	group.add(mast);

	const mainRotor = new THREE.Group();
	mainRotor.position.set(0, 0.38, 0.06);

	const bladeGeo = new THREE.BoxGeometry(1.56, 0.012, 0.055);
	geometriesToDispose.push(bladeGeo);
	const bladeA = new THREE.Mesh(bladeGeo, mats.hullPrimary);
	const bladeB = new THREE.Mesh(bladeGeo, mats.hullPrimary);
	bladeB.rotation.y = Math.PI / 2;
	mainRotor.add(bladeA, bladeB);
	group.add(mainRotor);

	// Faint rotor disc ring
	const discGeo = new THREE.RingGeometry(0.68, 0.79, 36);
	discGeo.rotateX(-Math.PI / 2);
	geometriesToDispose.push(discGeo);
	const discMesh = new THREE.Mesh(discGeo, mats.rotorDisc);
	discMesh.position.set(0, 0.378, 0.06);
	group.add(discMesh);

	// 5. Spinning tail rotor
	const tailRotor = new THREE.Group();
	tailRotor.position.set(0.045, 0.16, -1.12);
	const tailBladeGeo = new THREE.BoxGeometry(0.01, 0.28, 0.028);
	geometriesToDispose.push(tailBladeGeo);
	const tbA = new THREE.Mesh(tailBladeGeo, mats.hullPrimary);
	const tbB = new THREE.Mesh(tailBladeGeo, mats.hullPrimary);
	tbB.rotation.x = Math.PI / 2;
	tailRotor.add(tbA, tbB);
	group.add(tailRotor);

	// 6. Monochrome aviation strobe beacons (tail fin crest & belly)
	const strobeGeo = new THREE.OctahedronGeometry(0.036, 0);
	geometriesToDispose.push(strobeGeo);
	const tailStrobe = new THREE.Mesh(strobeGeo, strobeMat);
	tailStrobe.position.set(0, 0.37, -1.18);
	const bellyStrobe = new THREE.Mesh(strobeGeo, strobeMat);
	bellyStrobe.position.set(0, -0.11, -0.02);
	group.add(tailStrobe, bellyStrobe);

	return { group, mainRotor, tailRotor };
}

// ---------------------------------------------------------------------------
// Transient Flight Path Types & Spawner Helpers
// ---------------------------------------------------------------------------

type FlightBehavior = 'straight' | 'wander' | 'small-orbit';
type EntityKind = 'plane' | 'helicopter' | 'birds';

interface TransientPath {
	curve: THREE.CatmullRomCurve3;
	duration: number;
	age: number;
	behavior: FlightBehavior;
}

interface TornadoAvoidState {
	avoidX: number;
	avoidY: number;
	avoidZ: number;
	trapped: boolean;
	trapAge: number;
	trapDuration: number;
	trapAngle: number;
	trapRadius: number;
	trapY: number;
}

interface PlaneSlot extends TornadoAvoidState {
	active: boolean;
	mesh: THREE.Group;
	leftEngineLocal: THREE.Vector3;
	rightEngineLocal: THREE.Vector3;
	path: TransientPath | null;
	emitAccum: number;
	contrailBaseIdx: number;
	contrailHead: number;
	bobPhase: number;
	pos: THREE.Vector3;
}

interface HeliSlot extends TornadoAvoidState {
	active: boolean;
	mesh: THREE.Group;
	mainRotor: THREE.Group;
	tailRotor: THREE.Group;
	strobeMat: THREE.MeshBasicMaterial;
	path: TransientPath | null;
	strobeOffset: number;
	bobPhase: number;
}

interface FlockBirdMember {
	wingFreq: number;
	wingPhase: number;
	glideOffset: number;
	scale: number;
	localForward: number;
	localRight: number;
	localUp: number;
	altPhase: number;
}

interface BirdFlockSlot extends TornadoAvoidState {
	active: boolean;
	path: TransientPath | null;
	birdCount: number;
	formation: 'v-formation' | 'echelon' | 'loose';
	members: FlockBirdMember[];
}

/**
 * Generate a finite Catmull-Rom flight path that spawns at the distant
 * perimeter (SPAWN_RADIUS), performs one of three transient behaviors
 * ('straight', 'wander', or 'small-orbit'), and exits back into the distance
 * where the entity despawns.
 */
function buildTransientFlightPath(
	prng: () => number,
	behavior: FlightBehavior,
	baseY: number,
	altSpread: number,
	cruiseSpeed: number
): TransientPath {
	const pts: THREE.Vector3[] = [];
	const entryAngle = prng() * Math.PI * 2;
	const entryX = Math.cos(entryAngle) * SPAWN_RADIUS;
	const entryZ = Math.sin(entryAngle) * SPAWN_RADIUS;

	if (behavior === 'straight') {
		// Fly straight through with a gentle chord offset across the airspace
		const exitAngle = entryAngle + Math.PI + (prng() - 0.5) * 0.65;
		const exitX = Math.cos(exitAngle) * SPAWN_RADIUS;
		const exitZ = Math.sin(exitAngle) * SPAWN_RADIUS;

		const midX = (entryX + exitX) * 0.5 + (prng() - 0.5) * 10.0;
		const midZ = (entryZ + exitZ) * 0.5 + (prng() - 0.5) * 10.0;

		const y0 = baseY + (prng() - 0.5) * altSpread;
		const y1 = baseY + (prng() - 0.5) * altSpread;
		const y2 = baseY + (prng() - 0.5) * altSpread;

		pts.push(
			new THREE.Vector3(entryX, y0, entryZ),
			new THREE.Vector3(lerp(entryX, midX, 0.6), lerp(y0, y1, 0.5), lerp(entryZ, midZ, 0.6)),
			new THREE.Vector3(lerp(midX, exitX, 0.4), lerp(y1, y2, 0.5), lerp(midZ, exitZ, 0.4)),
			new THREE.Vector3(exitX, y2, exitZ)
		);
	} else if (behavior === 'wander') {
		// Enter from distance, meander through 3 interior waypoints, then depart
		pts.push(new THREE.Vector3(entryX, baseY + (prng() - 0.5) * altSpread, entryZ));

		const waypointCount = 3;
		let curAngle = entryAngle + Math.PI;
		for (let i = 0; i < waypointCount; i++) {
			const frac = (i + 1) / (waypointCount + 1);
			const r = lerp(14.0, 4.5, Math.sin(frac * Math.PI)) + (prng() - 0.5) * 5.0;
			curAngle += (prng() - 0.5) * 1.45;
			const wx = Math.cos(curAngle) * r + (prng() - 0.5) * 4.0;
			const wz = Math.sin(curAngle) * r + (prng() - 0.5) * 4.0;
			const wy = baseY + (prng() - 0.5) * altSpread * 1.3;
			pts.push(new THREE.Vector3(wx, wy, wz));
		}

		const exitAngle = curAngle + (prng() - 0.5) * 1.1;
		pts.push(
			new THREE.Vector3(
				Math.cos(exitAngle) * SPAWN_RADIUS,
				baseY + (prng() - 0.5) * altSpread,
				Math.sin(exitAngle) * SPAWN_RADIUS
			)
		);
	} else {
		// 'small-orbit': Approach -> 1.0 to 1.35 turns of a small localized orbit -> Depart
		const cx = (prng() - 0.5) * 9.0;
		const cz = (prng() - 0.5) * 9.0;
		const orbitR = 4.2 + prng() * 3.0;
		const cw: 1 | -1 = prng() < 0.5 ? 1 : -1;

		pts.push(new THREE.Vector3(entryX, baseY + (prng() - 0.5) * altSpread * 0.6, entryZ));

		// Tangent entry angle around the orbit center
		const toCenterAngle = Math.atan2(entryZ - cz, entryX - cx);
		const startTheta = toCenterAngle - cw * 0.65;
		const totalTurn = (1.0 + prng() * 0.35) * Math.PI * 2;
		const orbitSteps = 8;

		for (let s = 0; s <= orbitSteps; s++) {
			const frac = s / orbitSteps;
			const theta = startTheta + cw * frac * totalTurn;
			const r = orbitR + Math.sin(frac * Math.PI * 2) * 0.45;
			const oy = baseY + Math.sin(frac * Math.PI) * altSpread * 0.7;
			pts.push(new THREE.Vector3(cx + Math.cos(theta) * r, oy, cz + Math.sin(theta) * r));
		}

		// Depart smoothly along the exit tangent into the distance
		const endTheta = startTheta + cw * totalTurn;
		const tangentAngle = endTheta + cw * (Math.PI * 0.5);
		const exitX = cx + Math.cos(tangentAngle) * SPAWN_RADIUS;
		const exitZ = cz + Math.sin(tangentAngle) * SPAWN_RADIUS;
		pts.push(new THREE.Vector3(exitX, baseY + (prng() - 0.5) * altSpread * 0.6, exitZ));
	}

	const curve = new THREE.CatmullRomCurve3(pts, false, 'centripetal', 0.5);
	const arcLen = curve.getLength();
	const duration = Math.max(8.0, arcLen / Math.max(0.8, cruiseSpeed));

	return {
		curve,
		duration,
		age: 0,
		behavior
	};
}

// ---------------------------------------------------------------------------
// Main Factory
// ---------------------------------------------------------------------------

export function createAirTrafficSystem(scene: THREE.Scene): AirTrafficSystem {
	const root = new THREE.Group();
	root.name = 'air-traffic-root';
	scene.add(root);

	const planesGroup = new THREE.Group();
	const helicoptersGroup = new THREE.Group();
	const birdsGroup = new THREE.Group();
	root.add(planesGroup, helicoptersGroup, birdsGroup);

	const geometriesToDispose: THREE.BufferGeometry[] = [];
	const materialsToDispose: THREE.Material[] = [];

	// Shared monochrome materials
	const sharedMats: SharedMaterials = {
		hullPrimary: new THREE.MeshBasicMaterial({
			color: LIGHT_PALETTE.hullPrimary,
			side: THREE.DoubleSide
		}),
		hullSecondary: new THREE.MeshBasicMaterial({
			color: LIGHT_PALETTE.hullSecondary,
			side: THREE.DoubleSide
		}),
		edgeLine: new THREE.LineBasicMaterial({
			color: LIGHT_PALETTE.edgeLine,
			transparent: true,
			opacity: 0.9
		}),
		rotorDisc: new THREE.MeshBasicMaterial({
			color: LIGHT_PALETTE.rotorDisc,
			transparent: true,
			opacity: 0.36,
			side: THREE.DoubleSide,
			depthWrite: false
		}),
		dropLine: new THREE.LineBasicMaterial({
			color: LIGHT_PALETTE.dropLine,
			transparent: true,
			opacity: 0.36,
			depthWrite: false
		}),
		groundBlip: new THREE.LineBasicMaterial({
			color: LIGHT_PALETTE.groundBlip,
			transparent: true,
			opacity: 0.68,
			depthWrite: false
		}),
		strobeMats: []
	};
	materialsToDispose.push(
		sharedMats.hullPrimary,
		sharedMats.hullSecondary,
		sharedMats.edgeLine,
		sharedMats.rotorDisc,
		sharedMats.dropLine,
		sharedMats.groundBlip
	);

	const prng = createPrng(902107);

	// -------------------------------------------------------------------------
	// 1. Plane Pool (up to 3 slots) + Contrails + Radar Altitude Drop-Ticks
	// -------------------------------------------------------------------------
	const planes: PlaneSlot[] = Array.from({ length: PLANE_POOL_SIZE }, (_, idx) => {
		const built = createPlaneMesh(sharedMats, geometriesToDispose);
		built.group.visible = false;
		planesGroup.add(built.group);
		return {
			active: false,
			mesh: built.group,
			leftEngineLocal: built.leftEngineLocal,
			rightEngineLocal: built.rightEngineLocal,
			path: null,
			emitAccum: 0,
			contrailBaseIdx: idx * CONTRAIL_PER_PLANE,
			contrailHead: 0,
			bobPhase: prng() * Math.PI * 2,
			pos: new THREE.Vector3(),
			avoidX: 0,
			avoidY: 0,
			avoidZ: 0,
			trapped: false,
			trapAge: 0,
			trapDuration: 2.6,
			trapAngle: 0,
			trapRadius: 3.0,
			trapY: 10.0
		};
	});

	// Contrail particle ring buffer
	const contrailPositions = new Float32Array(CONTRAIL_TOTAL * 3);
	const contrailAges = new Float32Array(CONTRAIL_TOTAL);
	const contrailSizes = new Float32Array(CONTRAIL_TOTAL);
	const contrailSeeds = new Float32Array(CONTRAIL_TOTAL);
	const contrailSpreadVel = new Float32Array(CONTRAIL_TOTAL * 3);

	for (let i = 0; i < CONTRAIL_TOTAL; i++) {
		contrailAges[i] = -1; // inactive
		contrailSizes[i] = 1.45 + prng() * 0.9;
		contrailSeeds[i] = prng();
	}

	const contrailGeo = new THREE.BufferGeometry();
	const contrailPosAttr = new THREE.BufferAttribute(contrailPositions, 3);
	const contrailAgeAttr = new THREE.BufferAttribute(contrailAges, 1);
	contrailPosAttr.setUsage(THREE.DynamicDrawUsage);
	contrailAgeAttr.setUsage(THREE.DynamicDrawUsage);
	contrailGeo.setAttribute('position', contrailPosAttr);
	contrailGeo.setAttribute('aAge', contrailAgeAttr);
	contrailGeo.setAttribute('aSize', new THREE.BufferAttribute(contrailSizes, 1));
	contrailGeo.setAttribute('aSeed', new THREE.BufferAttribute(contrailSeeds, 1));
	geometriesToDispose.push(contrailGeo);

	const contrailUniforms = {
		uPixelRatio: {
			value: typeof window !== 'undefined' ? Math.min(window.devicePixelRatio || 1, 2) : 1
		},
		uDarkMode: { value: 0.0 }
	};
	const contrailMat = new THREE.ShaderMaterial({
		uniforms: contrailUniforms,
		vertexShader: CONTRAIL_VERTEX_SHADER,
		fragmentShader: CONTRAIL_FRAGMENT_SHADER,
		transparent: true,
		depthWrite: false,
		depthTest: true
	});
	materialsToDispose.push(contrailMat);

	const contrailPoints = new THREE.Points(contrailGeo, contrailMat);
	contrailPoints.frustumCulled = false;
	planesGroup.add(contrailPoints);

	// Vertical altitude drop-ticks (1 vertical line + 3 small altitude notches per plane)
	const DROP_SEGS_PER_PLANE = 4;
	const dropLinePositions = new Float32Array(PLANE_POOL_SIZE * DROP_SEGS_PER_PLANE * 2 * 3);
	const dropLineGeo = new THREE.BufferGeometry();
	const dropLinePosAttr = new THREE.BufferAttribute(dropLinePositions, 3);
	dropLinePosAttr.setUsage(THREE.DynamicDrawUsage);
	dropLineGeo.setAttribute('position', dropLinePosAttr);
	geometriesToDispose.push(dropLineGeo);

	const dropLines = new THREE.LineSegments(dropLineGeo, sharedMats.dropLine);
	dropLines.frustumCulled = false;
	planesGroup.add(dropLines);

	// Ground-projection stencil diamond + crosshair blip (6 segments per plane)
	const BLIP_SEGS_PER_PLANE = 6;
	const groundBlipPositions = new Float32Array(PLANE_POOL_SIZE * BLIP_SEGS_PER_PLANE * 2 * 3);
	const groundBlipGeo = new THREE.BufferGeometry();
	const groundBlipPosAttr = new THREE.BufferAttribute(groundBlipPositions, 3);
	groundBlipPosAttr.setUsage(THREE.DynamicDrawUsage);
	groundBlipGeo.setAttribute('position', groundBlipPosAttr);
	geometriesToDispose.push(groundBlipGeo);

	const groundBlips = new THREE.LineSegments(groundBlipGeo, sharedMats.groundBlip);
	groundBlips.frustumCulled = false;
	planesGroup.add(groundBlips);

	// -------------------------------------------------------------------------
	// 2. Helicopter Pool (up to 3 slots)
	// -------------------------------------------------------------------------
	const helicopters: HeliSlot[] = Array.from({ length: HELI_POOL_SIZE }, (_, idx) => {
		const strobeMat = new THREE.MeshBasicMaterial({
			color: LIGHT_PALETTE.strobeIdle
		});
		sharedMats.strobeMats.push(strobeMat);
		materialsToDispose.push(strobeMat);

		const built = createHelicopterMesh(sharedMats, strobeMat, geometriesToDispose);
		built.group.visible = false;
		helicoptersGroup.add(built.group);
		return {
			active: false,
			mesh: built.group,
			mainRotor: built.mainRotor,
			tailRotor: built.tailRotor,
			strobeMat,
			path: null,
			strobeOffset: idx * 0.33,
			bobPhase: prng() * Math.PI * 2,
			avoidX: 0,
			avoidY: 0,
			avoidZ: 0,
			trapped: false,
			trapAge: 0,
			trapDuration: 2.5,
			trapAngle: 0,
			trapRadius: 3.0,
			trapY: 3.0
		};
	});

	// -------------------------------------------------------------------------
	// 3. Bird Flock Pool (up to 3 flock slots, 7..11 birds per flock)
	// -------------------------------------------------------------------------
	function buildFlockMembers(formation: 'v-formation' | 'echelon' | 'loose', count: number): FlockBirdMember[] {
		const members: FlockBirdMember[] = [];
		for (let i = 0; i < count; i++) {
			const rank = i === 0 ? 0 : Math.ceil(i / 2);
			const side = i === 0 ? 0 : i % 2 === 1 ? -1 : 1;
			let localForward: number;
			let localRight: number;
			let localUp: number;
			if (formation === 'v-formation') {
				localForward = -rank * 0.66 + (prng() - 0.5) * 0.14;
				localRight = side * rank * 0.54 + (prng() - 0.5) * 0.14;
				localUp = (prng() - 0.5) * 0.18;
			} else if (formation === 'echelon') {
				localForward = -i * 0.48 + (prng() - 0.5) * 0.16;
				localRight = i * 0.38 - count * 0.18 + (prng() - 0.5) * 0.18;
				localUp = (prng() - 0.5) * 0.2;
			} else {
				// Loose naturalistic cluster
				localForward = (prng() - 0.5) * 3.4;
				localRight = (prng() - 0.5) * 3.4;
				localUp = (prng() - 0.5) * 1.1;
			}
			members.push({
				wingFreq: 7.6 + (prng() - 0.5) * 1.5,
				wingPhase: prng() * Math.PI * 2,
				glideOffset: prng() * Math.PI * 2,
				scale: 0.84 + prng() * 0.22,
				localForward,
				localRight,
				localUp,
				altPhase: prng() * Math.PI * 2
			});
		}
		return members;
	}

	const birdFlocks: BirdFlockSlot[] = Array.from({ length: FLOCK_POOL_SIZE }, () => ({
		active: false,
		path: null,
		birdCount: 0,
		formation: 'v-formation',
		members: [],
		avoidX: 0,
		avoidY: 0,
		avoidZ: 0,
		trapped: false,
		trapAge: 0,
		trapDuration: 2.4,
		trapAngle: 0,
		trapRadius: 3.0,
		trapY: 3.0
	}));

	// 6 triangles per bird = 18 vertices per bird
	const VERTS_PER_BIRD = 18;
	const birdPositions = new Float32Array(BIRD_POOL_TOTAL * VERTS_PER_BIRD * 3);
	const birdColors = new Float32Array(BIRD_POOL_TOTAL * VERTS_PER_BIRD * 3);

	const birdGeo = new THREE.BufferGeometry();
	const birdPosAttr = new THREE.BufferAttribute(birdPositions, 3);
	const birdColAttr = new THREE.BufferAttribute(birdColors, 3);
	birdPosAttr.setUsage(THREE.DynamicDrawUsage);
	birdGeo.setAttribute('position', birdPosAttr);
	birdGeo.setAttribute('color', birdColAttr);
	geometriesToDispose.push(birdGeo);

	const birdMat = new THREE.MeshBasicMaterial({
		vertexColors: true,
		side: THREE.DoubleSide
	});
	materialsToDispose.push(birdMat);

	const birdMesh = new THREE.Mesh(birdGeo, birdMat);
	birdMesh.frustumCulled = false;
	birdsGroup.add(birdMesh);

	// -------------------------------------------------------------------------
	// Reusable Scratch Vectors & Matrices (zero per-frame heap allocations)
	// -------------------------------------------------------------------------
	const _p0 = new THREE.Vector3();
	const _pPrev = new THREE.Vector3();
	const _pNext = new THREE.Vector3();
	const _vel = new THREE.Vector3();
	const _acc = new THREE.Vector3();
	const _fwd = new THREE.Vector3();
	const _right = new THREE.Vector3();
	const _up = new THREE.Vector3();
	const _worldUp = new THREE.Vector3(0, 1, 0);
	const _basisMat = new THREE.Matrix4();
	const _quat = new THREE.Quaternion();
	const _rollQuat = new THREE.Quaternion();
	const _pitchQuat = new THREE.Quaternion();
	const _engineWorld = new THREE.Vector3();
	const _localX = new THREE.Vector3(1, 0, 0);

	// -------------------------------------------------------------------------
	// Theme Synchronization
	// -------------------------------------------------------------------------
	let lastSurfaceMode: SurfaceMode | null = null;

	function applySurfaceMode(mode: SurfaceMode): void {
		if (lastSurfaceMode === mode) return;
		lastSurfaceMode = mode;

		const pal = mode === 'dark' ? DARK_PALETTE : LIGHT_PALETTE;
		sharedMats.hullPrimary.color.setHex(pal.hullPrimary);
		sharedMats.hullSecondary.color.setHex(pal.hullSecondary);
		sharedMats.edgeLine.color.setHex(pal.edgeLine);
		sharedMats.rotorDisc.color.setHex(pal.rotorDisc);
		sharedMats.dropLine.color.setHex(pal.dropLine);
		sharedMats.groundBlip.color.setHex(pal.groundBlip);
		contrailUniforms.uDarkMode.value = mode === 'dark' ? 1.0 : 0.0;

		const [pR, pG, pB] = pal.birdPrimary;
		const [sR, sG, sB] = pal.birdSecondary;
		for (let b = 0; b < BIRD_POOL_TOTAL; b++) {
			const base = b * VERTS_PER_BIRD * 3;
			for (let v = 0; v < VERTS_PER_BIRD; v++) {
				const isOuter = v >= 12;
				const idx = base + v * 3;
				birdColors[idx] = isOuter ? sR : pR;
				birdColors[idx + 1] = isOuter ? sG : pG;
				birdColors[idx + 2] = isOuter ? sB : pB;
			}
		}
		birdColAttr.needsUpdate = true;
	}

	// -------------------------------------------------------------------------
	// Contrail Emission Helper
	// -------------------------------------------------------------------------
	function emitContrailPair(slot: PlaneSlot): void {
		const engines = [slot.leftEngineLocal, slot.rightEngineLocal];
		for (let e = 0; e < 2; e++) {
			const idx = slot.contrailBaseIdx + slot.contrailHead;
			slot.contrailHead = (slot.contrailHead + 1) % CONTRAIL_PER_PLANE;

			_engineWorld.copy(engines[e]).applyMatrix4(slot.mesh.matrixWorld);

			const i3 = idx * 3;
			contrailPositions[i3] = _engineWorld.x;
			contrailPositions[i3 + 1] = _engineWorld.y;
			contrailPositions[i3 + 2] = _engineWorld.z;
			contrailAges[idx] = 0.001;

			const seed = contrailSeeds[idx] * 6.2831853;
			contrailSpreadVel[i3] = Math.cos(seed) * 0.09;
			contrailSpreadVel[i3 + 1] = (Math.sin(seed * 2.3) * 0.5 - 0.15) * 0.06;
			contrailSpreadVel[i3 + 2] = Math.sin(seed) * 0.09;
		}
	}

	function orientAircraft(
		group: THREE.Group,
		pos: THREE.Vector3,
		prev: THREE.Vector3,
		next: THREE.Vector3,
		eps: number,
		bankScale: number,
		maxBank: number,
		extraPitch: number
	): void {
		_vel.subVectors(next, prev).multiplyScalar(0.5 / eps);
		if (_vel.lengthSq() < 1e-6) return;

		_fwd.copy(_vel).normalize();
		_acc.copy(next).add(prev).addScaledVector(pos, -2.0).multiplyScalar(1.0 / (eps * eps));

		_right.crossVectors(_worldUp, _fwd);
		if (_right.lengthSq() < 1e-6) {
			_right.set(1, 0, 0);
		} else {
			_right.normalize();
		}
		_up.crossVectors(_fwd, _right).normalize();

		const lateralAcc = _acc.dot(_right);
		const bankAngle = clamp(-lateralAcc * bankScale, -maxBank, maxBank);

		_basisMat.makeBasis(_right, _up, _fwd);
		_quat.setFromRotationMatrix(_basisMat);

		if (Math.abs(bankAngle) > 1e-4) {
			_rollQuat.setFromAxisAngle(_fwd, bankAngle);
			_quat.premultiply(_rollQuat);
		}
		if (Math.abs(extraPitch) > 1e-4) {
			_pitchQuat.setFromAxisAngle(_localX, extraPitch);
			_quat.multiply(_pitchQuat);
		}

		group.position.copy(pos);
		group.quaternion.copy(_quat);
	}

	// -------------------------------------------------------------------------
	// Occasional Spawner State (Global Cap: MAX_ACTIVE_ENTITIES = 3)
	// -------------------------------------------------------------------------
	let simTime = 0;
	let initialSeeded = false;
	let spawnCooldown = 2.4;

	function countActiveEntities(): number {
		let count = 0;
		for (let i = 0; i < PLANE_POOL_SIZE; i++) if (planes[i].active) count++;
		for (let i = 0; i < HELI_POOL_SIZE; i++) if (helicopters[i].active) count++;
		for (let i = 0; i < FLOCK_POOL_SIZE; i++) if (birdFlocks[i].active) count++;
		return count;
	}

	function pickBehavior(kind: EntityKind): FlightBehavior {
		const r = prng();
		if (kind === 'plane') {
			// Planes mostly fly straight through or gently wander, occasionally doing a holding turn
			if (r < 0.55) return 'straight';
			if (r < 0.82) return 'wander';
			return 'small-orbit';
		}
		if (kind === 'helicopter') {
			// Helicopters split across straight transit, wandering patrol, and a short orbit
			if (r < 0.34) return 'straight';
			if (r < 0.68) return 'wander';
			return 'small-orbit';
		}
		// Birds split across migrating straight through, wandering, and brief thermal circling
		if (r < 0.38) return 'straight';
		if (r < 0.72) return 'wander';
		return 'small-orbit';
	}

	function resetTornadoState(state: TornadoAvoidState): void {
		state.avoidX = 0;
		state.avoidY = 0;
		state.avoidZ = 0;
		state.trapped = false;
		state.trapAge = 0;
		state.trapDuration = 2.4;
		state.trapAngle = 0;
		state.trapRadius = 3.0;
		state.trapY = 3.0;
	}

	/**
	 * Evaluate tornado avoidance steering OR inner vortex capture + spiral drag-in.
	 * Returns:
	 *  - 'ok': entity continues normally (with smooth avoidance offset applied to p0/pPrev/pNext)
	 *  - 'trapped': entity is spiraling into the tornado core (p0/pPrev/pNext overwritten with spiral)
	 *  - 'despawn': entity was dragged completely into the vortex and should despawn now
	 */
	function stepTornadoInteraction(
		state: TornadoAvoidState,
		pos: THREE.Vector3,
		prev: THREE.Vector3,
		next: THREE.Vector3,
		simDt: number,
		tornado: TornadoStatus | undefined,
		avoidRadius: number,
		captureRadius: number,
		maxHazardHeight: number,
		evadeStrength: number
	): { mode: 'ok' | 'trapped' | 'despawn'; scaleFade: number } {
		const hasTornado = Boolean(tornado && tornado.tornadoActive && tornado.funnelStrength > 0.12);

		if (state.trapped) {
			if (!hasTornado || !tornado) {
				// Tornado dissipated before consuming entity — despawn cleanly
				return { mode: 'despawn', scaleFade: 0 };
			}
			state.trapAge += simDt;
			const frac = clamp(state.trapAge / state.trapDuration, 0, 1);
			if (frac >= 1.0) {
				return { mode: 'despawn', scaleFade: 0 };
			}

			// Tightening accelerating cyclonic spiral pulled inward & upward into the funnel core
			const targetR = lerp(state.trapRadius, 0.28, clamp(simDt * 1.35, 0, 1));
			state.trapRadius = Math.max(0.24, targetR);
			const angularSpeed = (3.4 + (3.2 / Math.max(state.trapRadius, 0.45))) * (0.65 + tornado.funnelStrength * 0.55);
			state.trapAngle += angularSpeed * simDt;
			state.trapY = lerp(state.trapY, tornado.topY + 0.85, clamp(simDt * 0.85, 0, 1));

			const evalSpiral = (aOff: number, rOff: number, yOff: number, out: THREE.Vector3) => {
				const a = state.trapAngle + aOff;
				const r = Math.max(0.18, state.trapRadius + rOff);
				out.set(
					tornado.anchorX + Math.cos(a) * r,
					Math.max(0.45, state.trapY + yOff),
					tornado.anchorZ + Math.sin(a) * r
				);
			};
			const dA = angularSpeed * 0.04;
			evalSpiral(0, 0, 0, pos);
			evalSpiral(-dA, 0.08, -0.06, prev);
			evalSpiral(dA, -0.08, 0.06, next);

			const fade = 1.0 - smoothstep(0.68, 1.0, frac);
			return { mode: 'trapped', scaleFade: fade };
		}

		// Apply existing smooth avoidance offset before checking distance
		const curX = pos.x + state.avoidX;
		const curY = pos.y + state.avoidY;
		const curZ = pos.z + state.avoidZ;

		let targetAvoidX = 0;
		let targetAvoidY = 0;
		let targetAvoidZ = 0;

		if (hasTornado && tornado && curY <= maxHazardHeight) {
			const dx = curX - tornado.anchorX;
			const dz = curZ - tornado.anchorZ;
			const dist = Math.hypot(dx, dz);

			// Inner capture zone: caught by vortex suction!
			if (dist < captureRadius && tornado.funnelStrength > 0.32) {
				state.trapped = true;
				state.trapAge = 0;
				state.trapDuration = 2.1 + prng() * 0.9;
				state.trapAngle = Math.atan2(dz, dx);
				state.trapRadius = Math.max(0.6, dist);
				state.trapY = curY;
				return { mode: 'trapped', scaleFade: 1 };
			}

			// Outer avoidance zone: actively steer laterally & climb away from funnel path
			if (dist < avoidRadius) {
				const proximity = 1.0 - smoothstep(captureRadius * 0.9, avoidRadius, dist);
				const safeD = Math.max(dist, 0.4);
				const awayX = dx / safeD;
				const awayZ = dz / safeD;
				// Add a subtle tangential bias so entities bank smoothly around the tornado rather than stalling
				const tangX = -awayZ;
				const tangZ = awayX;
				const push = proximity * evadeStrength * tornado.funnelStrength;
				targetAvoidX = (awayX * 0.82 + tangX * 0.58) * push;
				targetAvoidZ = (awayZ * 0.82 + tangZ * 0.58) * push;
				targetAvoidY = proximity * 1.45 * tornado.funnelStrength;
			}
		}

		const blend = clamp(simDt * 2.2, 0, 1);
		state.avoidX = lerp(state.avoidX, targetAvoidX, blend);
		state.avoidY = lerp(state.avoidY, targetAvoidY, blend);
		state.avoidZ = lerp(state.avoidZ, targetAvoidZ, blend);

		pos.x += state.avoidX;
		pos.y += state.avoidY;
		pos.z += state.avoidZ;
		prev.x += state.avoidX;
		prev.y += state.avoidY;
		prev.z += state.avoidZ;
		next.x += state.avoidX;
		next.y += state.avoidY;
		next.z += state.avoidZ;

		return { mode: 'ok', scaleFade: 1 };
	}

	function spawnEntity(
		kind: EntityKind,
		cloudBaseY: number,
		cloudTopY: number,
		initialProgress = 0
	): boolean {
		if (countActiveEntities() >= MAX_ACTIVE_ENTITIES) return false;

		if (kind === 'plane') {
			const slot = planes.find((p) => !p.active);
			if (!slot) return false;
			const behavior = pickBehavior('plane');
			const basePlaneY = clamp(Math.max(cloudTopY + 1.4, 10.8) + prng() * 3.4, 10.8, 16.2);
			const cruiseSpeed = 3.8 + prng() * 1.4;
			const path = buildTransientFlightPath(prng, behavior, basePlaneY, 0.9, cruiseSpeed);
			path.age = clamp(initialProgress, 0, 0.75) * path.duration;
			slot.active = true;
			slot.path = path;
			slot.emitAccum = 0;
			slot.mesh.visible = true;
			resetTornadoState(slot);
			return true;
		}

		if (kind === 'helicopter') {
			const slot = helicopters.find((h) => !h.active);
			if (!slot) return false;
			const behavior = pickBehavior('helicopter');
			const minY = Math.max(2.2, cloudBaseY * 0.65);
			const maxY = Math.max(minY + 1.0, cloudBaseY + 1.4);
			const baseY = lerp(minY, maxY, prng());
			const cruiseSpeed = 2.6 + prng() * 1.1;
			const path = buildTransientFlightPath(prng, behavior, baseY, 0.9, cruiseSpeed);
			path.age = clamp(initialProgress, 0, 0.75) * path.duration;
			slot.active = true;
			slot.path = path;
			slot.mesh.visible = true;
			resetTornadoState(slot);
			return true;
		}

		// Bird flock
		const slot = birdFlocks.find((f) => !f.active);
		if (!slot) return false;
		const behavior = pickBehavior('birds');
		const baseY = clamp(lerp(1.9, Math.max(3.6, cloudBaseY * 0.85), prng()), 1.8, 5.4);
		const cruiseSpeed = 2.2 + prng() * 0.9;
		const path = buildTransientFlightPath(prng, behavior, baseY, 1.0, cruiseSpeed);
		path.age = clamp(initialProgress, 0, 0.75) * path.duration;

		const formations: Array<'v-formation' | 'echelon' | 'loose'> = [
			'v-formation',
			'echelon',
			'loose'
		];
		const formation =
			behavior === 'straight'
				? formations[prng() < 0.7 ? 0 : 1]
				: formations[Math.floor(prng() * formations.length)];
		const count = 6 + Math.floor(prng() * (MAX_BIRDS_PER_FLOCK - 5)); // 6..11 birds

		slot.active = true;
		slot.path = path;
		slot.formation = formation;
		slot.birdCount = count;
		slot.members = buildFlockMembers(formation, count);
		resetTornadoState(slot);
		return true;
	}

	function clearPlaneOverlaySlot(p: number): void {
		const dlBase = p * DROP_SEGS_PER_PLANE * 6;
		for (let i = 0; i < DROP_SEGS_PER_PLANE * 6; i++) {
			dropLinePositions[dlBase + i] = 0;
		}
		const gbBase = p * BLIP_SEGS_PER_PLANE * 6;
		for (let i = 0; i < BLIP_SEGS_PER_PLANE * 6; i++) {
			groundBlipPositions[gbBase + i] = 0;
		}
	}

	// -------------------------------------------------------------------------
	// Update Loop
	// -------------------------------------------------------------------------
	function update(
		dt: number,
		_elapsed: number,
		params: CloudParams,
		cloudBaseY: number,
		cloudTopY: number,
		tornado?: TornadoStatus
	): TrafficStatus {
		applySurfaceMode(params.surfaceMode);

		const clampedDt = clamp(dt, 0, 0.1);
		const simDt = params.paused ? 0 : clampedDt * params.timeSpeed;
		simTime += simDt;

		const windRad = (params.windDirection * Math.PI) / 180;
		const windDirX = Math.sin(windRad);
		const windDirZ = -Math.cos(windRad);
		const windDriftSpeed = params.windSpeed * 0.045;

		// Visibility toggles
		planesGroup.visible = params.showPlanes;
		helicoptersGroup.visible = params.showHelicopters;
		birdsGroup.visible = params.showBirds;

		// Despawn any active entities whose category was toggled off
		if (!params.showPlanes) {
			for (let p = 0; p < PLANE_POOL_SIZE; p++) {
				if (planes[p].active) {
					planes[p].active = false;
					planes[p].path = null;
					planes[p].mesh.visible = false;
					clearPlaneOverlaySlot(p);
				}
			}
		}
		if (!params.showHelicopters) {
			for (let h = 0; h < HELI_POOL_SIZE; h++) {
				if (helicopters[h].active) {
					helicopters[h].active = false;
					helicopters[h].path = null;
					helicopters[h].mesh.visible = false;
				}
			}
		}
		if (!params.showBirds) {
			for (let f = 0; f < FLOCK_POOL_SIZE; f++) {
				if (birdFlocks[f].active) {
					birdFlocks[f].active = false;
					birdFlocks[f].path = null;
					birdFlocks[f].birdCount = 0;
				}
			}
		}

		// Collect currently enabled categories
		const enabledKinds: EntityKind[] = [];
		if (params.showPlanes) enabledKinds.push('plane');
		if (params.showHelicopters) enabledKinds.push('helicopter');
		if (params.showBirds) enabledKinds.push('birds');

		// Seed 2 initial transient visitors mid-transit on first frame so airspace feels alive
		if (!initialSeeded && enabledKinds.length > 0) {
			initialSeeded = true;
			const firstKind = enabledKinds[0];
			spawnEntity(firstKind, cloudBaseY, cloudTopY, 0.28);
			if (enabledKinds.length > 1) {
				spawnEntity(enabledKinds[1], cloudBaseY, cloudTopY, 0.42);
			}
			spawnCooldown = 3.2 + prng() * 3.8;
		}

		// Occasional spawning up to MAX_ACTIVE_ENTITIES (at most 3 total)
		if (simDt > 0 && enabledKinds.length > 0 && countActiveEntities() < MAX_ACTIVE_ENTITIES) {
			spawnCooldown -= simDt;
			if (spawnCooldown <= 0) {
				const kind = enabledKinds[Math.floor(prng() * enabledKinds.length)];
				spawnEntity(kind, cloudBaseY, cloudTopY, 0);
				// Occasional staggered interval (3.5..9.5 sim seconds; longer when busier)
				const activeNow = countActiveEntities();
				spawnCooldown = (3.5 + prng() * 5.0) * (activeNow >= 2 ? 1.45 : 1.0);
			}
		}

		// ---------------------------------------------------------------------
		// 1. Update Active Planes, Contrails, & Radar Drop-Ticks
		// ---------------------------------------------------------------------
		let activePlanes = 0;
		const pathEps = 0.004;

		for (let p = 0; p < PLANE_POOL_SIZE; p++) {
			const slot = planes[p];
			if (!slot.active || !slot.path) {
				slot.mesh.visible = false;
				clearPlaneOverlaySlot(p);
				continue;
			}

			slot.path.age += simDt;
			const u = slot.path.age / slot.path.duration;
			if (u >= 1.0) {
				slot.active = false;
				slot.path = null;
				slot.mesh.visible = false;
				clearPlaneOverlaySlot(p);
				continue;
			}

			activePlanes++;
			const uPrev = Math.max(0, u - pathEps);
			const uNext = Math.min(1, u + pathEps);
			const dtSpan = (uNext - uPrev) * 0.5 * slot.path.duration;

			slot.path.curve.getPoint(u, _p0);
			slot.path.curve.getPoint(uPrev, _pPrev);
			slot.path.curve.getPoint(uNext, _pNext);

			const bobY = Math.sin(simTime * 0.35 + slot.bobPhase) * 0.14;
			_p0.y += bobY;
			_pPrev.y += bobY;
			_pNext.y += bobY;

			// Tornado avoidance / vortex capture (planes try to skirt around storm columns; lower ones can get pulled in)
			const tRes = stepTornadoInteraction(
				slot,
				_p0,
				_pPrev,
				_pNext,
				simDt,
				tornado,
				11.0,
				3.4,
				(tornado?.topY ?? 5.0) + 9.5,
				4.8
			);
			if (tRes.mode === 'despawn') {
				slot.active = false;
				slot.path = null;
				slot.mesh.visible = false;
				clearPlaneOverlaySlot(p);
				continue;
			}

			// Smooth horizon emergence & distant / vortex fade-out
			const edgeFade =
				smoothstep(0.0, 0.07, u) * (1.0 - smoothstep(0.93, 1.0, u)) * tRes.scaleFade;
			slot.mesh.visible = edgeFade > 0.01;
			slot.mesh.scale.setScalar(Math.max(0.01, edgeFade));

			const bankMax = tRes.mode === 'trapped' ? 1.15 : 0.44;
			orientAircraft(
				slot.mesh,
				_p0,
				_pPrev,
				_pNext,
				Math.max(0.02, tRes.mode === 'trapped' ? 0.04 : dtSpan),
				tRes.mode === 'trapped' ? 0.38 : 0.22,
				bankMax,
				tRes.mode === 'trapped' ? Math.sin(simTime * 7.0) * 0.35 : 0
			);
			slot.pos.copy(_p0);

			// Emit contrail particles along twin engines while inside visible sky & not trapped
			if (simDt > 0 && edgeFade > 0.35 && tRes.mode !== 'trapped') {
				slot.mesh.updateMatrixWorld(true);
				slot.emitAccum += simDt;
				while (slot.emitAccum >= CONTRAIL_EMIT_INTERVAL) {
					slot.emitAccum -= CONTRAIL_EMIT_INTERVAL;
					emitContrailPair(slot);
				}
			}

			// Radar drop-line & ground blip (fade out cleanly near distant horizon)
			if (edgeFade > 0.45) {
				const dlBase = p * DROP_SEGS_PER_PLANE * 6;
				const px = _p0.x;
				const py = _p0.y;
				const pz = _p0.z;
				const groundY = 0.03;

				dropLinePositions[dlBase] = px;
				dropLinePositions[dlBase + 1] = py - 0.18;
				dropLinePositions[dlBase + 2] = pz;
				dropLinePositions[dlBase + 3] = px;
				dropLinePositions[dlBase + 4] = groundY;
				dropLinePositions[dlBase + 5] = pz;

				const tickHalf = 0.16;
				for (let tk = 1; tk <= 3; tk++) {
					const ty = py * (tk / 4);
					const off = dlBase + tk * 6;
					dropLinePositions[off] = px - tickHalf;
					dropLinePositions[off + 1] = ty;
					dropLinePositions[off + 2] = pz;
					dropLinePositions[off + 3] = px + tickHalf;
					dropLinePositions[off + 4] = ty;
					dropLinePositions[off + 5] = pz;
				}

				const gbBase = p * BLIP_SEGS_PER_PLANE * 6;
				const r = 0.34;
				const cr = 0.18;
				const by = 0.028;

				groundBlipPositions[gbBase] = px;
				groundBlipPositions[gbBase + 1] = by;
				groundBlipPositions[gbBase + 2] = pz - r;
				groundBlipPositions[gbBase + 3] = px + r;
				groundBlipPositions[gbBase + 4] = by;
				groundBlipPositions[gbBase + 5] = pz;

				groundBlipPositions[gbBase + 6] = px + r;
				groundBlipPositions[gbBase + 7] = by;
				groundBlipPositions[gbBase + 8] = pz;
				groundBlipPositions[gbBase + 9] = px;
				groundBlipPositions[gbBase + 10] = by;
				groundBlipPositions[gbBase + 11] = pz + r;

				groundBlipPositions[gbBase + 12] = px;
				groundBlipPositions[gbBase + 13] = by;
				groundBlipPositions[gbBase + 14] = pz + r;
				groundBlipPositions[gbBase + 15] = px - r;
				groundBlipPositions[gbBase + 16] = by;
				groundBlipPositions[gbBase + 17] = pz;

				groundBlipPositions[gbBase + 18] = px - r;
				groundBlipPositions[gbBase + 19] = by;
				groundBlipPositions[gbBase + 20] = pz;
				groundBlipPositions[gbBase + 21] = px;
				groundBlipPositions[gbBase + 22] = by;
				groundBlipPositions[gbBase + 23] = pz - r;

				groundBlipPositions[gbBase + 24] = px - cr;
				groundBlipPositions[gbBase + 25] = by;
				groundBlipPositions[gbBase + 26] = pz;
				groundBlipPositions[gbBase + 27] = px + cr;
				groundBlipPositions[gbBase + 28] = by;
				groundBlipPositions[gbBase + 29] = pz;

				groundBlipPositions[gbBase + 30] = px;
				groundBlipPositions[gbBase + 31] = by;
				groundBlipPositions[gbBase + 32] = pz - cr;
				groundBlipPositions[gbBase + 33] = px;
				groundBlipPositions[gbBase + 34] = by;
				groundBlipPositions[gbBase + 35] = pz + cr;
			} else {
				clearPlaneOverlaySlot(p);
			}
		}

		dropLinePosAttr.needsUpdate = true;
		groundBlipPosAttr.needsUpdate = true;

		// Always advect & age existing contrail particles even after a plane despawns
		if (params.showPlanes && simDt > 0) {
			const ageStep = simDt / CONTRAIL_LIFETIME;
			const driftX = windDirX * windDriftSpeed * simDt;
			const driftZ = windDirZ * windDriftSpeed * simDt;
			for (let i = 0; i < CONTRAIL_TOTAL; i++) {
				const age = contrailAges[i];
				if (age < 0) continue;
				const nextAge = age + ageStep;
				if (nextAge >= 1.0) {
					contrailAges[i] = -1;
					continue;
				}
				contrailAges[i] = nextAge;
				const i3 = i * 3;
				contrailPositions[i3] += driftX + contrailSpreadVel[i3] * simDt;
				contrailPositions[i3 + 1] += contrailSpreadVel[i3 + 1] * simDt;
				contrailPositions[i3 + 2] += driftZ + contrailSpreadVel[i3 + 2] * simDt;
			}
			contrailPosAttr.needsUpdate = true;
			contrailAgeAttr.needsUpdate = true;
		}

		// ---------------------------------------------------------------------
		// 2. Update Active Helicopters
		// ---------------------------------------------------------------------
		let activeHelis = 0;
		const pal = params.surfaceMode === 'dark' ? DARK_PALETTE : LIGHT_PALETTE;

		for (let h = 0; h < HELI_POOL_SIZE; h++) {
			const slot = helicopters[h];
			if (!slot.active || !slot.path) {
				slot.mesh.visible = false;
				continue;
			}

			slot.path.age += simDt;
			const u = slot.path.age / slot.path.duration;
			if (u >= 1.0) {
				slot.active = false;
				slot.path = null;
				slot.mesh.visible = false;
				continue;
			}

			activeHelis++;
			const uPrev = Math.max(0, u - pathEps);
			const uNext = Math.min(1, u + pathEps);
			const dtSpan = (uNext - uPrev) * 0.5 * slot.path.duration;

			slot.path.curve.getPoint(u, _p0);
			slot.path.curve.getPoint(uPrev, _pPrev);
			slot.path.curve.getPoint(uNext, _pNext);

			const bob = Math.sin(simTime * 1.15 + slot.bobPhase) * 0.14;
			_p0.y += bob;
			_pPrev.y += bob;
			_pNext.y += bob;

			// Tornado avoidance / vortex capture for low-flying helicopters
			const tRes = stepTornadoInteraction(
				slot,
				_p0,
				_pPrev,
				_pNext,
				simDt,
				tornado,
				10.5,
				3.9,
				(tornado?.topY ?? 5.0) + 4.0,
				5.2
			);
			if (tRes.mode === 'despawn') {
				slot.active = false;
				slot.path = null;
				slot.mesh.visible = false;
				continue;
			}

			const edgeFade =
				smoothstep(0.0, 0.07, u) * (1.0 - smoothstep(0.93, 1.0, u)) * tRes.scaleFade;
			slot.mesh.visible = edgeFade > 0.01;
			slot.mesh.scale.setScalar(Math.max(0.01, edgeFade));

			// Pitch slightly forward (+0.14 rad around local +X tilts +Z nose downward); tumble if trapped
			orientAircraft(
				slot.mesh,
				_p0,
				_pPrev,
				_pNext,
				Math.max(0.02, tRes.mode === 'trapped' ? 0.04 : dtSpan),
				tRes.mode === 'trapped' ? 0.42 : 0.18,
				tRes.mode === 'trapped' ? 1.1 : 0.28,
				tRes.mode === 'trapped' ? Math.sin(simTime * 9.0) * 0.45 : 0.14
			);

			if (simDt > 0) {
				slot.mainRotor.rotation.y += simDt * 24.0;
				slot.tailRotor.rotation.x += simDt * 38.0;
			}

			const cycle = (simTime * 0.85 + slot.strobeOffset) % 1.0;
			const flash = (cycle > 0.0 && cycle < 0.065) || (cycle > 0.13 && cycle < 0.195);
			slot.strobeMat.color.setHex(flash ? pal.strobePeak : pal.strobeIdle);
		}

		// ---------------------------------------------------------------------
		// 3. Update Active Bird Flocks
		// ---------------------------------------------------------------------
		let activeBirdTotal = 0;
		let birdWriteIdx = 0;

		for (let f = 0; f < FLOCK_POOL_SIZE; f++) {
			const flock = birdFlocks[f];
			if (!flock.active || !flock.path) continue;

			flock.path.age += simDt;
			const u = flock.path.age / flock.path.duration;
			if (u >= 1.0) {
				flock.active = false;
				flock.path = null;
				flock.birdCount = 0;
				continue;
			}

			const uPrev = Math.max(0, u - pathEps);
			const uNext = Math.min(1, u + pathEps);
			const dtSpan = Math.max(0.02, (uNext - uPrev) * 0.5 * flock.path.duration);

			flock.path.curve.getPoint(u, _vel); // store lead center in _vel
			flock.path.curve.getPoint(uPrev, _pPrev);
			flock.path.curve.getPoint(uNext, _pNext);

			// Tornado avoidance / vortex capture for bird flocks
			const tRes = stepTornadoInteraction(
				flock,
				_vel,
				_pPrev,
				_pNext,
				simDt,
				tornado,
				9.5,
				4.0,
				(tornado?.topY ?? 5.0) + 3.0,
				4.8
			);
			if (tRes.mode === 'despawn') {
				flock.active = false;
				flock.path = null;
				flock.birdCount = 0;
				continue;
			}

			const edgeFade =
				smoothstep(0.0, 0.07, u) * (1.0 - smoothstep(0.93, 1.0, u)) * tRes.scaleFade;
			const spanForAcc = tRes.mode === 'trapped' ? 0.04 : dtSpan;

			_fwd.subVectors(_pNext, _pPrev).normalize();
			_right.crossVectors(_worldUp, _fwd);
			if (_right.lengthSq() < 1e-6) {
				_right.set(1, 0, 0);
			} else {
				_right.normalize();
			}
			_up.crossVectors(_fwd, _right).normalize();

			_acc
				.copy(_pNext)
				.add(_pPrev)
				.addScaledVector(_vel, -2.0)
				.multiplyScalar(1.0 / (spanForAcc * spanForAcc));

			const latAcc = _acc.dot(_right);
			const bank = clamp(-latAcc * 0.25, -0.56, 0.56);
			const cb = Math.cos(bank);
			const sb = Math.sin(bank);

			const rx = _right.x * cb + _up.x * sb;
			const ry = _right.y * cb + _up.y * sb;
			const rz = _right.z * cb + _up.z * sb;

			const ux = _up.x * cb - _right.x * sb;
			const uy = _up.y * cb - _right.y * sb;
			const uz = _up.z * cb - _right.z * sb;

			const fx = _fwd.x;
			const fy = _fwd.y;
			const fz = _fwd.z;

			for (let m = 0; m < flock.birdCount; m++) {
				const bird = flock.members[m];
				activeBirdTotal++;

				const sway = Math.sin(simTime * 1.35 + bird.altPhase) * 0.11;
				const vertBob = Math.cos(simTime * 1.1 + bird.altPhase) * 0.09;

				const px = _vel.x + fx * bird.localForward + _right.x * (bird.localRight + sway) + _up.x * (bird.localUp + vertBob);
				const py = _vel.y + fy * bird.localForward + _right.y * (bird.localRight + sway) + _up.y * (bird.localUp + vertBob);
				const pz = _vel.z + fz * bird.localForward + _right.z * (bird.localRight + sway) + _up.z * (bird.localUp + vertBob);

				const glideWave = Math.sin(simTime * 0.42 + bird.glideOffset);
				const flapEnv =
					flock.path.behavior === 'small-orbit'
						? smoothstep(-0.15, 0.55, glideWave) * 0.85 + 0.12
						: smoothstep(-0.65, 0.15, glideWave) * 0.88 + 0.12;
				const phase = simTime * bird.wingFreq + bird.wingPhase;
				const innerAngle = Math.sin(phase) * 0.58 * flapEnv + 0.09;
				const outerAngle = innerAngle + Math.sin(phase - 0.65) * 0.44 * flapEnv;

				const s = bird.scale * 0.34 * edgeFade;
				const bodyLenF = 0.68 * s;
				const bodyLenB = -0.78 * s;
				const shoulderW = 0.12 * s;
				const innerSpan = 0.56 * s;
				const outerSpan = 0.54 * s;

				const ci = Math.cos(innerAngle);
				const si = Math.sin(innerAngle);
				const co = Math.cos(outerAngle);
				const so = Math.sin(outerAngle);

				const elbowR = shoulderW + ci * innerSpan;
				const elbowU = si * innerSpan;
				const elbowF = 0.08 * s;

				const tipR = elbowR + co * outerSpan;
				const tipU = elbowU + so * outerSpan;
				const tipF = -0.26 * s;

				let ptr = birdWriteIdx * VERTS_PER_BIRD * 3;
				birdWriteIdx++;

				const writeV = (lr: number, lu: number, lf: number) => {
					birdPositions[ptr++] = px + rx * lr + ux * lu + fx * lf;
					birdPositions[ptr++] = py + ry * lr + uy * lu + fy * lf;
					birdPositions[ptr++] = pz + rz * lr + uz * lu + fz * lf;
				};

				// 1-2: Slender body / tail diamond
				writeV(0, 0.02 * s, bodyLenF);
				writeV(shoulderW, 0, -0.14 * s);
				writeV(0, 0, bodyLenB);

				writeV(0, 0.02 * s, bodyLenF);
				writeV(0, 0, bodyLenB);
				writeV(-shoulderW, 0, -0.14 * s);

				// 3: Right inner wing
				writeV(shoulderW * 0.7, 0.01 * s, 0.24 * s);
				writeV(elbowR, elbowU, elbowF);
				writeV(shoulderW * 0.8, 0, -0.24 * s);

				// 4: Left inner wing
				writeV(-shoulderW * 0.7, 0.01 * s, 0.24 * s);
				writeV(-shoulderW * 0.8, 0, -0.24 * s);
				writeV(-elbowR, elbowU, elbowF);

				// 5: Right outer primary wingtip
				writeV(elbowR, elbowU, elbowF);
				writeV(tipR, tipU, tipF);
				writeV(shoulderW * 0.8, 0, -0.24 * s);

				// 6: Left outer primary wingtip
				writeV(-elbowR, elbowU, elbowF);
				writeV(-shoulderW * 0.8, 0, -0.24 * s);
				writeV(-tipR, tipU, tipF);
			}
		}

		// Zero out remaining unused bird slots so inactive flocks disappear cleanly
		if (birdWriteIdx < BIRD_POOL_TOTAL) {
			birdPositions.fill(0, birdWriteIdx * VERTS_PER_BIRD * 3);
		}
		birdPosAttr.needsUpdate = true;

		return {
			aircraftCount: activePlanes + activeHelis,
			birdCount: activeBirdTotal
		};
	}

	function dispose(): void {
		scene.remove(root);
		for (const geo of geometriesToDispose) {
			geo.dispose();
		}
		for (const mat of materialsToDispose) {
			mat.dispose();
		}
		geometriesToDispose.length = 0;
		materialsToDispose.length = 0;
	}

	return {
		update,
		dispose
	};
}
