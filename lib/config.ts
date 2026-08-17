/**
 * Load llama-swap extension configuration from ~/.pi/agent/pi-llama-swap.json.
 *
 * Supports one or more llama-swap instances:
 *   - Legacy shape (single): top-level origin/port/basePath/apiKey/contextOverrides/modelCapabilities.
 *   - Multi shape: `instances` array, each entry optionally setting `id` and
 *     `name` plus the connection fields above.
 */

import { readFile, writeFile } from "node:fs/promises";
import { homedir } from "node:os";
import { join } from "node:path";

import { defaultInstance, mergeConfig, parsePortArg, parseUrlArg } from "./url.js";
import type { LlamaSwapConfig, LlamaSwapInstance, ModelCapabilities } from "./types.js";

const CONFIG_FILENAME = "pi-llama-swap.json";

/** Provider id used for the single (legacy) instance. */
export const DEFAULT_INSTANCE_ID = "llama-swap";

/** Human-readable name for the single (legacy) instance. */
export const DEFAULT_INSTANCE_NAME = "Llama Swap";

/**
 * Path to the config file under `~/.pi/agent/`.
 * @returns Absolute path to pi-llama-swap.json.
 */
export function configPath(): string {
	return join(homedir(), ".pi", "agent", CONFIG_FILENAME);
}

/**
 * Reads and parses the config file if it exists.
 * @param path - File path to read.
 * @returns Parsed config or null when missing.
 */
async function readConfigFile(path: string): Promise<Record<string, unknown> | null> {
	try {
		const raw = await readFile(path, "utf8");
		return JSON.parse(raw) as Record<string, unknown>;
	} catch (err) {
		if ((err as NodeJS.ErrnoException).code === "ENOENT") {
			return null;
		}
		throw err;
	}
}

/**
 * Serializes read-modify-write cycles on the config file so concurrent saves
 * (e.g. parallel instance refreshes) cannot lose updates.
 */
let configWriteChain: Promise<void> = Promise.resolve();

function withConfigFileLock<T>(fn: () => Promise<T>): Promise<T> {
	const run = configWriteChain.then(fn);
	configWriteChain = run.then(
		() => undefined,
		() => undefined,
	);
	return run;
}

/**
 * Validates and normalizes one raw instance entry (connection fields only).
 * @param raw - Raw instance object from JSON.
 * @returns Normalized connection fragment.
 */
function normalizeConnection(raw: Record<string, unknown>): Partial<LlamaSwapInstance> {
	const out: Partial<LlamaSwapInstance> = {};
	if (typeof raw.origin === "string" && raw.origin.trim()) {
		out.origin = raw.origin.trim();
	}
	if (typeof raw.port === "number" && Number.isInteger(raw.port) && raw.port >= 1 && raw.port <= 65535) {
		out.port = raw.port;
	}
	if (typeof raw.basePath === "string" && raw.basePath.trim()) {
		out.basePath = raw.basePath.trim();
	}
	if (typeof raw.apiKey === "string" && raw.apiKey.length > 0) {
		out.apiKey = raw.apiKey;
	}
	const overrides = normalizeOverrides(raw.contextOverrides);
	if (overrides) {
		out.contextOverrides = overrides;
	}
	const capabilities = normalizeCapabilities(raw.modelCapabilities);
	if (capabilities) {
		out.modelCapabilities = capabilities;
	}
	return out;
}

/**
 * Validates and normalizes a context-overrides map.
 * @param value - Raw contextOverrides value.
 * @returns Normalized map or undefined when empty/invalid.
 */
function normalizeOverrides(value: unknown): Record<string, number> | undefined {
	if (!value || typeof value !== "object") {
		return undefined;
	}
	const overrides: Record<string, number> = {};
	for (const [model, size] of Object.entries(value)) {
		if (typeof size === "number" && Number.isInteger(size) && size > 0) {
			overrides[model] = size;
		}
	}
	return Object.keys(overrides).length > 0 ? overrides : undefined;
}

/**
 * Validates and normalizes a model-capabilities map.
 * @param value - Raw modelCapabilities value.
 * @returns Normalized map or undefined when empty/invalid.
 */
function normalizeCapabilities(value: unknown): Record<string, ModelCapabilities> | undefined {
	if (!value || typeof value !== "object" || Array.isArray(value)) {
		return undefined;
	}
	const out: Record<string, ModelCapabilities> = {};
	for (const [model, raw] of Object.entries(value)) {
		if (!raw || typeof raw !== "object" || Array.isArray(raw)) {
			continue;
		}
		const entry: ModelCapabilities = {};
		const record = raw as Record<string, unknown>;
		if (typeof record.reasoning === "boolean") {
			entry.reasoning = record.reasoning;
		}
		if (typeof record.imageInput === "boolean") {
			entry.imageInput = record.imageInput;
		}
		if (typeof record.contextWindow === "number" && Number.isInteger(record.contextWindow) && record.contextWindow > 0) {
			entry.contextWindow = record.contextWindow;
		}
		if (typeof record.maxTokens === "number" && Number.isInteger(record.maxTokens) && record.maxTokens > 0) {
			entry.maxTokens = record.maxTokens;
		}
		if (Object.keys(entry).length > 0) {
			out[model] = entry;
		}
	}
	return Object.keys(out).length > 0 ? out : undefined;
}

/**
 * Generates a unique provider id for an instance without an explicit one.
 * @param index - Instance position.
 * @param taken - Ids already used.
 * @returns Unused id (`llama-swap`, `llama-swap-2`, …).
 */
function generateId(index: number, taken: Set<string>): string {
	let candidate = index === 0 ? DEFAULT_INSTANCE_ID : `${DEFAULT_INSTANCE_ID}-${index + 1}`;
	let suffix = 2;
	while (taken.has(candidate)) {
		candidate = `${DEFAULT_INSTANCE_ID}-${index + 1}-${suffix++}`;
	}
	taken.add(candidate);
	return candidate;
}

/**
 * Builds the display name for an instance without an explicit one.
 * @param index - Instance position.
 * @returns `Llama Swap`, `Llama Swap 2`, …
 */
function generateName(index: number): string {
	return index === 0 ? DEFAULT_INSTANCE_NAME : `${DEFAULT_INSTANCE_NAME} ${index + 1}`;
}

/**
 * Applies environment variable overrides (highest precedence) to the first
 * instance, mirroring the legacy single-instance behavior.
 * @param config - Config after file merge.
 * @returns Config with env overrides applied.
 */
export function applyEnvOverrides(config: LlamaSwapConfig): LlamaSwapConfig {
	const [first, ...rest] = config.instances;
	if (!first) {
		return config;
	}

	let result = first;
	const urlEnv = process.env.LLAMA_SWAP_URL?.trim();
	if (urlEnv) {
		const parsed = parseUrlArg(urlEnv);
		result = mergeConfig(result, {
			origin: parsed.origin,
			...(parsed.port !== undefined ? { port: parsed.port } : {}),
			...(parsed.basePath !== undefined ? { basePath: parsed.basePath } : {}),
		});
	}

	const portEnv = process.env.LLAMA_SWAP_PORT?.trim();
	if (portEnv) {
		result = mergeConfig(result, { port: parsePortArg(portEnv) });
	}

	const keyEnv = process.env.LLAMA_SWAP_API_KEY?.trim();
	if (keyEnv) {
		result = mergeConfig(result, { apiKey: keyEnv });
	}

	return { instances: [result, ...rest] };
}

/**
 * Loads config: defaults, optional ~/.pi/agent/pi-llama-swap.json, then env overrides.
 * @returns Effective connection settings (one or more instances).
 */
export async function loadConfig(): Promise<LlamaSwapConfig> {
	const file = await readConfigFile(configPath());

	if (file && Array.isArray(file.instances) && file.instances.length > 0) {
		const taken = new Set<string>();
		const instances: LlamaSwapInstance[] = file.instances.map((entry, index) => {
			const raw = entry && typeof entry === "object" ? (entry as Record<string, unknown>) : {};
			const id =
				typeof raw.id === "string" && raw.id.trim()
					? raw.id.trim()
					: generateId(index, taken);
			taken.add(id);
			const name = typeof raw.name === "string" && raw.name.trim() ? raw.name.trim() : generateName(index);
			return mergeConfig(defaultInstance(), { id, name, ...normalizeConnection(raw) });
		});
		return applyEnvOverrides({ instances });
	}

	// Legacy single-instance shape: top-level connection fields.
	const raw = file ?? {};
	const instance: LlamaSwapInstance = mergeConfig(defaultInstance(), {
		id: DEFAULT_INSTANCE_ID,
		name: DEFAULT_INSTANCE_NAME,
		...normalizeConnection(raw),
	});
	return applyEnvOverrides({ instances: [instance] });
}

/**
 * Updates the context-overrides map for a model in the config file.
 * @param instanceId - Provider id of the instance to update.
 * @param model - Model id to set or clear.
 * @param ctxSize - Context size in tokens, or undefined to remove the override.
 */
export async function saveContextOverride(instanceId: string, model: string, ctxSize: number | undefined): Promise<void> {
	await withConfigFileLock(() => saveContextOverrideLocked(instanceId, model, ctxSize));
}

/**
 * Updates the context-overrides map for a model in the config file.
 * Unlocked variant: callers must hold the config file lock.
 * @param instanceId - Provider id of the instance to update.
 * @param model - Model id to set or clear.
 * @param ctxSize - Context size in tokens, or undefined to remove the override.
 */
async function saveContextOverrideLocked(instanceId: string, model: string, ctxSize: number | undefined): Promise<void> {
	const path = configPath();
	let current: Record<string, unknown> = {};
	try {
		const raw = await readFile(path, "utf8");
		current = JSON.parse(raw) as Record<string, unknown>;
	} catch (err) {
		if ((err as NodeJS.ErrnoException).code !== "ENOENT") throw err;
	}

	if (Array.isArray(current.instances)) {
		const list = current.instances as Record<string, unknown>[];
		let entry = list.find((item) => item && item.id === instanceId);
		if (!entry) {
			entry = { id: instanceId };
			list.push(entry);
		}
		const overrides = normalizeOverrides(entry.contextOverrides) ?? {};
		if (ctxSize === undefined) {
			delete overrides[model];
		} else {
			overrides[model] = ctxSize;
		}
		entry.contextOverrides = overrides;
		current.instances = list;
	} else {
		// Legacy single-instance shape.
		const overrides = normalizeOverrides(current.contextOverrides) ?? {};
		if (ctxSize === undefined) {
			delete overrides[model];
		} else {
			overrides[model] = ctxSize;
		}
		current.contextOverrides = overrides;
	}

	await writeFile(path, JSON.stringify(current, null, 2), "utf8");
}

/**
 * Merges discovered model capabilities into the config file for an instance.
 * Per-field merge: previously cached fields are kept when absent from `caps`.
 * @param instanceId - Provider id of the instance to update.
 * @param caps - Discovered capabilities per model id.
 */
export async function saveModelCapabilities(instanceId: string, caps: Record<string, ModelCapabilities>): Promise<void> {
	if (Object.keys(caps).length === 0) {
		return;
	}
	await withConfigFileLock(() => saveModelCapabilitiesLocked(instanceId, caps));
}

/**
 * Writes merged capabilities to the config file.
 * Unlocked variant: callers must hold the config file lock.
 * @param instanceId - Provider id of the instance to update.
 * @param caps - Discovered capabilities per model id.
 */
async function saveModelCapabilitiesLocked(instanceId: string, caps: Record<string, ModelCapabilities>): Promise<void> {
	const path = configPath();
	let current: Record<string, unknown> = {};
	try {
		const raw = await readFile(path, "utf8");
		current = JSON.parse(raw) as Record<string, unknown>;
	} catch (err) {
		if ((err as NodeJS.ErrnoException).code !== "ENOENT") throw err;
	}

	const mergeInto = (target: Record<string, unknown>): void => {
		const existing = normalizeCapabilities(target.modelCapabilities) ?? {};
		for (const [model, discovered] of Object.entries(caps)) {
			existing[model] = { ...existing[model], ...discovered };
		}
		target.modelCapabilities = existing;
	};

	if (Array.isArray(current.instances)) {
		const list = current.instances as Record<string, unknown>[];
		let entry = list.find((item) => item && item.id === instanceId);
		if (!entry) {
			entry = { id: instanceId };
			list.push(entry);
		}
		mergeInto(entry);
		current.instances = list;
	} else {
		// Legacy single-instance shape.
		mergeInto(current);
	}

	await writeFile(path, JSON.stringify(current, null, 2), "utf8");
}
