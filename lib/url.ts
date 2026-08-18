/**
 * URL parsing and base URL construction for llama-swap endpoints.
 */

import type { LlamaSwapInstance } from "./types.js";

const DEFAULT_ORIGIN = "http://127.0.0.1";
const DEFAULT_PORT = 8080;
const DEFAULT_BASE_PATH = "/v1";

/**
 * Returns default llama-swap connection settings.
 * @returns Fresh default instance (id/name filled by caller).
 */
export function defaultInstance(): LlamaSwapInstance {
	return { id: "", name: "", origin: DEFAULT_ORIGIN, port: DEFAULT_PORT, basePath: DEFAULT_BASE_PATH };
}

/**
 * Normalizes a URL pathname into an OpenAI API base path ending in `/v1`.
 * @param pathname - Path from a parsed URL.
 * @returns Normalized base path (leading slash, no trailing slash).
 */
export function normalizeBasePath(pathname: string): string {
	if (!pathname || pathname === "/") {
		return DEFAULT_BASE_PATH;
	}
	const trimmed = pathname.replace(/\/$/, "");
	if (trimmed.endsWith("/v1")) {
		return trimmed;
	}
	return `${trimmed}/v1`;
}

/**
 * Builds llama-swap server origin (`{scheme}://{host}:{port}`) without API path.
 * @param config - Instance connection settings.
 * @returns Root server URL for endpoints like `/running`.
 */
export function buildServerOrigin(config: LlamaSwapInstance): string {
	const origin = config.origin.includes("://") ? config.origin : `http://${config.origin}`;
	const url = new URL(origin);
	url.port = String(config.port);
	url.pathname = "";
	url.search = "";
	url.hash = "";
	return url.origin;
}

/**
 * Builds the OpenAI API base URL (`{origin}:{port}{basePath}`) from config.
 * @param config - Instance connection settings with origin, port, and optional basePath.
 * @returns Normalized base URL without trailing slash.
 */
export function buildBaseUrl(config: LlamaSwapInstance): string {
	const origin = config.origin.includes("://") ? config.origin : `http://${config.origin}`;
	const url = new URL(origin);
	url.port = String(config.port);
	const basePath = normalizeBasePath(config.basePath ?? DEFAULT_BASE_PATH);
	url.pathname = basePath;
	url.search = "";
	url.hash = "";
	return url.toString().replace(/\/$/, "");
}

/**
 * Parses a URL argument into origin, optional port, and API base path.
 * @param arg - User input (scheme+host, host only, or URL with optional path).
 * @returns Parsed connection fields.
 * @throws {Error} When the argument is empty or not a valid URL.
 */
export function parseUrlArg(arg: string): { origin: string; port?: number; basePath?: string } {
	const trimmed = arg.trim();
	if (!trimmed) {
		throw new Error("URL is required. Example: http://127.0.0.1");
	}

	let raw = trimmed;
	if (!/^https?:\/\//i.test(raw)) {
		raw = `http://${raw}`;
	}

	let url: URL;
	try {
		url = new URL(raw);
	} catch {
		throw new Error(`Invalid URL: ${trimmed}`);
	}

	if (!url.hostname) {
		throw new Error(`Invalid URL: ${trimmed}`);
	}

	const origin = `${url.protocol}//${url.hostname}`;
	const port = url.port ? Number(url.port) : undefined;
	if (port !== undefined && (port < 1 || port > 65535)) {
		throw new Error(`Port out of range: ${port}`);
	}

	const basePath = normalizeBasePath(url.pathname);

	return { origin, port, basePath };
}

/**
 * Parses a port argument into a valid port number.
 * @param arg - Port string (1–65535).
 * @returns Valid port number.
 * @throws {Error} When the port is missing or invalid.
 */
export function parsePortArg(arg: string): number {
	const trimmed = arg.trim();
	if (!trimmed) {
		throw new Error("Port is required. Example: 8080");
	}

	const port = Number(trimmed);
	if (!Number.isInteger(port) || port < 1 || port > 65535) {
		throw new Error(`Invalid port: ${trimmed}. Use an integer from 1 to 65535.`);
	}

	return port;
}

/**
 * Merges a partial config update into an existing instance.
 * @param current - Current instance settings.
 * @param partial - Fields to update.
 * @returns New instance object.
 */
export function mergeConfig(current: LlamaSwapInstance, partial: Partial<LlamaSwapInstance>): LlamaSwapInstance {
	return {
		id: partial.id ?? current.id,
		name: partial.name ?? current.name,
		origin: partial.origin ?? current.origin,
		port: partial.port ?? current.port,
		basePath: partial.basePath ?? current.basePath,
		apiKey: partial.apiKey !== undefined ? partial.apiKey : current.apiKey,
		contextOverrides: partial.contextOverrides ?? current.contextOverrides,
		modelCapabilities: partial.modelCapabilities ?? current.modelCapabilities,
	};
}
