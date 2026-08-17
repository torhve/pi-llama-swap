/**
 * Shared types for the pi llama-swap extension.
 */

/** Persisted connection settings for one llama-swap endpoint. */
export interface LlamaSwapInstance {
	/** Provider id registered in pi (unique, e.g. `llama-swap`, `llama-swap-2`). */
	id: string;
	/** Human-readable provider name shown in pi (e.g. `/model` list). */
	name: string;
	/** Origin without path, e.g. `http://127.0.0.1`. */
	origin: string;
	/** TCP port (1–65535). */
	port: number;
	/** OpenAI API path prefix (default `/v1`). */
	basePath?: string;
	/** Optional API key sent as Bearer when set. */
	apiKey?: string;
	/** Per-model context window overrides (model id -> tokens). */
	contextOverrides?: Record<string, number>;
	/** Cached per-model capabilities discovered from llama-swap (model id -> caps). */
	modelCapabilities?: Record<string, ModelCapabilities>;
}

/** Extension config: one or more llama-swap instances. */
export interface LlamaSwapConfig {
	instances: LlamaSwapInstance[];
}

/**
 * Discovered capabilities for one model, cached in the config file between runs.
 * All fields optional: only values actually discovered are stored.
 */
export interface ModelCapabilities {
	/** Chat template supports reasoning (thinking). */
	reasoning?: boolean;
	/** Model accepts image input. */
	imageInput?: boolean;
	/** Discovered context window in tokens. */
	contextWindow?: number;
	/** Discovered max output tokens. */
	maxTokens?: number;
}

/** OpenAI-compatible model entry from GET /v1/models. */
export interface OpenAIModelEntry {
	id: string;
	name?: string;
	[key: string]: unknown;
}

/** OpenAI-compatible models list response. */
export interface OpenAIModelsListResponse {
	object?: string;
	data: OpenAIModelEntry[];
}

/** Result of a provider refresh attempt. */
export interface RefreshResult {
	baseUrl: string;
	modelCount: number;
	error?: string;
}
