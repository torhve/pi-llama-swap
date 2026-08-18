import { afterEach, describe, expect, it, vi } from "vitest";

import {
	buildModelLimits,
	DEFAULT_CONTEXT_WINDOW,
	extractContextFromModelEntry,
	extractMaxTokensFromModelEntry,
	parseContextFromCmd,
	resolveContextWindow,
	resolveMaxTokens,
} from "../lib/context.js";
import { createEntry, createInstance, jsonResponse, mockFetch } from "./mocks.js";
import type { LlamaSwapInstance, OpenAIModelEntry } from "../lib/types.js";

afterEach(() => {
	vi.unstubAllGlobals();
});

describe("DEFAULT_CONTEXT_WINDOW", () => {
	it("is 256K (262144)", () => {
		expect(DEFAULT_CONTEXT_WINDOW).toBe(262_144);
	});
});

describe("extractContextFromModelEntry", () => {
	it("prefers top-level context_length over the other top-level fields", () => {
		const entry = createEntry({ context_length: 4096, max_context_length: 8192, context_window: 16384 });
		expect(extractContextFromModelEntry(entry)).toBe(4096);
	});

	it("falls back to max_context_length, then context_window", () => {
		expect(extractContextFromModelEntry(createEntry({ max_context_length: 8192, context_window: 16384 }))).toBe(8192);
		expect(extractContextFromModelEntry(createEntry({ context_window: 16384 }))).toBe(16384);
	});

	it("reads meta.llamaswap in the order context_length > context > max_context > max_context_length", () => {
		expect(extractContextFromModelEntry(createEntry({ meta: { llamaswap: { context_length: 4096 } } }))).toBe(4096);
		expect(extractContextFromModelEntry(createEntry({ meta: { llamaswap: { context: 8192 } } }))).toBe(8192);
		expect(extractContextFromModelEntry(createEntry({ meta: { llamaswap: { max_context: 16384 } } }))).toBe(16384);
		expect(extractContextFromModelEntry(createEntry({ meta: { llamaswap: { max_context_length: 32768 } } }))).toBe(32768);
		expect(
			extractContextFromModelEntry(
				createEntry({ meta: { llamaswap: { context: 8192, max_context: 16384, max_context_length: 32768 } } }),
			),
		).toBe(8192);
	});

	it("prefers meta.n_ctx after meta.llamaswap", () => {
		expect(extractContextFromModelEntry(createEntry({ meta: { n_ctx: 4096 } }))).toBe(4096);
		expect(extractContextFromModelEntry(createEntry({ meta: { llamaswap: { context_length: 4096 }, n_ctx: 8192 } }))).toBe(
			4096,
		);
	});

	it("reads metadata.context_length and metadata.context last", () => {
		expect(extractContextFromModelEntry(createEntry({ metadata: { context_length: 4096 } }))).toBe(4096);
		expect(extractContextFromModelEntry(createEntry({ metadata: { context: 8192 } }))).toBe(8192);
		// meta beats metadata
		expect(extractContextFromModelEntry(createEntry({ meta: { n_ctx: 4096 }, metadata: { context_length: 8192 } }))).toBe(
			4096,
		);
	});

	it("prefers a top-level field over any meta/metadata value", () => {
		expect(extractContextFromModelEntry(createEntry({ context_length: 4096, meta: { n_ctx: 8192 } }))).toBe(4096);
	});

	it("returns undefined when nothing is present", () => {
		expect(extractContextFromModelEntry(createEntry({ id: "m" }))).toBeUndefined();
		expect(extractContextFromModelEntry(createEntry({ meta: {}, metadata: {} }))).toBeUndefined();
	});

	it("coerces numeric strings", () => {
		expect(extractContextFromModelEntry(createEntry({ context_length: "4096" }))).toBe(4096);
		expect(extractContextFromModelEntry(createEntry({ meta: { llamaswap: { context: "8192" } } }))).toBe(8192);
	});

	it("rejects zero, negative, float, and non-numeric values", () => {
		expect(extractContextFromModelEntry(createEntry({ context_length: 0 }))).toBeUndefined();
		expect(extractContextFromModelEntry(createEntry({ context_length: -5 }))).toBeUndefined();
		expect(extractContextFromModelEntry(createEntry({ context_length: 4096.5 }))).toBeUndefined();
		expect(extractContextFromModelEntry(createEntry({ context_length: "0" }))).toBeUndefined();
		expect(extractContextFromModelEntry(createEntry({ context_length: "abc" }))).toBeUndefined();
	});
});

describe("extractMaxTokensFromModelEntry", () => {
	it("prefers output_length over max_tokens", () => {
		expect(extractMaxTokensFromModelEntry(createEntry({ output_length: 1024, max_tokens: 2048 }))).toBe(1024);
		expect(extractMaxTokensFromModelEntry(createEntry({ max_tokens: 2048 }))).toBe(2048);
	});

	it("falls back to meta.llamaswap (output_length then max_tokens)", () => {
		expect(extractMaxTokensFromModelEntry(createEntry({ meta: { llamaswap: { output_length: 1024 } } }))).toBe(1024);
		expect(extractMaxTokensFromModelEntry(createEntry({ meta: { llamaswap: { max_tokens: 2048 } } }))).toBe(2048);
	});

	it("prefers top-level over meta.llamaswap", () => {
		expect(extractMaxTokensFromModelEntry(createEntry({ max_tokens: 2048, meta: { llamaswap: { output_length: 1024 } } }))).toBe(
			2048,
		);
	});

	it("returns undefined when nothing is present or valid", () => {
		expect(extractMaxTokensFromModelEntry(createEntry({ id: "m" }))).toBeUndefined();
		expect(extractMaxTokensFromModelEntry(createEntry({ max_tokens: 0 }))).toBeUndefined();
	});
});

describe("parseContextFromCmd", () => {
	it("parses -c with a space", () => {
		expect(parseContextFromCmd("llama-server -c 4096")).toBe(4096);
	});

	it("parses -c with an equals sign", () => {
		expect(parseContextFromCmd("llama-server -c=4096")).toBe(4096);
	});

	it("parses --ctx-size with a space", () => {
		expect(parseContextFromCmd("llama-server --ctx-size 8192")).toBe(8192);
	});

	it("parses --ctx-size with an equals sign", () => {
		expect(parseContextFromCmd("llama-server --ctx-size=8192")).toBe(8192);
	});

	it("prefers --ctx-size over -c when both are present", () => {
		expect(parseContextFromCmd("llama-server -c 4096 --ctx-size 8192")).toBe(8192);
	});

	it("does not let -c match inside --ctx-size", () => {
		// Only --ctx-size present; the -c pattern must not misfire on it.
		expect(parseContextFromCmd("llama-server --ctx-size 8192")).toBe(8192);
	});

	it("parses from a multi-line command", () => {
		expect(parseContextFromCmd("line one\nllama-server -c 4096\nline three")).toBe(4096);
	});

	it("returns undefined when no context flag is present", () => {
		expect(parseContextFromCmd("llama-server -m model.gguf")).toBeUndefined();
		expect(parseContextFromCmd("")).toBeUndefined();
	});
});

describe("resolveContextWindow", () => {
	it("returns the mapped value when present", () => {
		const map = new Map([["model-1", 4096]]);
		expect(resolveContextWindow("model-1", map)).toBe(4096);
	});

	it("falls back to the default when absent", () => {
		expect(resolveContextWindow("missing", new Map())).toBe(262_144);
	});
});

describe("resolveMaxTokens", () => {
	it("returns the mapped value when present", () => {
		const map = new Map([["model-1", 1024]]);
		expect(resolveMaxTokens("model-1", map, 4096)).toBe(1024);
	});

	it("defaults to floor(contextWindow / 2) when absent", () => {
		expect(resolveMaxTokens("missing", new Map(), 100)).toBe(50);
		expect(resolveMaxTokens("missing", new Map(), 101)).toBe(50);
		expect(resolveMaxTokens("missing", new Map(), 262_144)).toBe(131_072);
	});
});

// --- buildModelLimits (fetch mocked) ---------------------------------------

/**
 * Builds a fetch handler that serves `running` for the /running URL (or throws
 * when it is the string "unreachable") and routes /props URLs through `onProps`.
 */
function withRunning(running: Response | "unreachable", onProps: (url: string) => Response) {
	return (url: string | URL | Request): Response => {
		const u = typeof url === "string" ? url : url.toString();
		if (u.endsWith("/running")) {
			if (running === "unreachable") throw new Error("fetch failed");
			return running;
		}
		if (u.includes("/props")) {
			return onProps(u);
		}
		return jsonResponse({});
	};
}

/**
 * Routes a /props URL to a props object: origin props carry `?model=<id>`; a
 * proxy props URL (no query) returns `proxyProps`.
 */
function routeProps(originByModel: Record<string, unknown>, proxyProps?: unknown) {
	return (url: string): Response => {
		const m = url.match(/model=([^&]*)/);
		if (m) {
			const id = decodeURIComponent(m[1]);
			return jsonResponse(originByModel[id] ?? {});
		}
		return jsonResponse(proxyProps ?? {});
	};
}

describe("buildModelLimits", () => {
	it("returns only /v1/models-derived values when /running is unreachable", async () => {
		mockFetch(withRunning("unreachable", () => jsonResponse({})));
		const entries = [createEntry({ id: "model-1", context_length: 4096 })];
		const result = await buildModelLimits(entries, createInstance());

		expect(result.contextByModel.get("model-1")).toBe(4096);
		expect(result.detectedByModel.get("model-1")).toEqual({ contextWindow: 4096 });
		expect(result.runningStateByModel.size).toBe(0);
		expect(result.reasoningByModel.size).toBe(0);
		expect(result.imageInputByModel.size).toBe(0);
	});

	it("reads n_ctx from the proxy /props and defaults the state to running", async () => {
		const handler = withRunning(
			jsonResponse({ running: [{ model: "model-1", proxy: "http://127.0.0.1:9000" }] }),
			routeProps({}, { default_generation_settings: { n_ctx: 8192 } }),
		);
		const spy = mockFetch(handler);
		const result = await buildModelLimits([createEntry({ id: "model-1" })], createInstance());

		expect(result.contextByModel.get("model-1")).toBe(8192);
		expect(result.runningStateByModel.get("model-1")).toBe("running");
		expect(result.detectedByModel.get("model-1")).toEqual({
			contextWindow: 8192,
			reasoning: false,
			imageInput: false,
		});
		// The proxy props URL was hit; the origin fallback was not.
		const urls = spy.mock.calls.map((c) => String(c[0]));
		expect(urls).toContain("http://127.0.0.1:9000/props");
		expect(urls.some((u) => u.includes("127.0.0.1:8080/props"))).toBe(false);
	});

	it("records the explicit running state when present", async () => {
		mockFetch(
			withRunning(
				jsonResponse({ running: [{ model: "model-1", state: "starting" }] }),
				() => jsonResponse({}),
			),
		);
		const result = await buildModelLimits([createEntry({ id: "model-1" })], createInstance());
		expect(result.runningStateByModel.get("model-1")).toBe("starting");
	});

	it("falls back to the origin /props?model=<id> (URL-encoded) when there is no proxy", async () => {
		const spy = mockFetch(
			withRunning(
				jsonResponse({ running: [{ model: "model/1" }] }),
				routeProps({ "model/1": { default_generation_settings: { n_ctx: 4096 } } }),
			),
		);
		const result = await buildModelLimits([createEntry({ id: "model/1" })], createInstance());

		expect(result.contextByModel.get("model/1")).toBe(4096);
		const urls = spy.mock.calls.map((c) => String(c[0]));
		expect(urls).toContain("http://127.0.0.1:8080/props?model=model%2F1");
	});

	it("falls back to the origin /props when the proxy returns 500", async () => {
		const handler = (url: string | URL | Request): Response => {
			const u = typeof url === "string" ? url : url.toString();
			if (u.endsWith("/running")) {
				return jsonResponse({ running: [{ model: "model-1", proxy: "http://127.0.0.1:9000" }] });
			}
			if (u === "http://127.0.0.1:9000/props") {
				return new Response("proxy error", { status: 500 });
			}
			if (u.includes("/props")) {
				return jsonResponse({ default_generation_settings: { n_ctx: 4096 } });
			}
			return jsonResponse({});
		};
		const spy = mockFetch(handler);
		const result = await buildModelLimits([createEntry({ id: "model-1" })], createInstance());

		expect(result.contextByModel.get("model-1")).toBe(4096);
		const urls = spy.mock.calls.map((c) => String(c[0]));
		expect(urls).toContain("http://127.0.0.1:9000/props");
		expect(urls).toContain("http://127.0.0.1:8080/props?model=model-1");
	});

	it("sets reasoning when chat_template_caps.supports_preserve_reasoning is true", async () => {
		mockFetch(
			withRunning(
				jsonResponse({ running: [{ model: "model-1" }] }),
				routeProps({ "model-1": { chat_template_caps: { supports_preserve_reasoning: true } } }),
			),
		);
		const result = await buildModelLimits([createEntry({ id: "model-1" })], createInstance());

		expect(result.detectedByModel.get("model-1")?.reasoning).toBe(true);
		expect(result.reasoningByModel.get("model-1")).toBe(true);
	});

	it("detects image input from the various /props shapes", async () => {
		const positive = [
			{ vision: true },
			{ supports_vision: true },
			{ supports_images: true },
			{ capabilities: ["vision"] },
			{ modalities: ["text", "image"] },
			{ input: { image: true } },
		];
		for (const props of positive) {
			mockFetch(
				withRunning(jsonResponse({ running: [{ model: "model-1" }] }), routeProps({ "model-1": props })),
			);
			const result = await buildModelLimits([createEntry({ id: "model-1" })], createInstance());
			expect(result.imageInputByModel.get("model-1"), JSON.stringify(props)).toBe(true);
			expect(result.detectedByModel.get("model-1")?.imageInput, JSON.stringify(props)).toBe(true);
			vi.unstubAllGlobals();
		}
	});

	it("does not detect image input from negative /props shapes", async () => {
		const negative = [{ capabilities: ["text"] }, { input: { image: false } }, {}];
		for (const props of negative) {
			mockFetch(
				withRunning(jsonResponse({ running: [{ model: "model-1" }] }), routeProps({ "model-1": props })),
			);
			const result = await buildModelLimits([createEntry({ id: "model-1" })], createInstance());
			expect(result.imageInputByModel.has("model-1"), JSON.stringify(props)).toBe(false);
			expect(result.detectedByModel.get("model-1")?.imageInput, JSON.stringify(props)).toBe(false);
			vi.unstubAllGlobals();
		}
	});

	it("keeps the user override in the map but still records the live value as detected", async () => {
		mockFetch(
			withRunning(
				jsonResponse({ running: [{ model: "model-1" }] }),
				routeProps({ "model-1": { default_generation_settings: { n_ctx: 8192 } } }),
			),
		);
		const result = await buildModelLimits([createEntry({ id: "model-1" })], createInstance(), {
			"model-1": 12345,
		});

		// The map keeps the user's override; the live 8192 is ignored for the map.
		expect(result.contextByModel.get("model-1")).toBe(12345);
		// But the live value is still what was detected this pass.
		expect(result.detectedByModel.get("model-1")?.contextWindow).toBe(8192);
	});

	it("fills context/maxTokens gaps from the cache for non-running models", async () => {
		const config = createInstance({
			modelCapabilities: { "model-1": { contextWindow: 4096, maxTokens: 1024 } },
		});
		mockFetch(withRunning(jsonResponse({ running: [] }), () => jsonResponse({})));
		const result = await buildModelLimits([], config);

		expect(result.contextByModel.get("model-1")).toBe(4096);
		expect(result.maxTokensByModel.get("model-1")).toBe(1024);
		// Cache is not a live discovery.
		expect(result.detectedByModel.size).toBe(0);
	});

	it("applies cached reasoning/imageInput to non-running models", async () => {
		const config = createInstance({
			modelCapabilities: { "model-1": { reasoning: true, imageInput: true } },
		});
		mockFetch(withRunning(jsonResponse({ running: [] }), () => jsonResponse({})));
		const result = await buildModelLimits([], config);

		expect(result.reasoningByModel.get("model-1")).toBe(true);
		expect(result.imageInputByModel.get("model-1")).toBe(true);
	});

	it("does not let the cache override live-discovered values", async () => {
		const config = createInstance({
			modelCapabilities: { "model-1": { contextWindow: 99999 } },
		});
		mockFetch(withRunning(jsonResponse({ running: [] }), () => jsonResponse({})));
		const entries = [createEntry({ id: "model-1", context_length: 4096 })];
		const result = await buildModelLimits(entries, config);

		expect(result.contextByModel.get("model-1")).toBe(4096);
	});

	it("lets user overrides beat every discovered value in the final map", async () => {
		const config = createInstance({
			modelCapabilities: { "model-1": { contextWindow: 99999 } },
		});
		mockFetch(
			withRunning(
				jsonResponse({ running: [{ model: "model-1" }] }),
				routeProps({ "model-1": { default_generation_settings: { n_ctx: 8192 } } }),
			),
		);
		const entries = [createEntry({ id: "model-1", context_length: 4096 })];
		const result = await buildModelLimits(entries, config, { "model-1": 11111 });

		expect(result.contextByModel.get("model-1")).toBe(11111);
	});

	it("only records values actually discovered this pass in detectedByModel", async () => {
		const config = createInstance({
			modelCapabilities: { "cached-model": { contextWindow: 4096, reasoning: true } },
		});
		mockFetch(
			withRunning(
				jsonResponse({ running: [{ model: "live-model" }] }),
				routeProps({ "live-model": { default_generation_settings: { n_ctx: 2048 } } }),
			),
		);
		const result = await buildModelLimits([], config);

		// Only the live model is in detectedByModel; the cached model is not.
		expect(result.detectedByModel.has("live-model")).toBe(true);
		expect(result.detectedByModel.has("cached-model")).toBe(false);
		expect(result.detectedByModel.get("live-model")).toEqual({
			contextWindow: 2048,
			reasoning: false,
			imageInput: false,
		});
	});
});
