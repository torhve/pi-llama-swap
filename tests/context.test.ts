import { afterEach, describe, expect, it, vi } from "vitest";

import {
	buildModelLimits,
	DEFAULT_CONTEXT_WINDOW,
	detectAnswerNow,
	detectEngine,
	detectReasoning,
	extractContextFromModelEntry,
	extractMaxTokensFromModelEntry,
	parseContextFromCmd,
	resolveContextWindow,
	resolveMaxTokens,
	supportsImageInputEntry,
} from "../lib/context.js";
import { createEntry, createInstance, jsonResponse, mockFetch } from "./mocks.js";

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
		expect(extractContextFromModelEntry(createEntry({ max_context_length: 8192, context_window: 16384 }))).toBe(
			8192,
		);
		expect(extractContextFromModelEntry(createEntry({ context_window: 16384 }))).toBe(16384);
	});

	it("reads meta.llamaswap in the order context_length > context > max_context > max_context_length", () => {
		expect(extractContextFromModelEntry(createEntry({ meta: { llamaswap: { context_length: 4096 } } }))).toBe(4096);
		expect(extractContextFromModelEntry(createEntry({ meta: { llamaswap: { context: 8192 } } }))).toBe(8192);
		expect(extractContextFromModelEntry(createEntry({ meta: { llamaswap: { max_context: 16384 } } }))).toBe(16384);
		expect(extractContextFromModelEntry(createEntry({ meta: { llamaswap: { max_context_length: 32768 } } }))).toBe(
			32768,
		);
		expect(
			extractContextFromModelEntry(
				createEntry({ meta: { llamaswap: { context: 8192, max_context: 16384, max_context_length: 32768 } } }),
			),
		).toBe(8192);
	});

	it("prefers meta.n_ctx after meta.llamaswap", () => {
		expect(extractContextFromModelEntry(createEntry({ meta: { n_ctx: 4096 } }))).toBe(4096);
		expect(
			extractContextFromModelEntry(createEntry({ meta: { llamaswap: { context_length: 4096 }, n_ctx: 8192 } })),
		).toBe(4096);
	});

	it("reads metadata.context_length and metadata.context last", () => {
		expect(extractContextFromModelEntry(createEntry({ metadata: { context_length: 4096 } }))).toBe(4096);
		expect(extractContextFromModelEntry(createEntry({ metadata: { context: 8192 } }))).toBe(8192);
		// meta beats metadata
		expect(
			extractContextFromModelEntry(createEntry({ meta: { n_ctx: 4096 }, metadata: { context_length: 8192 } })),
		).toBe(4096);
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
		expect(extractMaxTokensFromModelEntry(createEntry({ meta: { llamaswap: { output_length: 1024 } } }))).toBe(
			1024,
		);
		expect(extractMaxTokensFromModelEntry(createEntry({ meta: { llamaswap: { max_tokens: 2048 } } }))).toBe(2048);
	});

	it("prefers top-level over meta.llamaswap", () => {
		expect(
			extractMaxTokensFromModelEntry(
				createEntry({ max_tokens: 2048, meta: { llamaswap: { output_length: 1024 } } }),
			),
		).toBe(2048);
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

// --- supportsImageInputEntry (GET /v1/models entry flags) -----------------

describe("supportsImageInputEntry", () => {
	it("detects the shapes llama-swap publishes for a vision model", () => {
		expect(supportsImageInputEntry(createEntry({ capabilities: { vision: true, function_calling: true } }))).toBe(
			true,
		);
		expect(supportsImageInputEntry(createEntry({ architecture: { input_modalities: ["text", "image"] } }))).toBe(
			true,
		);
		expect(supportsImageInputEntry(createEntry({ capabilities: ["vision"] }))).toBe(true);
		expect(supportsImageInputEntry(createEntry({ modalities: { vision: true } }))).toBe(true);
		expect(supportsImageInputEntry(createEntry({ input_modalities: ["text", "image"] }))).toBe(true);
		expect(supportsImageInputEntry(createEntry({ vision: true }))).toBe(true);
	});

	it("returns false for text-only and empty entries", () => {
		expect(
			supportsImageInputEntry(
				createEntry({ capabilities: { function_calling: true }, architecture: { input_modalities: ["text"] } }),
			),
		).toBe(false);
		expect(supportsImageInputEntry(createEntry({ capabilities: { vision: false } }))).toBe(false);
		expect(supportsImageInputEntry(createEntry({ vision: false }))).toBe(false);
		expect(supportsImageInputEntry(createEntry({ architecture: {} }))).toBe(false);
		expect(supportsImageInputEntry(createEntry())).toBe(false);
	});
});

// --- detectReasoning (unit) --------------------------------------------------

describe("detectReasoning", () => {
	it("takes the capability flag when the engine reports it true", () => {
		expect(detectReasoning({ chat_template_caps: { supports_preserve_reasoning: true } })).toBe(true);
	});

	it("reads a think block from the template when the capability flag is false or absent", () => {
		expect(
			detectReasoning({
				chat_template_caps: { supports_preserve_reasoning: false },
				chat_template: "{{ '<think>' }}",
			}),
		).toBe(true);
		expect(detectReasoning({ chat_template: "{%- if enable_thinking is true %}" })).toBe(true);
		expect(detectReasoning({ chat_template: "{%- set r = reasoning_effort|default('xhigh') %}" })).toBe(true);
		expect(detectReasoning({ chat_template: "{%- if preserve_thinking %}" })).toBe(true);
		expect(detectReasoning({ chat_template: "{%- if reasoning_content %}" })).toBe(true);
		expect(detectReasoning({ chat_template: "{{ '<|thinking|>' }}" })).toBe(true);
	});

	it("returns false when a reported template renders no reasoning", () => {
		expect(detectReasoning({ chat_template: "{{ message.content }}" })).toBe(false);
		expect(detectReasoning({ chat_template: "Just prose about thinking.", chat_template_caps: {} })).toBe(false);
	});

	it("returns undefined when the engine gives no reasoning signal at all", () => {
		expect(detectReasoning({ default_generation_settings: { n_ctx: 2048 } })).toBeUndefined();
		expect(detectReasoning({ modalities: { vision: true } })).toBeUndefined();
	});
});

// --- detectAnswerNow (unit) --------------------------------------------------

describe("detectAnswerNow", () => {
	it("takes the capability when the engine lists answer_now", () => {
		expect(detectAnswerNow({ basalt: { version: "0.3.3", capabilities: ["answer_now", "logprobs"] } })).toBe(true);
	});

	it("returns false when the engine lists capabilities without answer_now", () => {
		expect(detectAnswerNow({ basalt: { capabilities: ["logprobs", "documents"] } })).toBe(false);
	});

	it("returns undefined when the engine reports no basalt block", () => {
		expect(detectAnswerNow({ default_generation_settings: { n_ctx: 2048 } })).toBeUndefined();
		expect(detectAnswerNow({ chat_template: "{{ '<think>' }}" })).toBeUndefined();
	});

	it("ignores llama.cpp's own capabilities field, which never carries answer_now", () => {
		expect(detectAnswerNow({ capabilities: ["answer_now"] })).toBeUndefined();
	});

	it("returns false when basalt reports no capabilities list", () => {
		expect(detectAnswerNow({ basalt: { version: "0.3.3" } })).toBe(false);
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

describe("detectEngine", () => {
	it("names basalt when /props carries a basalt block", () => {
		expect(detectEngine({ basalt: { version: "0.4.1", capabilities: ["answer_now"] } })).toEqual({ id: "basalt" });
	});

	it("names strata from a build string that starts with strata", () => {
		expect(detectEngine({ build_info: "strata 0.2.0" })).toEqual({ id: "strata", build: "strata 0.2.0" });
	});

	it("names llama.cpp from a build string of the form b9468-354ebac8", () => {
		expect(detectEngine({ build_info: "b9468-354ebac8" })).toEqual({ id: "llamacpp", build: "b9468-354ebac8" });
	});

	it("trims the build string before storing it", () => {
		expect(detectEngine({ build_info: "  b9468-354ebac8 " })).toEqual({ id: "llamacpp", build: "b9468-354ebac8" });
	});

	it("caps a long build string at 120 characters", () => {
		const detected = detectEngine({ build_info: `b1-${"x".repeat(300)}` });

		expect(detected?.id).toBe("llamacpp");
		expect(detected?.build).toHaveLength(120);
	});

	it("names nothing when the props carry no engine signal", () => {
		expect(detectEngine({ chat_template: "{{ '<think>' }}" })).toBeUndefined();
		expect(detectEngine({ build_info: "built locally" })).toBeUndefined();
		expect(detectEngine({ build_info: "" })).toBeUndefined();
		expect(detectEngine({ build_info: 42 })).toBeUndefined();
	});

	it("names nothing for the error body llama-swap answers for a model it refused to load", () => {
		expect(detectEngine({ error: { message: "model is not loaded" } })).toBeUndefined();
	});

	it("names nothing for llama-swap's own props stub", () => {
		expect(detectEngine({ default_generation_settings: { n_ctx: 2048 } })).toBeUndefined();
	});

	it("keeps the basalt signal ahead of a build string", () => {
		expect(detectEngine({ basalt: {}, build_info: "b9468-354ebac8" })).toEqual({ id: "basalt" });
	});
});

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
			imageInput: false,
		});
		// The proxy props URL was hit; the origin fallback was not.
		const urls = spy.mock.calls.map((c) => String(c[0]));
		expect(urls).toContain("http://127.0.0.1:9000/props");
		expect(urls.some((u) => u.includes("127.0.0.1:8080/props"))).toBe(false);
	});

	it("records the explicit running state when present", async () => {
		mockFetch(
			withRunning(jsonResponse({ running: [{ model: "model-1", state: "starting" }] }), () => jsonResponse({})),
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
		expect(urls).toContain("http://127.0.0.1:8080/props?model=model%2F1&autoload=false");
	});

	it("falls through to /upstream/<id>/props when /props?model= answers 404", async () => {
		const strataProps = {
			default_generation_settings: { n_ctx: 262144 },
			build_info: "Strata 0.1.63",
			chat_template: "{% if enable_thinking %}think{% endif %}",
		};
		const handler = (url: string | URL | Request): Response => {
			const u = typeof url === "string" ? url : url.toString();
			if (u.endsWith("/running")) {
				return jsonResponse({
					running: [{ model: "model-1", proxy: "http://127.0.0.1:9000", state: "ready" }],
				});
			}
			if (u === "http://127.0.0.1:9000/props") {
				return new Response("connection refused", { status: 502 });
			}
			if (u === "http://127.0.0.1:8080/upstream/model-1/props") {
				return jsonResponse(strataProps);
			}
			return jsonResponse({ error: { message: "model not found" } }, 404);
		};
		const spy = mockFetch(handler);
		const result = await buildModelLimits([createEntry({ id: "model-1" })], createInstance());

		expect(result.contextByModel.get("model-1")).toBe(262144);
		expect(result.reasoningByModel.get("model-1")).toBe(true);
		expect(result.detectedByModel.get("model-1")).toEqual({
			contextWindow: 262144,
			imageInput: false,
			reasoning: true,
			engine: "strata",
			engineBuild: "Strata 0.1.63",
		});
		const urls = spy.mock.calls.map((c) => String(c[0]));
		expect(urls).toContain("http://127.0.0.1:8080/upstream/model-1/props");
	});

	it("does not probe /upstream when the engine answers /props?model=", async () => {
		const spy = mockFetch(
			withRunning(
				jsonResponse({ running: [{ model: "qwen-basalt" }] }),
				routeProps({
					"qwen-basalt": {
						default_generation_settings: { n_ctx: 196608 },
						basalt: { capabilities: ["answer_now"] },
					},
				}),
			),
		);
		const result = await buildModelLimits([createEntry({ id: "qwen-basalt" })], createInstance());

		expect(result.detectedByModel.get("qwen-basalt")).toEqual({
			contextWindow: 196608,
			imageInput: false,
			engine: "basalt",
			answerNow: true,
		});
		const urls = spy.mock.calls.map((c) => String(c[0]));
		expect(urls.some((u) => u.includes("/upstream/"))).toBe(false);
	});

	it("does not probe /upstream for a model that is not ready", async () => {
		const spy = mockFetch(
			withRunning(jsonResponse({ running: [{ model: "model-1", state: "starting" }] }), () =>
				jsonResponse({ error: { message: "model not found" } }, 404),
			),
		);
		const result = await buildModelLimits([createEntry({ id: "model-1" })], createInstance());

		expect(result.runningStateByModel.get("model-1")).toBe("starting");
		const urls = spy.mock.calls.map((c) => String(c[0]));
		expect(urls.some((u) => u.includes("/upstream/"))).toBe(false);
	});

	it("does not probe /upstream when autoload=false refused to load the model", async () => {
		const busy = jsonResponse(
			{
				src: "llama-swap",
				error: { message: "model model-1 is not loaded; autoload=false requests cannot start it" },
			},
			409,
		);
		const spy = mockFetch(
			withRunning(jsonResponse({ running: [{ model: "model-1", state: "ready" }] }), () => busy),
		);

		await buildModelLimits([createEntry({ id: "model-1" })], createInstance());

		const urls = spy.mock.calls.map((c) => String(c[0]));
		expect(urls.some((u) => u.includes("/upstream/"))).toBe(false);
	});

	it("does not probe /upstream when /props?model= gives no answer at all", async () => {
		const spy = mockFetch(
			withRunning(jsonResponse({ running: [{ model: "model-1", state: "ready" }] }), () => {
				throw new Error("fetch failed");
			}),
		);

		await buildModelLimits([createEntry({ id: "model-1" })], createInstance());

		const urls = spy.mock.calls.map((c) => String(c[0]));
		expect(urls.some((u) => u.includes("/upstream/"))).toBe(false);
	});

	it("does not probe /upstream when /running reports no state", async () => {
		const spy = mockFetch(
			withRunning(jsonResponse({ running: [{ model: "model-1" }] }), () =>
				jsonResponse({ error: { message: "model not found" } }, 404),
			),
		);

		await buildModelLimits([createEntry({ id: "model-1" })], createInstance());

		const urls = spy.mock.calls.map((c) => String(c[0]));
		expect(urls.some((u) => u.includes("/upstream/"))).toBe(false);
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
		expect(urls).toContain("http://127.0.0.1:8080/props?model=model-1&autoload=false");
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

	it("detects reasoning from the chat template of a non-llama.cpp engine", async () => {
		const template = "{%- if enable_thinking is true %}{{ reasoning_effort }}{% endif %}{{ '<think>' }}";
		mockFetch(
			withRunning(
				jsonResponse({ running: [{ model: "basalt-model" }] }),
				routeProps({
					"basalt-model": {
						chat_template: template,
						modalities: { vision: true },
						build_info: "basalt 0.1.0",
					},
				}),
			),
		);
		const result = await buildModelLimits([createEntry({ id: "basalt-model" })], createInstance());

		expect(result.detectedByModel.get("basalt-model")?.reasoning).toBe(true);
		expect(result.reasoningByModel.get("basalt-model")).toBe(true);
	});

	it("records no reasoning flag for an engine that reports neither caps nor template", async () => {
		mockFetch(
			withRunning(
				jsonResponse({ running: [{ model: "model-1" }] }),
				routeProps({ "model-1": { default_generation_settings: { n_ctx: 2048 } } }),
			),
		);
		const result = await buildModelLimits([createEntry({ id: "model-1" })], createInstance());

		expect(result.reasoningByModel.has("model-1")).toBe(false);
		expect("reasoning" in (result.detectedByModel.get("model-1") ?? {})).toBe(false);
	});

	it("keeps a cached reasoning true across a pass where the running engine says nothing", async () => {
		const config = createInstance({ modelCapabilities: { "model-1": { reasoning: true } } });
		mockFetch(
			withRunning(
				jsonResponse({ running: [{ model: "model-1" }] }),
				routeProps({ "model-1": { default_generation_settings: { n_ctx: 2048 } } }),
			),
		);
		const result = await buildModelLimits([createEntry({ id: "model-1" })], config);

		expect(result.reasoningByModel.get("model-1")).toBe(true);
		expect("reasoning" in (result.detectedByModel.get("model-1") ?? {})).toBe(false);
	});

	it("records answerNow true from a running engine that advertises the capability", async () => {
		mockFetch(
			withRunning(
				jsonResponse({ running: [{ model: "basalt-model" }] }),
				routeProps({
					"basalt-model": {
						chat_template: "{{ '<think>' }}",
						basalt: { version: "0.3.3", capabilities: ["answer_now", "logprobs", "documents"] },
					},
				}),
			),
		);
		const result = await buildModelLimits([createEntry({ id: "basalt-model" })], createInstance());

		expect(result.detectedByModel.get("basalt-model")?.answerNow).toBe(true);
	});

	it("records answerNow false from a running engine that reports basalt without it", async () => {
		mockFetch(
			withRunning(
				jsonResponse({ running: [{ model: "basalt-model" }] }),
				routeProps({ "basalt-model": { basalt: { version: "0.2.1", capabilities: ["logprobs"] } } }),
			),
		);
		const result = await buildModelLimits([createEntry({ id: "basalt-model" })], createInstance());

		expect(result.detectedByModel.get("basalt-model")?.answerNow).toBe(false);
	});

	it("records no answerNow flag for an engine whose props carry no basalt block", async () => {
		mockFetch(
			withRunning(
				jsonResponse({ running: [{ model: "model-1" }] }),
				routeProps({ "model-1": { chat_template: "{{ '<think>' }}", capabilities: ["vision"] } }),
			),
		);
		const result = await buildModelLimits([createEntry({ id: "model-1" })], createInstance());

		expect("answerNow" in (result.detectedByModel.get("model-1") ?? {})).toBe(false);
	});

	it("records the engine from a running Basalt engine", async () => {
		mockFetch(
			withRunning(
				jsonResponse({ running: [{ model: "basalt-model" }] }),
				routeProps({ "basalt-model": { basalt: { version: "0.4.1", capabilities: ["answer_now"] } } }),
			),
		);
		const result = await buildModelLimits([createEntry({ id: "basalt-model" })], createInstance());

		expect(result.detectedByModel.get("basalt-model")?.engine).toBe("basalt");
		expect(result.detectedByModel.get("basalt-model")?.engineBuild).toBeUndefined();
	});

	it("records the build string from a running llama.cpp engine", async () => {
		mockFetch(
			withRunning(
				jsonResponse({ running: [{ model: "llamacpp-model" }] }),
				routeProps({ "llamacpp-model": { chat_template: "{{ '<think>' }}", build_info: " b9468-354ebac8 " } }),
			),
		);
		const result = await buildModelLimits([createEntry({ id: "llamacpp-model" })], createInstance());

		expect(result.detectedByModel.get("llamacpp-model")?.engine).toBe("llamacpp");
		expect(result.detectedByModel.get("llamacpp-model")?.engineBuild).toBe("b9468-354ebac8");
	});

	it("records the engine from a running Strata engine", async () => {
		mockFetch(
			withRunning(
				jsonResponse({ running: [{ model: "strata-model" }] }),
				routeProps({ "strata-model": { build_info: "strata 0.2.0" } }),
			),
		);
		const result = await buildModelLimits([createEntry({ id: "strata-model" })], createInstance());

		expect(result.detectedByModel.get("strata-model")?.engine).toBe("strata");
	});

	it("records no engine for an engine whose props carry no engine signal", async () => {
		mockFetch(
			withRunning(
				jsonResponse({ running: [{ model: "model-1" }] }),
				routeProps({ "model-1": { chat_template: "{{ '<think>' }}", build_info: "built locally" } }),
			),
		);
		const result = await buildModelLimits([createEntry({ id: "model-1" })], createInstance());

		expect("engine" in (result.detectedByModel.get("model-1") ?? {})).toBe(false);
		expect("engineBuild" in (result.detectedByModel.get("model-1") ?? {})).toBe(false);
	});

	it("records no engine from an error body, so a refusal names nothing", async () => {
		mockFetch(
			withRunning(
				jsonResponse({ running: [{ model: "model-1" }] }),
				routeProps({ "model-1": { error: { message: "model is not loaded" } } }),
			),
		);
		const result = await buildModelLimits([createEntry({ id: "model-1" })], createInstance());

		expect("engine" in (result.detectedByModel.get("model-1") ?? {})).toBe(false);
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
			mockFetch(withRunning(jsonResponse({ running: [{ model: "model-1" }] }), routeProps({ "model-1": props })));
			const result = await buildModelLimits([createEntry({ id: "model-1" })], createInstance());
			expect(result.imageInputByModel.get("model-1"), JSON.stringify(props)).toBe(true);
			expect(result.detectedByModel.get("model-1")?.imageInput, JSON.stringify(props)).toBe(true);
			vi.unstubAllGlobals();
		}
	});

	it("does not detect image input from negative /props shapes", async () => {
		const negative = [{ capabilities: ["text"] }, { input: { image: false } }, {}];
		for (const props of negative) {
			mockFetch(withRunning(jsonResponse({ running: [{ model: "model-1" }] }), routeProps({ "model-1": props })));
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

	it("detects image input for an unloaded vision model from the /v1/models entry", async () => {
		mockFetch(withRunning(jsonResponse({ running: [] }), () => jsonResponse({})));
		const entries = [
			createEntry({
				id: "gemma4-26b-vision",
				context_length: 131072,
				capabilities: { function_calling: true, vision: true },
				architecture: { input_modalities: ["text", "image"], modality: "text+image->text" },
			}),
			createEntry({
				id: "qwen35-9b-summar",
				capabilities: { function_calling: true },
				architecture: { input_modalities: ["text"] },
			}),
		];
		const result = await buildModelLimits(entries, createInstance());

		// No /running entry and no cache: the entry flags are the only source.
		expect(result.imageInputByModel.get("gemma4-26b-vision")).toBe(true);
		expect(result.imageInputByModel.has("qwen35-9b-summar")).toBe(false);
		// Entry-derived vision is a live discovery, so it gets cached too.
		expect(result.detectedByModel.get("gemma4-26b-vision")).toEqual({ contextWindow: 131072, imageInput: true });
		expect(result.detectedByModel.has("qwen35-9b-summar")).toBe(false);
	});

	it("keeps a declared vision model image-capable when the running /props reports none", async () => {
		mockFetch(
			withRunning(
				jsonResponse({ running: [{ model: "model-1", state: "ready" }] }),
				routeProps({ "model-1": { capabilities: ["text"] } }),
			),
		);
		const result = await buildModelLimits(
			[createEntry({ id: "model-1", capabilities: { vision: true } })],
			createInstance(),
		);

		// An mmproj that is not loaded right now must not demote the model.
		expect(result.imageInputByModel.get("model-1")).toBe(true);
		expect(result.detectedByModel.get("model-1")?.imageInput).toBe(true);
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
			imageInput: false,
		});
	});

	it("stops the live probes when the caller cancels", async () => {
		const urls: string[] = [];
		const controller = new AbortController();
		controller.abort();
		mockFetch((url, init) => {
			const u = typeof url === "string" ? url : url.toString();
			urls.push(u);
			if (init?.signal?.aborted) {
				throw new DOMException("The operation was aborted.", "AbortError");
			}
			return jsonResponse({ running: [{ model: "model-1", proxy: "http://127.0.0.1:9000" }] });
		});

		const result = await buildModelLimits(
			[createEntry({ id: "model-1", context_length: 4096 })],
			createInstance(),
			undefined,
			controller.signal,
		);

		// Values from the /v1/models entries survive; the cancelled live pass adds nothing.
		expect(result.contextByModel.get("model-1")).toBe(4096);
		expect(result.runningStateByModel.size).toBe(0);
		// The aborted /running probe never reached /props.
		expect(urls.some((u) => u.includes("/props"))).toBe(false);
	});
});
