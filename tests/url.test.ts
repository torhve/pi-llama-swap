import { describe, expect, it } from "vitest";

import {
	buildBaseUrl,
	buildServerOrigin,
	defaultInstance,
	mergeConfig,
	normalizeBasePath,
	parsePortArg,
	parseUrlArg,
} from "../lib/url.js";
import type { LlamaSwapInstance } from "../lib/types.js";

describe("defaultInstance", () => {
	it("returns the documented defaults with empty id/name", () => {
		expect(defaultInstance()).toEqual({
			id: "",
			name: "",
			origin: "http://127.0.0.1",
			port: 8080,
			basePath: "/v1",
		});
	});

	it("returns a fresh object on each call", () => {
		expect(defaultInstance()).not.toBe(defaultInstance());
	});
});

describe("normalizeBasePath", () => {
	it("maps empty and root paths to /v1", () => {
		expect(normalizeBasePath("")).toBe("/v1");
		expect(normalizeBasePath("/")).toBe("/v1");
	});

	it("keeps /v1 as-is, with or without trailing slash", () => {
		expect(normalizeBasePath("/v1")).toBe("/v1");
		expect(normalizeBasePath("/v1/")).toBe("/v1");
	});

	it("appends /v1 to a custom path, stripping the trailing slash", () => {
		expect(normalizeBasePath("/custom")).toBe("/custom/v1");
		expect(normalizeBasePath("/custom/")).toBe("/custom/v1");
	});
});

describe("buildServerOrigin", () => {
	it("builds {scheme}://{host}:{port}", () => {
		expect(buildServerOrigin({ origin: "http://127.0.0.1", port: 8080 } as LlamaSwapInstance)).toBe(
			"http://127.0.0.1:8080",
		);
	});

	it("prepends http:// to a bare host", () => {
		expect(buildServerOrigin({ origin: "127.0.0.1", port: 8080 } as LlamaSwapInstance)).toBe(
			"http://127.0.0.1:8080",
		);
	});

	it("applies the configured port", () => {
		expect(buildServerOrigin({ origin: "http://example.com", port: 9000 } as LlamaSwapInstance)).toBe(
			"http://example.com:9000",
		);
	});

	it("strips any path, search, and hash from the origin", () => {
		expect(
			buildServerOrigin({ origin: "http://example.com/some/path?x=1#frag", port: 8080 } as LlamaSwapInstance),
		).toBe("http://example.com:8080");
	});
});

describe("buildBaseUrl", () => {
	it("defaults the base path to /v1 when basePath is unset", () => {
		expect(buildBaseUrl({ origin: "http://127.0.0.1", port: 8080 } as LlamaSwapInstance)).toBe(
			"http://127.0.0.1:8080/v1",
		);
	});

	it("normalizes a custom basePath", () => {
		expect(
			buildBaseUrl({ origin: "http://127.0.0.1", port: 8080, basePath: "/custom" } as LlamaSwapInstance),
		).toBe("http://127.0.0.1:8080/custom/v1");
	});

	it("prepends http:// to a bare host", () => {
		expect(buildBaseUrl({ origin: "127.0.0.1", port: 8080 } as LlamaSwapInstance)).toBe(
			"http://127.0.0.1:8080/v1",
		);
	});

	it("strips a trailing slash from the result", () => {
		expect(
			buildBaseUrl({ origin: "http://127.0.0.1", port: 8080, basePath: "/v1/" } as LlamaSwapInstance),
		).toBe("http://127.0.0.1:8080/v1");
	});
});

describe("parseUrlArg", () => {
	it("parses a bare host into an http origin with /v1", () => {
		expect(parseUrlArg("127.0.0.1")).toEqual({
			origin: "http://127.0.0.1",
			port: undefined,
			basePath: "/v1",
		});
	});

	it("parses a scheme + host", () => {
		expect(parseUrlArg("http://example.com")).toEqual({
			origin: "http://example.com",
			port: undefined,
			basePath: "/v1",
		});
	});

	it("parses host:port", () => {
		expect(parseUrlArg("example.com:9000")).toEqual({
			origin: "http://example.com",
			port: 9000,
			basePath: "/v1",
		});
	});

	it("normalizes the base path from a URL with a path", () => {
		expect(parseUrlArg("http://example.com/custom/")).toEqual({
			origin: "http://example.com",
			port: undefined,
			basePath: "/custom/v1",
		});
	});

	it("throws on an empty argument", () => {
		expect(() => parseUrlArg("")).toThrow();
		expect(() => parseUrlArg("   ")).toThrow();
	});

	it("throws on an invalid URL", () => {
		expect(() => parseUrlArg("http://")).toThrow("Invalid URL: http://");
	});

	it("rejects out-of-range ports", () => {
		// The WHATWG URL parser itself rejects ports above 65535.
		expect(() => parseUrlArg("http://example.com:99999")).toThrow();
		// Port 0 survives the URL parser and is rejected by the explicit range check.
		expect(() => parseUrlArg("http://example.com:0")).toThrow("Port out of range: 0");
	});
});

describe("parsePortArg", () => {
	it("parses a valid port", () => {
		expect(parsePortArg("8080")).toBe(8080);
	});

	it("accepts the upper bound 65535", () => {
		expect(parsePortArg("65535")).toBe(65535);
	});

	it("throws on an empty argument", () => {
		expect(() => parsePortArg("")).toThrow();
		expect(() => parsePortArg("   ")).toThrow();
	});

	it("throws on a non-numeric argument", () => {
		expect(() => parsePortArg("abc")).toThrow();
	});

	it("throws on zero", () => {
		expect(() => parsePortArg("0")).toThrow();
	});

	it("throws on a port above 65535", () => {
		expect(() => parsePortArg("65536")).toThrow();
	});

	it("throws on a non-integer port", () => {
		expect(() => parsePortArg("80.5")).toThrow();
	});
});

describe("mergeConfig", () => {
	const current: LlamaSwapInstance = {
		id: "llama-swap",
		name: "Llama Swap",
		origin: "http://127.0.0.1",
		port: 8080,
		basePath: "/v1",
		apiKey: "old-key",
		contextOverrides: { "model-1": 4096 },
		modelCapabilities: { "model-1": { reasoning: true } },
	};

	it("lets defined partial fields win over current", () => {
		const merged = mergeConfig(current, { origin: "http://other", port: 9000 });
		expect(merged.origin).toBe("http://other");
		expect(merged.port).toBe(9000);
		// Untouched fields keep the current values.
		expect(merged.id).toBe("llama-swap");
		expect(merged.name).toBe("Llama Swap");
		expect(merged.basePath).toBe("/v1");
	});

	it("keeps every current field when the partial is empty", () => {
		expect(mergeConfig(current, {})).toEqual(current);
	});

	it("replaces apiKey when explicitly set, even to an empty string", () => {
		expect(mergeConfig(current, { apiKey: "" }).apiKey).toBe("");
	});

	it("keeps the current apiKey when the partial omits it", () => {
		expect(mergeConfig(current, { port: 9000 }).apiKey).toBe("old-key");
	});

	it("passes through contextOverrides and modelCapabilities when provided", () => {
		const overrides = { "model-2": 8192 };
		const caps = { "model-2": { imageInput: true } };
		const merged = mergeConfig(current, { contextOverrides: overrides, modelCapabilities: caps });
		expect(merged.contextOverrides).toBe(overrides);
		expect(merged.modelCapabilities).toBe(caps);
	});

	it("returns a new object, not the same reference", () => {
		expect(mergeConfig(current, {})).not.toBe(current);
	});
});
