import { afterEach, describe, expect, it, vi } from "vitest";

import { fetchModels, LlamaSwapClientError } from "../lib/client.js";
import { jsonResponse, mockFetch } from "./mocks.js";

const BASE = "http://127.0.0.1:8080/v1";

afterEach(() => {
	vi.unstubAllGlobals();
});

/** Resolves the caught error from an awaited promise, or fails the test. */
async function catchError(p: Promise<unknown>): Promise<unknown> {
	try {
		await p;
	} catch (err) {
		return err;
	}
	throw new Error("expected the promise to reject");
}

/**
 * A fetch mock that emulates the real fetch's abort behavior: it rejects with
 * an AbortError as soon as the (internal) signal aborts, and otherwise hangs.
 */
function abortableFetch() {
	return (_url: string | URL | Request, init?: RequestInit): Promise<Response> =>
		new Promise<Response>((resolve, reject) => {
			const signal = init?.signal;
			if (signal?.aborted) {
				reject(new DOMException("The operation was aborted.", "AbortError"));
				return;
			}
			signal?.addEventListener(
				"abort",
				() => reject(new DOMException("The operation was aborted.", "AbortError")),
				{ once: true },
			);
			// No resolve: simulates a request that stays in flight until aborted.
			void resolve;
		});
}

describe("fetchModels", () => {
	it("returns the data array and calls {base}/models", async () => {
		const spy = mockFetch(() => jsonResponse({ data: [{ id: "m1" }, { id: "m2" }] }));
		const result = await fetchModels(BASE);

		expect(result).toEqual([{ id: "m1" }, { id: "m2" }]);
		expect(spy).toHaveBeenCalledWith(BASE + "/models", expect.objectContaining({ method: "GET" }));
	});

	it("strips a trailing slash from the base URL", async () => {
		const spy = mockFetch(() => jsonResponse({ data: [] }));
		await fetchModels(BASE + "/");
		expect(spy).toHaveBeenCalledWith(BASE + "/models", expect.anything());
	});

	it("sends a Bearer Authorization header when an apiKey is provided", async () => {
		const spy = mockFetch(() => jsonResponse({ data: [] }));
		await fetchModels(BASE, "secret");
		expect(spy).toHaveBeenCalledWith(
			BASE + "/models",
			expect.objectContaining({ headers: expect.objectContaining({ Authorization: "Bearer secret" }) }),
		);
	});

	it("omits the Authorization header when no apiKey is provided", async () => {
		const spy = mockFetch(() => jsonResponse({ data: [] }));
		await fetchModels(BASE);
		const init = spy.mock.calls[0][1] as RequestInit;
		expect(init.headers).not.toHaveProperty("Authorization");
	});

	it("throws a 401 LlamaSwapClientError with the apiKey hint", async () => {
		mockFetch(() => new Response("nope", { status: 401, statusText: "Unauthorized" }));
		const err = (await catchError(fetchModels(BASE))) as LlamaSwapClientError;

		expect(err).toBeInstanceOf(LlamaSwapClientError);
		expect(err.status).toBe(401);
		expect(err.message).toContain("401");
		expect(err.message).toContain("apiKey");
	});

	it("includes the status and a 200-char body snippet on a 500", async () => {
		const body = "x".repeat(500);
		mockFetch(() => new Response(body, { status: 500, statusText: "Internal Server Error" }));
		const err = (await catchError(fetchModels(BASE))) as LlamaSwapClientError;

		expect(err.status).toBe(500);
		expect(err.message).toContain("500");
		expect(err.message).toContain("x".repeat(200));
		expect(err.message).not.toContain("x".repeat(201));
	});

	it("throws on invalid JSON", async () => {
		mockFetch(() => new Response("this is not json", { status: 200 }));
		await expect(fetchModels(BASE)).rejects.toThrow("Invalid JSON from llama-swap /v1/models");
	});

	it("throws when the response has no data array", async () => {
		mockFetch(() => jsonResponse({ object: "list" }));
		await expect(fetchModels(BASE)).rejects.toThrow("missing data array");
	});

	it("throws a LlamaSwapClientError when fetch rejects", async () => {
		mockFetch(() => {
			throw new Error("ECONNREFUSED");
		});
		const err = (await catchError(fetchModels(BASE))) as LlamaSwapClientError;
		expect(err).toBeInstanceOf(LlamaSwapClientError);
		expect(err.message).toMatch(/^Cannot reach llama-swap at/);
	});

	it("rejects when the signal is already aborted", async () => {
		mockFetch(abortableFetch());
		const controller = new AbortController();
		controller.abort();
		const err = (await catchError(fetchModels(BASE, undefined, controller.signal))) as LlamaSwapClientError;
		expect(err).toBeInstanceOf(LlamaSwapClientError);
		expect(err.message).toMatch(/^Cannot reach llama-swap at/);
	});

	it("rejects when the signal aborts mid-flight", async () => {
		mockFetch(abortableFetch());
		const controller = new AbortController();
		const pending = fetchModels(BASE, undefined, controller.signal);
		// Let fetchModels wire up the abort listener before we abort.
		await new Promise((r) => setTimeout(r, 0));
		controller.abort();
		const err = (await catchError(pending)) as LlamaSwapClientError;
		expect(err).toBeInstanceOf(LlamaSwapClientError);
		expect(err.message).toMatch(/^Cannot reach llama-swap at/);
	});
});

describe("LlamaSwapClientError", () => {
	it("sets name and leaves status undefined by default", () => {
		const err = new LlamaSwapClientError("boom");
		expect(err.name).toBe("LlamaSwapClientError");
		expect(err.message).toBe("boom");
		expect(err.status).toBeUndefined();
		expect(err).toBeInstanceOf(Error);
	});

	it("records the optional status", () => {
		const err = new LlamaSwapClientError("boom", 404);
		expect(err.status).toBe(404);
	});
});
