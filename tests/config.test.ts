import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { join } from "node:path";

const { readFileMock, writeFileMock, homedirMock } = vi.hoisted(() => ({
	readFileMock: vi.fn(),
	writeFileMock: vi.fn(),
	homedirMock: vi.fn(() => "/fake/home"),
}));

vi.mock("node:fs/promises", () => ({
	readFile: readFileMock,
	writeFile: writeFileMock,
}));

vi.mock("node:os", async (importOriginal) => {
	const original = await importOriginal<typeof import("node:os")>();
	return { ...original, homedir: homedirMock };
});

import {
	applyEnvOverrides,
	configPath,
	loadConfig,
	saveContextOverride,
	saveModelCapabilities,
} from "../lib/config.js";
import type { LlamaSwapConfig } from "../lib/types.js";

const FAKE_HOME = "/fake/home";
const CONFIG_PATH = join(FAKE_HOME, ".pi", "agent", "pi-llama-swap.json");

/** An ENOENT error, the shape fs rejects with when a file is missing. */
function enoent(): NodeJS.ErrnoException {
	const err = new Error("ENOENT: no such file or directory") as NodeJS.ErrnoException;
	err.code = "ENOENT";
	return err;
}

/** Reads the JSON that the most recent writeFile call was given. */
function lastWritten(): Record<string, unknown> {
	const call = writeFileMock.mock.calls[writeFileMock.mock.calls.length - 1];
	return JSON.parse(call[1] as string);
}

function baseConfig(): LlamaSwapConfig {
	return {
		instances: [
			{ id: "llama-swap", name: "Llama Swap", origin: "http://127.0.0.1", port: 8080, basePath: "/v1" },
			{ id: "llama-swap-2", name: "Llama Swap 2", origin: "http://127.0.0.1", port: 8081, basePath: "/v1" },
		],
	};
}

beforeEach(() => {
	vi.clearAllMocks();
	homedirMock.mockReturnValue(FAKE_HOME);
});

afterEach(() => {
	vi.unstubAllEnvs();
});

describe("configPath", () => {
	it("points to ~/.pi/agent/pi-llama-swap.json", () => {
		expect(configPath()).toBe(CONFIG_PATH);
	});
});

describe("loadConfig", () => {
	it("returns the default single instance when the config file is missing", async () => {
		readFileMock.mockRejectedValue(enoent());
		const config = await loadConfig();

		expect(config.instances).toHaveLength(1);
		const inst = config.instances[0];
		expect(inst.id).toBe("llama-swap");
		expect(inst.name).toBe("Llama Swap");
		expect(inst.origin).toBe("http://127.0.0.1");
		expect(inst.port).toBe(8080);
		expect(inst.basePath).toBe("/v1");
		expect(inst.apiKey).toBeUndefined();
	});

	it("parses the legacy single-instance shape", async () => {
		readFileMock.mockResolvedValue(
			JSON.stringify({ origin: "http://10.0.0.5", port: 9090, apiKey: "legacy-key", contextOverrides: { "model-a": 8192 } }),
		);
		const config = await loadConfig();

		expect(config.instances).toHaveLength(1);
		const inst = config.instances[0];
		expect(inst.id).toBe("llama-swap");
		expect(inst.name).toBe("Llama Swap");
		expect(inst.origin).toBe("http://10.0.0.5");
		expect(inst.port).toBe(9090);
		expect(inst.apiKey).toBe("legacy-key");
		expect(inst.contextOverrides).toEqual({ "model-a": 8192 });
	});

	it("parses the multi-instance shape, generating ids and names for entries that lack them", async () => {
		readFileMock.mockResolvedValue(
			JSON.stringify({
				instances: [
					{ origin: "http://127.0.0.1", port: 8080 },
					{ origin: "http://127.0.0.1", port: 8081 },
				],
			}),
		);
		const config = await loadConfig();

		expect(config.instances).toHaveLength(2);
		expect(config.instances[0].id).toBe("llama-swap");
		expect(config.instances[0].name).toBe("Llama Swap");
		expect(config.instances[0].port).toBe(8080);
		expect(config.instances[1].id).toBe("llama-swap-2");
		expect(config.instances[1].name).toBe("Llama Swap 2");
		expect(config.instances[1].port).toBe(8081);
	});

	it("keeps explicit ids and names in the multi-instance shape", async () => {
		readFileMock.mockResolvedValue(
			JSON.stringify({
				instances: [{ id: "my-swap", name: "My Swap", origin: "http://127.0.0.1", port: 8080 }],
			}),
		);
		const config = await loadConfig();

		expect(config.instances[0].id).toBe("my-swap");
		expect(config.instances[0].name).toBe("My Swap");
	});

	it("generates a unique id when an explicit id collides with an auto-generated one", async () => {
		readFileMock.mockResolvedValue(
			JSON.stringify({
				instances: [
					{ id: "llama-swap-2", origin: "http://a" },
					{ origin: "http://b" },
				],
			}),
		);
		const config = await loadConfig();

		// index 1 would normally generate "llama-swap-2", but that id is taken,
		// so it falls back to "llama-swap-2-2".
		expect(config.instances[0].id).toBe("llama-swap-2");
		expect(config.instances[1].id).toBe("llama-swap-2-2");
	});

	it("ignores invalid connection fields and falls back to defaults", async () => {
		readFileMock.mockResolvedValue(JSON.stringify({ origin: "http://127.0.0.1", port: 99999, basePath: "  " }));
		const config = await loadConfig();

		const inst = config.instances[0];
		expect(inst.origin).toBe("http://127.0.0.1");
		// port 99999 is out of range -> dropped -> default 8080.
		expect(inst.port).toBe(8080);
		// blank basePath -> dropped -> default /v1.
		expect(inst.basePath).toBe("/v1");
	});

	it("throws on a malformed JSON file", async () => {
		readFileMock.mockResolvedValue("{ not valid json");
		await expect(loadConfig()).rejects.toThrow();
	});

	it("applies environment overrides to the first instance", async () => {
		readFileMock.mockRejectedValue(enoent());
		vi.stubEnv("LLAMA_SWAP_URL", "http://10.1.1.1:9999/api");
		vi.stubEnv("LLAMA_SWAP_API_KEY", "env-key");
		const config = await loadConfig();

		const inst = config.instances[0];
		expect(inst.origin).toBe("http://10.1.1.1");
		expect(inst.port).toBe(9999);
		expect(inst.basePath).toBe("/api/v1");
		expect(inst.apiKey).toBe("env-key");
	});
});

describe("applyEnvOverrides", () => {
	it("overrides origin, port, and basePath on the first instance via LLAMA_SWAP_URL", () => {
		vi.stubEnv("LLAMA_SWAP_URL", "http://10.1.1.1:9999/api");
		const result = applyEnvOverrides(baseConfig());

		expect(result.instances[0].origin).toBe("http://10.1.1.1");
		expect(result.instances[0].port).toBe(9999);
		expect(result.instances[0].basePath).toBe("/api/v1");
	});

	it("LLAMA_SWAP_PORT takes precedence over the port embedded in LLAMA_SWAP_URL", () => {
		vi.stubEnv("LLAMA_SWAP_URL", "http://10.1.1.1:9999/api");
		vi.stubEnv("LLAMA_SWAP_PORT", "7777");
		const result = applyEnvOverrides(baseConfig());

		expect(result.instances[0].origin).toBe("http://10.1.1.1");
		expect(result.instances[0].port).toBe(7777);
		expect(result.instances[0].basePath).toBe("/api/v1");
	});

	it("overrides apiKey via LLAMA_SWAP_API_KEY", () => {
		vi.stubEnv("LLAMA_SWAP_API_KEY", "env-key");
		const result = applyEnvOverrides(baseConfig());
		expect(result.instances[0].apiKey).toBe("env-key");
	});

	it("only touches the first instance", () => {
		vi.stubEnv("LLAMA_SWAP_URL", "http://10.1.1.1:9999");
		const config = baseConfig();
		const result = applyEnvOverrides(config);

		expect(result.instances[0].origin).toBe("http://10.1.1.1");
		// The second instance is returned untouched (same reference).
		expect(result.instances[1]).toBe(config.instances[1]);
	});

	it("returns the config unchanged when no env vars are set", () => {
		const config = baseConfig();
		const result = applyEnvOverrides(config);
		// No merge happens, so the first instance is the same reference.
		expect(result.instances[0]).toBe(config.instances[0]);
	});

	it("returns the same config when there are no instances", () => {
		const config: LlamaSwapConfig = { instances: [] };
		vi.stubEnv("LLAMA_SWAP_URL", "http://10.1.1.1:9999");
		expect(applyEnvOverrides(config)).toBe(config);
	});
});

describe("saveContextOverride", () => {
	it("sets a context override on an existing instance (multi shape)", async () => {
		readFileMock.mockResolvedValue(JSON.stringify({ instances: [{ id: "llama-swap", origin: "http://127.0.0.1", port: 8080 }] }));
		writeFileMock.mockResolvedValue(undefined);

		await saveContextOverride("llama-swap", "model-a", 8192);

		expect(writeFileMock).toHaveBeenCalledTimes(1);
		expect(writeFileMock.mock.calls[0][0]).toBe(CONFIG_PATH);
		expect(lastWritten().instances).toEqual([
			expect.objectContaining({ id: "llama-swap", contextOverrides: { "model-a": 8192 } }),
		]);
	});

	it("clears an override when ctxSize is undefined", async () => {
		readFileMock.mockResolvedValue(
			JSON.stringify({ instances: [{ id: "llama-swap", contextOverrides: { "model-a": 8192, "model-b": 4096 } }] }),
		);
		writeFileMock.mockResolvedValue(undefined);

		await saveContextOverride("llama-swap", "model-a", undefined);

		expect(lastWritten().instances).toEqual([expect.objectContaining({ contextOverrides: { "model-b": 4096 } })]);
	});

	it("appends a new instance entry when the id is not present", async () => {
		readFileMock.mockResolvedValue(JSON.stringify({ instances: [{ id: "other" }] }));
		writeFileMock.mockResolvedValue(undefined);

		await saveContextOverride("llama-swap", "model-a", 8192);

		const instances = lastWritten().instances as Record<string, unknown>[];
		expect(instances).toHaveLength(2);
		const entry = instances.find((i) => i.id === "llama-swap");
		expect(entry).toEqual({ id: "llama-swap", contextOverrides: { "model-a": 8192 } });
	});

	it("writes the legacy top-level shape when there is no instances array", async () => {
		readFileMock.mockResolvedValue(JSON.stringify({ origin: "http://127.0.0.1", port: 8080 }));
		writeFileMock.mockResolvedValue(undefined);

		await saveContextOverride("llama-swap", "model-a", 8192);

		expect(lastWritten().contextOverrides).toEqual({ "model-a": 8192 });
		expect(lastWritten()).not.toHaveProperty("instances");
	});

	it("starts from an empty object when the file is missing", async () => {
		readFileMock.mockRejectedValue(enoent());
		writeFileMock.mockResolvedValue(undefined);

		await saveContextOverride("llama-swap", "model-a", 8192);

		expect(lastWritten().contextOverrides).toEqual({ "model-a": 8192 });
	});
});

describe("saveModelCapabilities", () => {
	it("merges discovered caps per field, keeping previously cached fields", async () => {
		readFileMock.mockResolvedValue(
			JSON.stringify({ instances: [{ id: "llama-swap", modelCapabilities: { "model-a": { reasoning: true } } }] }),
		);
		writeFileMock.mockResolvedValue(undefined);

		await saveModelCapabilities("llama-swap", { "model-a": { contextWindow: 4096 }, "model-b": { imageInput: true } });

		expect(lastWritten().instances).toEqual([
			expect.objectContaining({
				modelCapabilities: {
					"model-a": { reasoning: true, contextWindow: 4096 },
					"model-b": { imageInput: true },
				},
			}),
		]);
	});

	it("does not write when caps is empty", async () => {
		readFileMock.mockResolvedValue(JSON.stringify({ instances: [] }));
		writeFileMock.mockResolvedValue(undefined);

		await saveModelCapabilities("llama-swap", {});

		expect(writeFileMock).not.toHaveBeenCalled();
	});

	it("appends a new instance entry when the id is not present", async () => {
		readFileMock.mockResolvedValue(JSON.stringify({ instances: [{ id: "other" }] }));
		writeFileMock.mockResolvedValue(undefined);

		await saveModelCapabilities("llama-swap", { "model-a": { reasoning: true } });

		const instances = lastWritten().instances as Record<string, unknown>[];
		const entry = instances.find((i) => i.id === "llama-swap");
		expect(entry).toEqual({ id: "llama-swap", modelCapabilities: { "model-a": { reasoning: true } } });
	});

	it("writes the legacy top-level shape when there is no instances array", async () => {
		readFileMock.mockResolvedValue(JSON.stringify({ origin: "http://127.0.0.1" }));
		writeFileMock.mockResolvedValue(undefined);

		await saveModelCapabilities("llama-swap", { "model-a": { reasoning: true } });

		expect(lastWritten().modelCapabilities).toEqual({ "model-a": { reasoning: true } });
	});
});
