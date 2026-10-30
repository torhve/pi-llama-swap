import { homedir } from "node:os";
import { join } from "node:path";
import { afterEach, describe, expect, it, vi } from "vitest";

import { agentDir, configPath } from "../lib/config.js";

describe("agent directory against the real pi package", () => {
	afterEach(() => {
		vi.unstubAllEnvs();
	});

	it("defaults to ~/.pi/agent", () => {
		vi.stubEnv("PI_CODING_AGENT_DIR", "");
		expect(agentDir()).toBe(join(homedir(), ".pi", "agent"));
		expect(configPath()).toBe(join(homedir(), ".pi", "agent", "pi-llama-swap.json"));
	});

	it("follows PI_CODING_AGENT_DIR", () => {
		vi.stubEnv("PI_CODING_AGENT_DIR", "/tmp/pi-agent-dir");
		expect(agentDir()).toBe("/tmp/pi-agent-dir");
		expect(configPath()).toBe("/tmp/pi-agent-dir/pi-llama-swap.json");
	});
});
