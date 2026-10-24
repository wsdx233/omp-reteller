import { describe, expect, test } from "bun:test";
import { getBundledModels } from "@oh-my-pi/pi-catalog/models";
import type { Model } from "@oh-my-pi/pi-ai";
import { projectContext, readContext, retell } from "../src/retell.ts";

type WireRequest = { messages: Array<{ role: string; content: unknown }>; tools?: unknown[]; tool_choice?: string };
type Reply = { text?: string; calls?: Array<{ name: string; arguments: Record<string, unknown> }>; finish?: string };

function response(reply: Reply): Response {
  const chunk = (delta: unknown, finish_reason: string | null) => ({
    id: "chatcmpl-reteller-test",
    object: "chat.completion.chunk",
    created: 1,
    model: "reteller-test",
    choices: [{ index: 0, delta, finish_reason }],
  });
  const delta = reply.calls ? {
    role: "assistant",
    tool_calls: reply.calls.map((call, index) => ({
      index, id: `call_${index}`, type: "function",
      function: { name: call.name, arguments: JSON.stringify(call.arguments) },
    })),
  } : { role: "assistant", content: reply.text ?? "" };
  return new Response([
    `data: ${JSON.stringify(chunk(delta, null))}`,
    `data: ${JSON.stringify(chunk({}, reply.finish ?? (reply.calls ? "tool_calls" : "stop")))}`,
    "data: [DONE]",
    "",
  ].join("\n\n"), { headers: { "content-type": "text/event-stream" } });
}

function endpoint(reply: (request: WireRequest, index: number) => Reply | Promise<Reply>) {
  const requests: WireRequest[] = [];
  const server = Bun.serve({
    hostname: "127.0.0.1",
    port: 0,
    async fetch(request) {
      const body = await request.json() as WireRequest;
      requests.push(body);
      return response(await reply(body, requests.length - 1));
    },
  });
  const base = getBundledModels("openai")[0];
  const model: Model = { ...base, id: "reteller-test", api: "openai-completions", baseUrl: `${server.url}v1`, reasoning: false };
  return { model, requests, close: () => server.stop(true) };
}

const history = [
  { role: "system", content: "SECRET_SYSTEM" },
  { role: "developer", content: "SECRET_DEVELOPER" },
  { role: "user", content: [{ type: "text", text: "用户需要解释配额" }, { type: "image", data: "SECRET_IMAGE" }] },
  { role: "user", synthetic: true, content: "SECRET_SYNTHETIC" },
  { role: "user", attribution: "agent", content: "SECRET_INJECTED" },
  { role: "assistant", responseId: "SECRET_RESPONSE_ID", content: [
    { type: "thinking", thinking: "SECRET_THINKING", thinkingSignature: "SECRET_SIGNATURE" },
    { type: "redactedThinking", data: "SECRET_REDACTED" },
    { type: "text", text: "配额是每日最多 20 次请求。", textSignature: "SECRET_TEXT_SIGNATURE" },
    { type: "toolCall", name: "bash", arguments: { command: "SECRET_COMMAND" } },
  ] },
  { role: "toolResult", details: { password: "SECRET_DETAILS" }, content: [
    { type: "text", text: "剩余 12 次。" }, { type: "image", data: "SECRET_TOOL_IMAGE" },
  ] },
  { role: "custom", customType: "org.omp-reteller.result", content: "SECRET_PREVIOUS_RETELLING" },
];

describe("context disclosure boundary", () => {
  test("projects only visible user/assistant/tool text, never policy, thinking or metadata", () => {
    expect(projectContext(history)).toEqual([
      { source: "user", text: "用户需要解释配额" },
      { source: "assistant", text: "配额是每日最多 20 次请求。" },
      { source: "tool", text: "剩余 12 次。" },
    ]);
  });

  test("searches only the requested source and bounds an excerpt around its match", () => {
    const excerpts = [
      { source: "assistant" as const, text: "前".repeat(2500) + "配额是 20 次" + "后".repeat(2500) },
      { source: "assistant" as const, text: "不相关的新回复" },
      { source: "user" as const, text: "用户的配额问题" },
    ];
    const result = JSON.parse(readContext(excerpts, "assistant", "配额"));
    expect(result.text).toBe("前".repeat(300) + "配额是 20 次" + "后".repeat(1692));
    expect(result.truncatedBefore).toBe(true);
    expect(result.truncatedAfter).toBe(true);
    expect(JSON.parse(readContext(excerpts, "tool", "配额")).text).toBeNull();
  });
});

describe("isolated provider requests", () => {
  test("initial request carries only the final body and does not eagerly read history", async () => {
    const api = endpoint(() => ({ text: "每日配额为 20 次，目前剩余 12 次。" }));
    let reads = 0;
    try {
      await retell("剩12，每日20。", {
        model: api.model, apiKey: "local-test", signal: new AbortController().signal,
        context: () => { reads++; return projectContext(history); },
      });
      expect(api.requests[0].messages.filter(message => message.role === "user")).toEqual([
        { role: "user", content: "剩12，每日20。" },
      ]);
      expect(JSON.stringify(api.requests)).not.toContain("SECRET_");
      expect(reads).toBe(0);
    } finally { api.close(); }
  });

  test("an explicit tool request discloses only one matching visible excerpt", async () => {
    const api = endpoint((_request, index) => index === 0 ? {
      calls: [{ name: "read_context", arguments: { source: "assistant", query: "配额" } }],
    } : { text: "每日最多允许 20 次请求，目前剩余 12 次。" });
    try {
      const result = await retell("配额如前，剩余12次。", {
        model: api.model, apiKey: "local-test", signal: new AbortController().signal,
        context: () => projectContext(history),
      });
      expect(result.contextReads).toBe(1);
      const tools = api.requests[1].messages.filter(message => message.role === "tool");
      expect(tools).toHaveLength(1);
      expect(JSON.parse(String(tools[0].content)).text).toBe("配额是每日最多 20 次请求。");
      expect(JSON.stringify(api.requests)).not.toContain("SECRET_");
      expect(JSON.stringify(api.requests[1])).not.toContain("用户需要解释配额");
      expect(JSON.stringify(api.requests[1])).not.toContain("剩余 12 次。");
    } finally { api.close(); }
  });

  test("context off removes tools and refuses even a provider-issued read request", async () => {
    const api = endpoint((_request, index) => index === 0 ? {
      calls: [{ name: "read_context", arguments: { source: "user", query: "" } }],
    } : { text: "保留原意的复述。" });
    try {
      await retell("原正文。", { model: api.model, apiKey: "local-test", signal: new AbortController().signal });
      expect(api.requests[0].tools).toBeUndefined();
      const tools = api.requests[1].messages.filter(message => message.role === "tool");
      expect(tools).toHaveLength(1);
      expect(tools[0].content).not.toContain("用户需要解释配额");
      expect(JSON.stringify(api.requests)).not.toContain("SECRET_");
    } finally { api.close(); }
  });

  test("bounds contextual access and never executes arbitrary requested tools", async () => {
    const api = endpoint((_request, index) => index === 0 ? {
      calls: [
        { name: "bash", arguments: { command: "touch /tmp/reteller-must-not-execute" } },
        ...Array.from({ length: 4 }, () => ({ name: "read_context", arguments: { source: "user", query: "" } })),
      ],
    } : { text: "复述正文。" });
    let reads = 0;
    try {
      const result = await retell("原正文。", {
        model: api.model, apiKey: "local-test", signal: new AbortController().signal,
        context: () => { reads++; return [{ source: "user", text: "必要的历史正文。" }]; },
      });
      expect(reads).toBe(2);
      expect(result.contextReads).toBe(2);
      const tools = api.requests[1].messages.filter(message => message.role === "tool");
      expect(tools[0].content).not.toContain("必要的历史正文。");
      expect(tools[3].content).not.toContain("必要的历史正文。");
      expect(api.requests[1].tool_choice).toBe("none");
    } finally { api.close(); }
  });

  test.each([
    { text: "截断的正文", finish: "length" },
    { text: "", finish: "stop" },
  ])("rejects incomplete output: $finish", async reply => {
    const api = endpoint(() => reply);
    try {
      await expect(retell("原正文。", {
        model: api.model, apiKey: "local-test", signal: new AbortController().signal,
      })).rejects.toBeInstanceOf(Error);
    } finally { api.close(); }
  });

  test("cancellation discards a late provider response", async () => {
    const arrived = Promise.withResolvers<void>();
    const release = Promise.withResolvers<void>();
    const api = endpoint(async () => {
      arrived.resolve();
      await release.promise;
      return { text: "已经过期的复述。" };
    });
    const controller = new AbortController();
    try {
      const result = retell("原正文。", { model: api.model, apiKey: "local-test", signal: controller.signal });
      await arrived.promise;
      controller.abort();
      release.resolve();
      await expect(result).rejects.toThrow();
    } finally { release.resolve(); api.close(); }
  });
});
