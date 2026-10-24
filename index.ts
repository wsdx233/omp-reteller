import type {
  ExtensionAPI,
  ExtensionCommandContext,
  ExtensionContext,
  SessionEntry,
} from "@oh-my-pi/pi-coding-agent";
import { Loader, Markdown } from "@oh-my-pi/pi-tui";
import { getMarkdownTheme } from "@oh-my-pi/pi-tui/theme";
import { projectContext, retell, textOnly, type ContextExcerpt } from "./src/retell.ts";

const SETTINGS_TYPE = "org.omp-reteller.settings";
const RESULT_TYPE = "org.omp-reteller.result";
const PROGRESS_WIDGET = "org.omp-reteller.progress";

interface Settings {
  enabled: boolean;
  model: string;
  context: boolean;
  prompt: string;
}

interface Source {
  id: string;
  text: string;
  index: number;
}

interface RetellJob {
  controller: AbortController;
  sourceId: string;
  clearProgress?: () => void;
}

/** Do not fall back to an older answer if the latest assistant message failed or used tools. */
function latestSource(entries: readonly SessionEntry[]): Source | undefined {
  for (let index = entries.length - 1; index >= 0; index--) {
    const entry = entries[index];
    if (entry.type !== "message" || entry.message.role !== "assistant") continue;
    const message = entry.message;
    if (message.stopReason !== "stop" || message.content.some(block => block.type === "toolCall")) return;
    const text = textOnly(message.content);
    return text.trim() ? { id: entry.id, text, index } : undefined;
  }
}

const HELP = `/retell — 在原回复下方追加复述版（自动关闭时也可用）
/retell on | off — 开启或关闭后续回复的自动复述
/retell model [provider/model] — 指定复述模型；省略时打开选择器
/retell context on | off — 允许或禁止按需读取历史正文片段
/retell prompt <提示词> — 设置自定义复述风格
/retell prompt — 清除自定义提示词，恢复默认
/retell cancel — 取消正在进行的复述
/retell status — 查看当前配置
配置保存在当前会话分支，不切换主 agent 的模型。`;

export default function reteller(pi: ExtensionAPI) {
  pi.setLabel("Reteller");
  pi.registerFlag("retell-model", { type: "string", description: "复述模型（provider/model 或模型别名）" });
  pi.registerFlag("retell-auto", { type: "boolean", default: false, description: "新会话默认开启自动复述" });
  pi.registerFlag("retell-no-context", { type: "boolean", default: false, description: "新会话禁止复述模型读取上下文" });
  pi.registerFlag("retell-prompt", { type: "string", description: "新会话的自定义复述风格提示词" });

  let settings: Settings = { enabled: false, model: "", context: true, prompt: "" };
  let pending: RetellJob | undefined;
  const attempted = new Set<string>();

  function cancel() {
    const job = pending;
    pending = undefined;
    job?.controller.abort();
    job?.clearProgress?.();
  }

  function save() {
    pi.appendEntry(SETTINGS_TYPE, { ...settings });
  }

  function restore(ctx: ExtensionContext) {
    cancel();
    attempted.clear();
    settings = {
      enabled: pi.getFlag("retell-auto") === true,
      model: String(pi.getFlag("retell-model") || ""),
      context: pi.getFlag("retell-no-context") !== true,
      prompt: String(pi.getFlag("retell-prompt") || "").trim(),
    };
    for (const entry of ctx.sessionManager.getBranch()) {
      if (entry.type !== "custom" || entry.customType !== SETTINGS_TYPE) continue;
      const data = entry.data as Partial<Settings> | undefined;
      if (data && typeof data.enabled === "boolean" && typeof data.model === "string" && typeof data.context === "boolean") {
        settings = {
          enabled: data.enabled,
          model: data.model,
          context: data.context,
          prompt: typeof data.prompt === "string" ? data.prompt : settings.prompt,
        };
      }
    }
    if ((ctx.mode === "print" || ctx.mode === "json") && settings.enabled) {
      console.error("[reteller] 自动追加复述需要交互 TUI 或持续运行的 RPC；单次 --print 模式不运行复述。");
    }
  }

  async function chooseModel(spec: string, ctx: ExtensionCommandContext): Promise<boolean> {
    let selected = spec.trim();
    if (!selected) {
      if (!ctx.hasUI) {
        ctx.ui.notify("请使用 /retell model provider/model 指定模型。", "warning");
        return false;
      }
      const models = ctx.models.list().map(model => `${model.provider}/${model.id}`);
      if (!models.length) {
        ctx.ui.notify("没有已认证的可用模型，请先配置 omp 模型。", "error");
        return false;
      }
      selected = await ctx.ui.select("选择复述模型（不会切换主模型）", models) || "";
      if (!selected) return false;
    }
    const model = ctx.models.resolve(selected);
    if (!model) {
      ctx.ui.notify(`找不到复述模型：${selected}`, "error");
      return false;
    }
    cancel();
    settings.model = `${model.provider}/${model.id}`;
    save();
    ctx.ui.notify(`复述模型：${settings.model}`, "info");
    return true;
  }

  function start(ctx: ExtensionContext, entries: readonly SessionEntry[], source: Source, manual = false) {
    if (pending) {
      ctx.ui.notify("已有复述正在进行；可先 /retell cancel。", "warning");
      return;
    }
    const model = settings.model ? ctx.models.resolve(settings.model) : undefined;
    if (!model) {
      ctx.ui.notify("请先 /retell model 选择可用的复述模型。", "warning");
      return;
    }
    const job: RetellJob = { controller: new AbortController(), sourceId: source.id };
    pending = job;
    if (manual && ctx.hasUI) {
      const message = "reteller 正在复述… · /retell cancel 可取消";
      ctx.ui.setWidget(PROGRESS_WIDGET, ctx.mode === "tui"
        ? (tui, theme) => new Loader(tui, text => theme.fg("accent", text), text => theme.fg("muted", text), message)
        : [message]);
      job.clearProgress = () => ctx.ui.setWidget(PROGRESS_WIDGET, undefined);
    }
    const sessionId = ctx.sessionManager.getSessionId();
    const contextEnabled = settings.context;
    const prompt = settings.prompt;
    let excerpts: ContextExcerpt[] | undefined;

    // agent_end is notification-only with a short host timeout. Own this side request
    // explicitly instead of making a long model call hold a lifecycle handler open.
    const run = async () => {
      try {
        const apiKey = await ctx.modelRegistry.getApiKey(model, crypto.randomUUID(), { signal: job.controller.signal });
        if (apiKey === undefined) throw new Error(`复述模型 ${model.provider}/${model.id} 没有可用凭据。`);
        const result = await retell(source.text, {
          model,
          apiKey,
          signal: job.controller.signal,
          prompt,
          ...(contextEnabled ? {
            context: () => excerpts ??= projectContext(entries.slice(0, source.index).flatMap(entry =>
              entry.type === "message" ? [entry.message] : [])),
          } : {}),
        });
        if (job.controller.signal.aborted || pending !== job) return;
        if (ctx.sessionManager.getSessionId() !== sessionId || !ctx.isIdle() || ctx.hasPendingMessages()) return;
        if (latestSource(ctx.sessionManager.getBranch())?.id !== source.id) return;
        pi.sendMessage({
          customType: RESULT_TYPE,
          content: `## 复述版\n\n${result.text}\n\n---\n复述模型：\`${model.provider}/${model.id}\` · 上下文读取 ${result.contextReads} 次`,
          display: true,
          attribution: "agent",
          details: { sourceEntryId: source.id, model: `${model.provider}/${model.id}`, contextReads: result.contextReads },
        }, { triggerTurn: false });
      } catch (error) {
        if (!job.controller.signal.aborted && pending === job) {
          ctx.ui.notify(`复述失败，原回复未改动：${error instanceof Error ? error.message : String(error)}`, "error");
        }
      } finally {
        if (pending === job) {
          pending = undefined;
          job.clearProgress?.();
        }
      }
    };
    void run().catch(error => pi.logger.error("Reteller side request failed", { error: String(error) }));
  }

  pi.registerMessageRenderer(RESULT_TYPE, message =>
    new Markdown(textOnly(message.content), 0, 0, getMarkdownTheme()));

  // Persist/display the retelling, but never feed it back to the main model.
  pi.on("context", event => ({
    messages: event.messages.filter(message => message.role !== "custom" || message.customType !== RESULT_TYPE),
  }));

  pi.on("session_start", (_event, ctx) => { if (ctx.agent.kind === "main") restore(ctx); });
  pi.on("session_switch", (_event, ctx) => { if (ctx.agent.kind === "main") restore(ctx); });
  pi.on("session_branch", (_event, ctx) => { if (ctx.agent.kind === "main") restore(ctx); });
  pi.on("session_tree", (_event, ctx) => { if (ctx.agent.kind === "main") restore(ctx); });
  pi.on("session_before_switch", cancel);
  pi.on("session_before_branch", cancel);
  pi.on("session_before_tree", cancel);
  pi.on("session_shutdown", cancel);
  pi.on("before_agent_start", cancel);
  pi.on("input", event => {
    if (!/^\/retell(?:\s|$)/.test(event.text.trim())) cancel();
  });

  pi.on("agent_end", (event, ctx) => {
    if (ctx.agent.kind !== "main" || !settings.enabled || event.willContinue || ctx.hasPendingMessages()) return;
    if (ctx.mode === "print" || ctx.mode === "json") return;
    const entries = ctx.sessionManager.getBranch();
    const source = latestSource(entries);
    if (!source || pending || attempted.has(source.id)) return;
    const alreadyDisplayed = entries.some(entry => entry.type === "custom_message" && entry.customType === RESULT_TYPE
      && (entry.details as { sourceEntryId?: string } | undefined)?.sourceEntryId === source.id);
    if (alreadyDisplayed) return;
    attempted.add(source.id);
    start(ctx, entries, source);
  });

  pi.registerCommand("retell", {
    description: "用独立模型复述最终正文；on/off、model、prompt、context、cancel、status",
    getArgumentCompletions: prefix => {
      const values = ["on", "off", "model", "prompt", "context on", "context off", "cancel", "status", "help"];
      const matches = values.filter(value => value.startsWith(prefix)).map(value => ({ value, label: value }));
      return matches.length ? matches : null;
    },
    handler: async (args, ctx) => {
      if (ctx.agent.kind !== "main") {
        ctx.ui.notify("Reteller 只处理主 agent 的最终回复。", "warning");
        return;
      }
      const input = args.trim();
      const [command = ""] = input.split(/\s+/, 1);
      const value = input.slice(command.length).trim();
      switch (command) {
        case "": {
          if (!ctx.isIdle()) {
            ctx.ui.notify("请等待主 agent 完成回复后再 /retell。", "warning");
            return;
          }
          if (pending) {
            ctx.ui.notify("已有复述正在进行；可先 /retell cancel。", "warning");
            return;
          }
          const entries = ctx.sessionManager.getBranch();
          const source = latestSource(entries);
          if (!source) {
            ctx.ui.notify("当前没有已完整结束、可复述的主 agent 回复。", "warning");
            return;
          }
          if (!settings.model && !await chooseModel("", ctx)) return;
          start(ctx, entries, source, true);
          return;
        }
        case "prompt":
          cancel();
          settings.prompt = value;
          save();
          ctx.ui.notify(settings.prompt
            ? "已更新复述提示词；对后续手动和自动复述生效。"
            : "已恢复默认复述提示词。", "info");
          return;
        case "model":
          await chooseModel(value, ctx);
          return;
        case "on":
          if (value) break;
          if (!settings.model && !await chooseModel("", ctx)) return;
          if (!ctx.models.resolve(settings.model)) {
            ctx.ui.notify("配置的复述模型不可用，请重新 /retell model。", "error");
            return;
          }
          settings.enabled = true;
          save();
          ctx.ui.notify("已开启自动复述；从下一次完整回复开始生效。", "info");
          return;
        case "off":
          if (value) break;
          settings.enabled = false;
          cancel();
          save();
          ctx.ui.notify("已关闭自动复述；仍可 /retell 手动复述。", "info");
          return;
        case "context":
          if (value !== "on" && value !== "off") break;
          cancel();
          settings.context = value === "on";
          save();
          ctx.ui.notify(settings.context
            ? "上下文工具已开启：仅按需发送历史用户正文、助手正文或工具结果片段。"
            : "上下文工具已关闭：复述模型只接收最终回复正文与编辑提示词。", "info");
          return;
        case "cancel":
          if (value) break;
          cancel();
          ctx.ui.notify("已取消当前复述。", "info");
          return;
        case "status":
          if (value) break;
          ctx.ui.notify(`自动复述：${settings.enabled ? "开" : "关"}\n模型：${settings.model || "未选择"}\n上下文工具：${settings.context ? "开（仅按需）" : "关"}\n复述提示词：${settings.prompt || "默认"}\n进行中：${pending ? "是" : "否"}`, "info");
          return;
        case "help":
          if (value) break;
          ctx.ui.notify(HELP, "info");
          return;
      }
      ctx.ui.notify(HELP, "warning");
    },
  });
}
