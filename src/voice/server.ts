import { createServer, type Server, type IncomingMessage, type ServerResponse } from "http";
import { WebSocketServer, WebSocket as WS } from "ws";
import { query } from "../sdk.js";
import type { VoiceServerOptions, ClientMessage, ServerMessage, MultimodalAdapter } from "./adapter.js";
import { readFileSync, readdirSync, statSync, existsSync, writeFileSync, mkdirSync, appendFileSync } from "fs";
import { execSync } from "child_process";
import { join, dirname, resolve, relative } from "path";
import { writeFile, readFile, mkdir, stat } from "fs/promises";
import { fileURLToPath } from "url";
import { OpenAIRealtimeAdapter } from "./openai-realtime.js";
import { GeminiLiveAdapter } from "./gemini-live.js";
import { ComposioAdapter } from "../composio/index.js";
import type { GCToolDefinition } from "../sdk-types.js";
import { appendMessage, loadHistory, deleteHistory, summarizeHistory } from "./chat-history.js";
import { getVoiceContext, getAgentContext } from "../context.js";

const dim = (s: string) => `\x1b[2m${s}\x1b[0m`;
const bold = (s: string) => `\x1b[1m${s}\x1b[0m`;

// ── Background memory saver ────────────────────────────────────────────
// Patterns that indicate the user is sharing personal info worth saving.
// This runs server-side so we don't depend on the voice LLM deciding to save.
const MEMORY_PATTERNS = [
	/\bi (?:like|love|enjoy|prefer|hate|dislike)\b/i,
	/\bmy (?:name|dog|cat|favorite|fav|hobby|job|car|team)\b/i,
	/\bi(?:'m| am) (?:a |into |from |working on )/i,
	/\bcall me\b/i,
	/\bremember (?:that|this)\b/i,
	/\bi (?:play|watch|drive|use|work with|listen to)\b/i,
];

function isMemoryWorthy(text: string): boolean {
	return MEMORY_PATTERNS.some((p) => p.test(text));
}

// ── Moment detection for photo capture ─────────────────────────────────
const MOMENT_PATTERNS = [
	/\bhaha\b/i,
	/\blol\b/i,
	/\blmao\b/i,
	/\blove it\b/i,
	/\bthat'?s amazing\b/i,
	/\bso happy\b/i,
	/\bbest day\b/i,
	/\bwe did it\b/i,
	/\bnailed it\b/i,
	/\blet'?s go\b/i,
	/\bhell yeah\b/i,
	/\bawesome\b/i,
	/\bthank you so much\b/i,
	/\bfirst time\b/i,
	/\bmilestone\b/i,
	/\bcelebrat/i,
	/\bincredible\b/i,
];

function isMomentWorthy(text: string): boolean {
	return MOMENT_PATTERNS.some((p) => p.test(text));
}

const PHOTOS_DIR = "memory/photos";
const INDEX_FILE = "memory/photos/INDEX.md";
const LATEST_FRAME_FILE = "memory/.latest-frame.jpg";
const LATEST_SCREEN_FILE = "memory/.latest-screen.jpg";

function slugify(text: string): string {
	return text
		.toLowerCase()
		.replace(/[^a-z0-9]+/g, "-")
		.replace(/^-|-$/g, "")
		.slice(0, 40);
}

// ── Mood tracking ──────────────────────────────────────────────────────
type Mood = "happy" | "frustrated" | "curious" | "excited" | "calm";
const MOOD_SIGNALS: { mood: Mood; patterns: RegExp[] }[] = [
	{ mood: "happy", patterns: [/\bhaha\b/i, /\blol\b/i, /\blove it\b/i, /\bthat'?s great\b/i, /\bnice\b/i, /\bawesome\b/i, /\bamazing\b/i] },
	{ mood: "frustrated", patterns: [/\bugh\b/i, /\bwhat the\b/i, /\bdamn\b/i, /\bstill broken\b/i, /\bnot working\b/i, /\bwhy (?:is|does|won'?t)\b/i, /\bfuck\b/i] },
	{ mood: "curious", patterns: [/\bhow (?:do|does|can|would)\b/i, /\bwhat (?:is|are|if)\b/i, /\bwhy (?:do|does|is)\b/i, /\bexplain\b/i, /\btell me about\b/i] },
	{ mood: "excited", patterns: [/\blet'?s go\b/i, /\bhell yeah\b/i, /\bwe did it\b/i, /\bnailed it\b/i, /\byes!\b/i, /\bfinally\b/i] },
	{ mood: "calm", patterns: [/\bokay\b/i, /\bsure\b/i, /\bcool\b/i, /\bsounds good\b/i, /\bgot it\b/i] },
];

function detectMood(text: string): Mood | null {
	for (const { mood, patterns } of MOOD_SIGNALS) {
		if (patterns.some((p) => p.test(text))) return mood;
	}
	return null;
}

interface MoodCounts { happy: number; frustrated: number; curious: number; excited: number; calm: number }

function dominantMood(counts: MoodCounts): Mood {
	let best: Mood = "calm";
	let max = 0;
	for (const [mood, count] of Object.entries(counts) as [Mood, number][]) {
		if (count > max) { max = count; best = mood; }
	}
	return best;
}

async function saveMoodEntry(agentDir: string, counts: MoodCounts, messageCount: number): Promise<void> {
	if (messageCount < 3) return; // Skip trivially short sessions

	const now = new Date();
	const pad = (n: number) => String(n).padStart(2, "0");
	const date = `${now.getFullYear()}-${pad(now.getMonth() + 1)}-${pad(now.getDate())}`;
	const time = `${pad(now.getHours())}:${pad(now.getMinutes())}`;
	const mood = dominantMood(counts);

	const moodPath = join(agentDir, "memory", "mood.md");
	let existing = "";
	try { existing = await readFile(moodPath, "utf-8"); } catch {
		existing = "# Mood Log\n\n";
	}

	const detail = Object.entries(counts).filter(([, v]) => v > 0).map(([k, v]) => `${k}:${v}`).join(" ");
	existing += `- ${date} ${time} — **${mood}** (${detail}) [${messageCount} msgs]\n`;

	await mkdir(join(agentDir, "memory"), { recursive: true });
	await writeFile(moodPath, existing, "utf-8");

	try {
		execSync(`git add "memory/mood.md" && git commit -m "Mood: ${mood} session (${date} ${time})"`, {
			cwd: agentDir, stdio: "pipe",
		});
	} catch { /* file saved even if commit fails */ }
}

// ── Session journaling ─────────────────────────────────────────────────
async function writeJournalEntry(
	agentDir: string,
	branch: string,
	moodCounts: MoodCounts,
	model?: string,
	env?: string,
): Promise<void> {
	const messages = loadHistory(agentDir, branch);
	if (messages.length < 5) return;

	const lines: string[] = [];
	for (const msg of messages.slice(-50)) {
		if (msg.type === "transcript") lines.push(`${msg.role}: ${msg.text}`);
		else if (msg.type === "agent_done") lines.push(`agent: ${msg.result.slice(0, 200)}`);
	}
	if (lines.length < 3) return;

	let transcript = lines.join("\n");
	if (transcript.length > 3000) transcript = transcript.slice(-3000);

	const mood = dominantMood(moodCounts);
	const prompt = `Write a brief journal entry (3-5 sentences) reflecting on this conversation session. Mood was mostly: ${mood}. Note what was accomplished, any unfinished threads, and how the user seemed. Write in first person as the agent. Be genuine, not corporate.\n\nTranscript:\n${transcript}`;

	try {
		const result = query({
			prompt,
			dir: agentDir,
			model,
			env,
			maxTurns: 1,
			replaceBuiltinTools: true,
			tools: [],
			systemPrompt: "You are journaling about your day as an AI assistant. Write naturally and briefly.",
		});

		let entry = "";
		for await (const msg of result) {
			if (msg.type === "assistant" && msg.content) entry += msg.content;
		}
		entry = entry.trim();
		if (!entry) return;

		const now = new Date();
		const pad = (n: number) => String(n).padStart(2, "0");
		const date = `${now.getFullYear()}-${pad(now.getMonth() + 1)}-${pad(now.getDate())}`;
		const time = `${pad(now.getHours())}:${pad(now.getMinutes())}`;

		const journalDir = join(agentDir, "memory", "journal");
		await mkdir(journalDir, { recursive: true });
		const journalPath = join(journalDir, `${date}.md`);

		let existing = "";
		try { existing = await readFile(journalPath, "utf-8"); } catch {
			existing = `# Journal — ${date}\n\n`;
		}
		existing += `### ${time} (${mood})\n${entry}\n\n`;
		await writeFile(journalPath, existing, "utf-8");

		try {
			execSync(`git add "memory/journal/${date}.md" && git commit -m "Journal: ${date} ${time} session reflection"`, {
				cwd: agentDir, stdio: "pipe",
			});
			console.error(dim(`[voice] Journal entry written for ${date} ${time}`));
		} catch { /* saved even if commit fails */ }
	} catch (err: any) {
		console.error(dim(`[voice] Journal write failed: ${err.message}`));
	}
}

async function capturePhoto(
	agentDir: string,
	reason: string,
	frameData?: Buffer,
): Promise<void> {
	// If no frame passed directly, read from temp file
	let frame = frameData;
	if (!frame) {
		const framePath = join(agentDir, LATEST_FRAME_FILE);
		try {
			const frameStat = await stat(framePath);
			if (Date.now() - frameStat.mtimeMs > 5000) {
				console.error(dim("[voice] No recent camera frame, skipping photo capture"));
				return;
			}
			frame = await readFile(framePath);
		} catch {
			console.error(dim("[voice] No camera frame available, skipping photo capture"));
			return;
		}
	}

	const now = new Date();
	const pad = (n: number) => String(n).padStart(2, "0");
	const datePart = `${now.getFullYear()}-${pad(now.getMonth() + 1)}-${pad(now.getDate())}`;
	const timePart = `${pad(now.getHours())}${pad(now.getMinutes())}${pad(now.getSeconds())}`;
	const slug = slugify(reason);
	const filename = `${datePart}_${timePart}_${slug}.jpg`;
	const photoRelPath = `${PHOTOS_DIR}/${filename}`;
	const photoAbsPath = join(agentDir, photoRelPath);

	await mkdir(join(agentDir, PHOTOS_DIR), { recursive: true });
	await writeFile(photoAbsPath, frame);

	// Update INDEX.md
	const indexPath = join(agentDir, INDEX_FILE);
	let indexContent = "";
	try {
		indexContent = await readFile(indexPath, "utf-8");
	} catch {
		indexContent = "# Memorable Moments\n\nPhotos captured during happy and memorable moments.\n\n";
	}
	const entry = `- **${datePart} ${pad(now.getHours())}:${pad(now.getMinutes())}:${pad(now.getSeconds())}** — ${reason} → [\`${filename}\`](${filename})\n`;
	indexContent += entry;
	await writeFile(indexPath, indexContent, "utf-8");

	// Git add + commit
	const commitMsg = `Capture moment: ${reason}`;
	try {
		execSync(`git add "${photoRelPath}" "${INDEX_FILE}" && git commit -m "${commitMsg.replace(/"/g, '\\"')}"`, {
			cwd: agentDir,
			stdio: "pipe",
		});
		console.error(dim(`[voice] Photo captured: ${filename}`));
	} catch (err: any) {
		console.error(dim(`[voice] Photo saved but git commit failed: ${err.stderr?.toString().trim() || "unknown"}`));
	}
}

function saveMemoryInBackground(
	text: string,
	agentDir: string,
	model?: string,
	env?: string,
	onComplete?: () => void,
): void {
	const prompt = `The user just said: "${text}"\n\nSave any personal information, preferences, or facts about the user to memory. Use the memory tool to write or update a memory file. Use a descriptive commit message like "Remember: user likes mustangs" or "Save preference: favorite game is GTA 5". Be concise. If there's nothing meaningful to save, do nothing.`;
	console.error(dim(`[voice] Background memory save triggered for: "${text.slice(0, 60)}..."`));

	// Fire and forget — don't block the voice conversation
	(async () => {
		try {
			const result = query({
				prompt,
				dir: agentDir,
				model,
				env,
				maxTurns: 3,
			});
			// Drain the iterator to completion
			for await (const msg of result) {
				if (msg.type === "tool_use") {
					console.error(dim(`[voice/memory] Tool: ${msg.toolName}`));
				}
			}
			console.error(dim("[voice/memory] Background save complete"));
			if (onComplete) onComplete();
		} catch (err: any) {
			console.error(dim(`[voice/memory] Background save failed: ${err.message}`));
		}
	})();
}

/** Load .env file into process.env (won't overwrite existing vars) */
function loadEnvFile(dir: string) {
	const envPath = join(dir, ".env");
	try {
		const content = readFileSync(envPath, "utf-8");
		for (const line of content.split("\n")) {
			const trimmed = line.trim();
			if (!trimmed || trimmed.startsWith("#")) continue;
			const eq = trimmed.indexOf("=");
			if (eq < 1) continue;
			const key = trimmed.slice(0, eq).trim();
			let val = trimmed.slice(eq + 1).trim();
			// Strip surrounding quotes
			if ((val.startsWith('"') && val.endsWith('"')) || (val.startsWith("'") && val.endsWith("'"))) {
				val = val.slice(1, -1);
			}
			if (!process.env[key]) {
				process.env[key] = val;
			}
		}
	} catch {
		// No .env file — that's fine
	}
}

function createAdapter(opts: VoiceServerOptions): MultimodalAdapter {
	switch (opts.adapter) {
		case "openai-realtime":
			return new OpenAIRealtimeAdapter(opts.adapterConfig);
		case "gemini-live":
			return new GeminiLiveAdapter(opts.adapterConfig);
		default:
			throw new Error(`Unknown adapter: ${opts.adapter}`);
	}
}

function loadUIHtml(): string {
	// Try dist/voice/ui.html first (built), then src/voice/ui.html (dev)
	const thisDir = dirname(fileURLToPath(import.meta.url));
	const candidates = [
		join(thisDir, "ui.html"),
		join(thisDir, "..", "..", "src", "voice", "ui.html"),
	];
	for (const path of candidates) {
		try {
			return readFileSync(path, "utf-8");
		} catch {
			// try next
		}
	}
	return "<html><body><h1>UI not found</h1><p>Run: npm run build</p></body></html>";
}

export async function startVoiceServer(opts: VoiceServerOptions): Promise<() => Promise<void>> {
	// Load .env from agent directory (won't overwrite existing env vars)
	loadEnvFile(resolve(opts.agentDir));

	const port = opts.port || 3333;
	let agentName = "GitClaw";
	try {
		const yamlRaw = readFileSync(join(resolve(opts.agentDir), "agent.yaml"), "utf-8");
		const m = yamlRaw.match(/^name:\s*(.+)$/m);
		if (m) agentName = m[1].trim();
	} catch { /* fallback to default */ }
	const uiHtml = loadUIHtml().replace(/\{\{AGENT_NAME\}\}/g, agentName);

	// Creates a per-connection tool handler that can stream events to the browser
	function createToolHandler(sendToBrowser: (msg: ServerMessage) => void) {
		return async (prompt: string): Promise<string> => {
			let composioTools: GCToolDefinition[] = [];
			let connectedSlugs: string[] = [];
			if (composioAdapter) {
				try {
					connectedSlugs = await composioAdapter.getConnectedToolkitSlugs();
					console.error(`[voice] Connected toolkit slugs: [${connectedSlugs.join(", ")}]`);
					if (connectedSlugs.length > 0) {
						// Try semantic search first, fall back to all connected tools
						composioTools = await composioAdapter.getToolsForQuery(prompt);
						console.error(`[voice] Semantic search returned ${composioTools.length} tools`);
						if (composioTools.length === 0) {
							composioTools = await composioAdapter.getTools();
							console.error(`[voice] Fallback getTools returned ${composioTools.length} tools`);
						}
						console.error(`[voice] Composio: ${composioTools.length} tools: ${composioTools.map(t => t.name).join(", ")}`);
					} else {
						console.error(`[voice] No connected toolkits found for user`);
					}
				} catch (err: any) {
					console.error(`[voice] Composio tool fetch FAILED: ${err.message}\n${err.stack}`);
				}
			} else {
				console.error(`[voice] composioAdapter is NULL — COMPOSIO_API_KEY not set?`);
			}

			// Build system prompt suffix: always tell the agent about Composio capabilities
			let systemPromptSuffix: string | undefined;
			if (composioAdapter) {
				const parts = [
					`You have access to external services via Composio integration (Gmail, Google Calendar, GitHub, Slack, and many more).`,
					`You CAN perform real actions — send emails, read emails, check calendars, create events, manage repos, etc.`,
					`NEVER tell the user you "can't access" or "don't have access to" external services. Always attempt to use the available Composio tools (prefixed "composio_") first.`,
					`When the user asks to send an email, use the composio SEND_EMAIL tool directly — do NOT create a draft unless they explicitly ask for a draft.`,
					`When the user asks about their calendar, use the composio calendar tools to fetch real events.`,
					`Prefer Composio tools over CLI commands for any external service interaction.`,
				];
				if (connectedSlugs.length > 0) {
					const services = connectedSlugs.map((s) => s.replace(/_/g, " ")).join(", ");
					parts.unshift(`Currently connected services: ${services}.`);
				}
				systemPromptSuffix = parts.join(" ");
			}

			// Inject shared context (memory + conversation summary)
			const agentContext = await getAgentContext(opts.agentDir, activeBranch);
			if (agentContext) {
				systemPromptSuffix = (systemPromptSuffix || "") + "\n\n" + agentContext;
			}

			const result = query({
				prompt,
				dir: opts.agentDir,
				model: opts.model,
				env: opts.env,
				...(composioTools.length ? { tools: composioTools } : {}),
				...(systemPromptSuffix ? { systemPromptSuffix } : {}),
			});

			let text = "";
			const toolResults: string[] = [];
			const errors: string[] = [];

			for await (const msg of result) {
				if (msg.type === "assistant" && msg.content) {
					text += msg.content;
				} else if (msg.type === "tool_use") {
					sendToBrowser({ type: "tool_call", toolName: msg.toolName, args: msg.args });
					console.log(dim(`[voice] Tool call: ${msg.toolName}(${JSON.stringify(msg.args).slice(0, 80)})`));
				} else if (msg.type === "tool_result") {
					sendToBrowser({ type: "tool_result", toolName: msg.toolName, content: msg.content, isError: msg.isError });
					if (msg.content) toolResults.push(msg.content);
					console.log(dim(`[voice] Tool ${msg.toolName}: ${msg.content.slice(0, 100)}${msg.content.length > 100 ? "..." : ""}`));
				} else if (msg.type === "system" && msg.subtype === "error") {
					errors.push(msg.content);
					console.error(dim(`[voice] Agent error: ${msg.content}`));
				} else if (msg.type === "delta" && msg.deltaType === "thinking") {
					sendToBrowser({ type: "agent_thinking", text: msg.content });
				}
			}

			if (text) return text;
			if (errors.length > 0) return `Error: ${errors.join("; ")}`;
			if (toolResults.length > 0) return toolResults.join("\n");
			return "(no response)";
		};
	}

	// ── File API helpers ────────────────────────────────────────────────
	const HIDDEN_DIRS = new Set([".git", "node_modules", ".gitagent", "dist", ".next", "__pycache__", ".venv"]);
	const agentRoot = resolve(opts.agentDir);
	let activeBranch = execSync("git rev-parse --abbrev-ref HEAD", { cwd: agentRoot, encoding: "utf-8" }).trim();
	const pendingShutdownWork: Promise<any>[] = [];

	// ── Composio integration (optional) ────────────────────────────────
	let composioAdapter: ComposioAdapter | null = null;
	if (process.env.COMPOSIO_API_KEY) {
		composioAdapter = new ComposioAdapter({
			apiKey: process.env.COMPOSIO_API_KEY,
			userId: process.env.COMPOSIO_USER_ID || "default",
		});
		console.log(dim("[voice] Composio integration enabled"));
	}

	/** Resolve and validate a requested path stays within agentDir */
	function safePath(reqPath: string): string | null {
		const abs = resolve(agentRoot, reqPath);
		if (!abs.startsWith(agentRoot)) return null;
		return abs;
	}

	interface FileEntry {
		name: string;
		path: string;
		type: "file" | "directory";
		mtime?: number;
		children?: FileEntry[];
	}

	function listDir(dirPath: string, depth: number): FileEntry[] {
		if (depth > 4) return [];
		try {
			const entries = readdirSync(dirPath);
			const result: FileEntry[] = [];
			for (const name of entries) {
				if (name.startsWith(".") && HIDDEN_DIRS.has(name)) continue;
				if (HIDDEN_DIRS.has(name)) continue;
				const fullPath = join(dirPath, name);
				const relPath = relative(agentRoot, fullPath);
				try {
					const st = statSync(fullPath);
					if (st.isDirectory()) {
						result.push({
							name,
							path: relPath,
							type: "directory",
							children: listDir(fullPath, depth + 1),
						});
					} else if (st.isFile()) {
						result.push({ name, path: relPath, type: "file", mtime: st.mtimeMs });
					}
				} catch {
					// skip unreadable entries
				}
			}
			// Sort: directories first, then alphabetical
			result.sort((a, b) => {
				if (a.type !== b.type) return a.type === "directory" ? -1 : 1;
				return a.name.localeCompare(b.name);
			});
			return result;
		} catch {
			return [];
		}
	}

	function readBody(req: IncomingMessage): Promise<string> {
		return new Promise((res, rej) => {
			let body = "";
			req.on("data", (c: Buffer) => { body += c.toString(); });
			req.on("end", () => res(body));
			req.on("error", rej);
		});
	}

	function jsonReply(res: ServerResponse, status: number, data: any) {
		res.writeHead(status, { "Content-Type": "application/json" });
		res.end(JSON.stringify(data));
	}

	// HTTP server
	const httpServer: Server = createServer(async (req, res) => {
		res.setHeader("Access-Control-Allow-Origin", "*");
		res.setHeader("Access-Control-Allow-Methods", "GET, POST, PUT, DELETE, OPTIONS");
		res.setHeader("Access-Control-Allow-Headers", "Content-Type");

		if (req.method === "OPTIONS") {
			res.writeHead(204);
			return res.end();
		}

		const url = new URL(req.url || "/", `http://localhost:${port}`);

		if (url.pathname === "/health") {
			jsonReply(res, 200, { status: "ok" });

		} else if (url.pathname === "/" || url.pathname === "/test") {
			res.writeHead(200, { "Content-Type": "text/html" });
			res.end(uiHtml);

		} else if (url.pathname === "/api/files" && req.method === "GET") {
			// List files as a tree
			const reqPath = url.searchParams.get("path") || ".";
			const abs = safePath(reqPath);
			if (!abs) return jsonReply(res, 403, { error: "Path outside workspace" });
			const tree = listDir(abs, 0);
			jsonReply(res, 200, { root: relative(agentRoot, abs) || ".", entries: tree });

		} else if (url.pathname === "/api/file" && req.method === "GET") {
			// Read a file
			const reqPath = url.searchParams.get("path");
			if (!reqPath) return jsonReply(res, 400, { error: "Missing path param" });
			const abs = safePath(reqPath);
			if (!abs) return jsonReply(res, 403, { error: "Path outside workspace" });
			if (!existsSync(abs)) return jsonReply(res, 404, { error: "File not found" });
			try {
				const st = statSync(abs);
				if (st.size > 1024 * 1024) return jsonReply(res, 413, { error: "File too large (>1MB)" });
				const content = readFileSync(abs, "utf-8");
				jsonReply(res, 200, { path: reqPath, content });
			} catch (err: any) {
				jsonReply(res, 500, { error: err.message });
			}

		} else if (url.pathname === "/api/file/raw" && req.method === "GET") {
			// Serve raw file with correct MIME type (for images, etc.)
			const reqPath = url.searchParams.get("path");
			if (!reqPath) return jsonReply(res, 400, { error: "Missing path param" });
			const abs = safePath(reqPath);
			if (!abs) return jsonReply(res, 403, { error: "Path outside workspace" });
			if (!existsSync(abs)) return jsonReply(res, 404, { error: "File not found" });
			try {
				const st = statSync(abs);
				if (st.size > 10 * 1024 * 1024) return jsonReply(res, 413, { error: "File too large (>10MB)" });
				const ext = reqPath.split(".").pop()?.toLowerCase() || "";
				const mimeMap: Record<string, string> = {
					png: "image/png", jpg: "image/jpeg", jpeg: "image/jpeg", gif: "image/gif",
					webp: "image/webp", svg: "image/svg+xml", ico: "image/x-icon", bmp: "image/bmp",
				};
				const mime = mimeMap[ext] || "application/octet-stream";
				const data = readFileSync(abs);
				res.writeHead(200, { "Content-Type": mime, "Content-Length": data.length, "Cache-Control": "no-cache" });
				res.end(data);
			} catch (err: any) {
				jsonReply(res, 500, { error: err.message });
			}

		} else if (url.pathname === "/api/file" && req.method === "PUT") {
			// Write a file
			const body = await readBody(req);
			let parsed: { path: string; content: string };
			try {
				parsed = JSON.parse(body);
			} catch {
				return jsonReply(res, 400, { error: "Invalid JSON body" });
			}
			if (!parsed.path || parsed.content === undefined) return jsonReply(res, 400, { error: "Missing path or content" });
			const abs = safePath(parsed.path);
			if (!abs) return jsonReply(res, 403, { error: "Path outside workspace" });
			try {
				writeFileSync(abs, parsed.content, "utf-8");
				jsonReply(res, 200, { ok: true, path: parsed.path });
			} catch (err: any) {
				jsonReply(res, 500, { error: err.message });
			}

		// ── Composio OAuth callback ─────────────────────────────────────
		} else if (url.pathname === "/api/composio/callback") {
			// OAuth popup lands here after Composio processes the auth code.
			// Send a message to the opener window and close the popup.
			res.writeHead(200, { "Content-Type": "text/html" });
			res.end(`<!DOCTYPE html><html><body><script>
				if(window.opener){window.opener.postMessage({type:'composio_auth_complete'},'*');}
				window.close();
				</script><p>Authentication complete. You can close this window.</p></body></html>`);

		// ── Chat branch API routes ──────────────────────────────────────
		} else if (url.pathname === "/api/chat/list" && req.method === "GET") {
			try {
				const git = (cmd: string) => execSync(cmd, { cwd: agentRoot, encoding: "utf-8" }).trim();
				const current = git("git rev-parse --abbrev-ref HEAD");
				// List branches matching chat/* pattern, plus the current branch
				let branches: string[];
				try {
					branches = git("git branch --list 'chat/*' --sort=-committerdate --format='%(refname:short)|%(committerdate:relative)'")
						.split("\n").filter(Boolean);
				} catch {
					branches = [];
				}
				const chats = branches.map((line) => {
					const [branch, time] = line.split("|");
					const name = branch.replace("chat/", "");
					return { branch, name, time: time || "" };
				});
				// If current branch is not a chat/* branch, add it at the top
				if (!current.startsWith("chat/")) {
					chats.unshift({ branch: current, name: current, time: "current" });
				}
				jsonReply(res, 200, { current, chats });
			} catch (err: any) {
				jsonReply(res, 500, { error: err.message });
			}

		} else if (url.pathname === "/api/chat/new" && req.method === "POST") {
			try {
				const git = (cmd: string) => execSync(cmd, { cwd: agentRoot, encoding: "utf-8" }).trim();
				// Generate branch name: chat/YYYY-MM-DD-HHMMSS
				const now = new Date();
				const pad = (n: number) => String(n).padStart(2, "0");
				const branch = `chat/${now.getFullYear()}${pad(now.getMonth()+1)}${pad(now.getDate())}-${pad(now.getHours())}${pad(now.getMinutes())}${pad(now.getSeconds())}`;
				// Stage and commit any pending changes on current branch
				try {
					git("git add -A");
					git('git commit -m "auto-save before new chat" --allow-empty');
				} catch {
					// No changes to commit, that's fine
				}
				// Create and switch to new branch
				git(`git checkout -b ${branch}`);
				activeBranch = branch;
				jsonReply(res, 200, { branch });
			} catch (err: any) {
				jsonReply(res, 500, { error: err.message });
			}

		} else if (url.pathname === "/api/chat/switch" && req.method === "POST") {
			try {
				const body = await readBody(req);
				const { branch } = JSON.parse(body);
				if (!branch) return jsonReply(res, 400, { error: "Missing branch" });
				const git = (cmd: string) => execSync(cmd, { cwd: agentRoot, encoding: "utf-8" }).trim();
				// Auto-save current branch
				try {
					git("git add -A");
					git('git commit -m "auto-save before switching chat" --allow-empty');
				} catch {}
				git(`git checkout ${branch}`);
				activeBranch = branch;
				jsonReply(res, 200, { branch });
			} catch (err: any) {
				jsonReply(res, 500, { error: err.message });
			}

		} else if (url.pathname === "/api/chat/delete" && req.method === "POST") {
			try {
				const body = await readBody(req);
				const { branch } = JSON.parse(body);
				if (!branch) return jsonReply(res, 400, { error: "Missing branch" });
				const git = (cmd: string) => execSync(cmd, { cwd: agentRoot, encoding: "utf-8" }).trim();
				const current = git("git rev-parse --abbrev-ref HEAD");
				if (branch === current) return jsonReply(res, 400, { error: "Cannot delete the active branch" });
				git(`git branch -D ${branch}`);
				deleteHistory(opts.agentDir, branch);
				jsonReply(res, 200, { ok: true });
			} catch (err: any) {
				jsonReply(res, 500, { error: err.message });
			}

	} else if (url.pathname === "/api/chat/history" && req.method === "GET") {
			const branch = url.searchParams.get("branch");
			if (!branch) return jsonReply(res, 400, { error: "Missing branch param" });
			const messages = loadHistory(opts.agentDir, branch);
			jsonReply(res, 200, { branch, messages });

		// ── Composio API routes ─────────────────────────────────────────
		} else if (url.pathname === "/api/composio/toolkits" && req.method === "GET") {
			if (!composioAdapter) return jsonReply(res, 501, { error: "Composio not configured" });
			try {
				const toolkits = await composioAdapter.getToolkits();
				jsonReply(res, 200, toolkits);
			} catch (err: any) {
				jsonReply(res, 502, { error: err.message });
			}

		} else if (url.pathname === "/api/composio/connect" && req.method === "POST") {
			if (!composioAdapter) return jsonReply(res, 501, { error: "Composio not configured" });
			const body = await readBody(req);
			let parsed: { toolkit: string; redirectUrl?: string };
			try { parsed = JSON.parse(body); } catch { return jsonReply(res, 400, { error: "Invalid JSON" }); }
			if (!parsed.toolkit) return jsonReply(res, 400, { error: "Missing toolkit" });
			try {
				const result = await composioAdapter.connect(parsed.toolkit, parsed.redirectUrl);
				jsonReply(res, 200, result);
			} catch (err: any) {
				jsonReply(res, 502, { error: err.message });
			}

		} else if (url.pathname === "/api/composio/connections" && req.method === "GET") {
			if (!composioAdapter) return jsonReply(res, 501, { error: "Composio not configured" });
			try {
				const connections = await composioAdapter.getConnections();
				jsonReply(res, 200, connections);
			} catch (err: any) {
				jsonReply(res, 502, { error: err.message });
			}

		} else if (url.pathname.match(/^\/api\/composio\/connections\/[^/]+$/) && req.method === "DELETE") {
			if (!composioAdapter) return jsonReply(res, 501, { error: "Composio not configured" });
			const connId = url.pathname.split("/").pop()!;
			try {
				await composioAdapter.disconnect(connId);
				jsonReply(res, 200, { ok: true });
			} catch (err: any) {
				jsonReply(res, 502, { error: err.message });
			}

		} else {
			res.writeHead(404);
			res.end();
		}
	});

	// WebSocket server — adapter-agnostic proxy
	const wss = new WebSocketServer({ server: httpServer });

	wss.on("connection", async (browserWs: WS) => {
		console.log(dim("[voice] Browser connected"));

		// ── Per-connection frame buffer + moment capture state ──────────
		let latestVideoFrame: { frame: string; mimeType: string; ts: number } | null = null;
		let lastFrameWriteTs = 0;
		let latestScreenFrame: { frame: string; mimeType: string; ts: number } | null = null;
		let lastScreenWriteTs = 0;
		let lastMomentCaptureTs = 0;
		const FRAME_WRITE_INTERVAL = 2000; // Write temp frame to disk every 2s
		const MOMENT_COOLDOWN = 60000;     // 60s between auto-captures
		const moodCounts: MoodCounts = { happy: 0, frustrated: 0, curious: 0, excited: 0, calm: 0 };
		let sessionMessageCount = 0;

		// Inject shared context (memory + conversation summary) into voice LLM instructions
		const voiceContext = await getVoiceContext(opts.agentDir, activeBranch);
		let instructions = opts.adapterConfig.instructions || "";
		if (voiceContext) {
			instructions += "\n\n" + voiceContext;
		}

		// Inject Composio awareness into adapter instructions so the voice LLM
		// never tells the user "I can't access" external services
		const adapterOpts = composioAdapter ? {
			...opts,
			adapterConfig: {
				...opts.adapterConfig,
				instructions: instructions +
					" The agent has FULL access to external services via Composio — Gmail, Google Calendar, GitHub, Slack, and more. " +
					"When the user asks to send emails, check calendars, or interact with any external service, ALWAYS use run_agent to handle it. " +
					"NEVER say you can't access these services or that you don't have these tools. The agent has them. Just call run_agent.",
			},
		} : {
			...opts,
			adapterConfig: {
				...opts.adapterConfig,
				instructions,
			},
		};
		const adapter = createAdapter(adapterOpts);
		const sendToBrowser = (msg: ServerMessage) => {
			safeSend(browserWs, JSON.stringify(msg));
			appendMessage(opts.agentDir, activeBranch, msg);
			// Track mood from user transcripts
			if (msg.type === "transcript" && msg.role === "user" && !msg.partial) {
				sessionMessageCount++;
				const mood = detectMood(msg.text);
				if (mood) moodCounts[mood]++;
			}
			// Detect personal info in voice transcripts and save to memory
			if (msg.type === "transcript" && msg.role === "user" && !msg.partial && isMemoryWorthy(msg.text)) {
				saveMemoryInBackground(msg.text, opts.agentDir, opts.model, opts.env, () => {
					safeSend(browserWs, JSON.stringify({ type: "files_changed" }));
				});
			}
			// Auto-capture photo on memorable moments (with 60s cooldown)
			if (msg.type === "transcript" && msg.role === "user" && !msg.partial && isMomentWorthy(msg.text)) {
				const now = Date.now();
				if (now - lastMomentCaptureTs >= MOMENT_COOLDOWN) {
					lastMomentCaptureTs = now;
					// Use buffered frame if available and fresh (<5s)
					let frameBuffer: Buffer | undefined;
					if (latestVideoFrame && (now - latestVideoFrame.ts) < 5000) {
						frameBuffer = Buffer.from(latestVideoFrame.frame, "base64");
					}
					capturePhoto(agentRoot, msg.text.slice(0, 60), frameBuffer).catch((err) => {
						console.error(dim(`[voice] Auto photo capture failed: ${err.message}`));
					});
				}
			}
		};

		try {
			await adapter.connect({
				toolHandler: createToolHandler(sendToBrowser),
				onMessage: sendToBrowser,
			});
			console.log(dim(`[voice] Adapter ready (${opts.adapter})`));
		} catch (err: any) {
			console.error(dim(`[voice] Adapter connection failed: ${err.message}`));
			safeSend(browserWs, JSON.stringify({ type: "error", message: `Adapter failed: ${err.message}` }));
			browserWs.close();
			return;
		}

		// Parse browser messages into ClientMessage and forward to adapter
		browserWs.on("message", (data) => {
			try {
				const msg = JSON.parse(data.toString()) as ClientMessage;

				// Buffer video frames and throttle-write to disk for capture_photo tool
				if (msg.type === "video_frame") {
					const source = msg.source || "camera";
					if (source === "screen") {
						latestScreenFrame = { frame: msg.frame, mimeType: msg.mimeType, ts: Date.now() };
						const now = Date.now();
						if (now - lastScreenWriteTs >= 3000) {
							lastScreenWriteTs = now;
							const frameBuffer = Buffer.from(msg.frame, "base64");
							const framePath = join(agentRoot, LATEST_SCREEN_FILE);
							writeFile(framePath, frameBuffer).catch(() => {});
						}
					} else {
						latestVideoFrame = { frame: msg.frame, mimeType: msg.mimeType, ts: Date.now() };
						const now = Date.now();
						if (now - lastFrameWriteTs >= FRAME_WRITE_INTERVAL) {
							lastFrameWriteTs = now;
							const frameBuffer = Buffer.from(msg.frame, "base64");
							const framePath = join(agentRoot, LATEST_FRAME_FILE);
							writeFile(framePath, frameBuffer).catch(() => {});
						}
					}
				}

				if (msg.type === "text") {
					appendMessage(opts.agentDir, activeBranch, { type: "transcript", role: "user", text: msg.text });
					// Detect personal info and save to memory in background
					if (isMemoryWorthy(msg.text)) {
						saveMemoryInBackground(msg.text, opts.agentDir, opts.model, opts.env, () => {
							safeSend(browserWs, JSON.stringify({ type: "files_changed" }));
						});
					}
				} else if (msg.type === "file") {
					// Save uploaded file to disk so the text agent can use it
					const uploadsDir = join(agentRoot, "workspace");
					mkdirSync(uploadsDir, { recursive: true });
					const safeName = (msg as any).name.replace(/[^a-zA-Z0-9._-]/g, "_");
					const filePath = join(uploadsDir, safeName);
					writeFileSync(filePath, Buffer.from((msg as any).data, "base64"));
					const relPath = relative(agentRoot, filePath);
					console.log(dim(`[voice] Saved uploaded file: ${relPath}`));

					// Inject path into message so voice LLM tells the agent where the file is
					const userText = (msg as any).text || "";
					(msg as any).text = `${userText}${userText ? " " : ""}[File saved to: ${relPath} (absolute: ${filePath})]`;

					appendMessage(opts.agentDir, activeBranch, {
						type: "transcript", role: "user",
						text: `${userText} [Attached: ${safeName} → ${relPath}]`.trim(),
					});
				}
				adapter.send(msg);
			} catch {
				// Ignore unparseable messages
			}
		});

		browserWs.on("close", () => {
			console.log(dim("[voice] Browser disconnected"));
			adapter.disconnect().catch(() => {});
			// Summarize chat history, save mood, and write journal — track promises for graceful shutdown
			const p = Promise.allSettled([
				summarizeHistory(opts.agentDir, activeBranch).catch((err) => {
					console.error(dim(`[voice] Background summarization failed: ${err.message}`));
				}),
				saveMoodEntry(opts.agentDir, moodCounts, sessionMessageCount).catch((err) => {
					console.error(dim(`[voice] Mood save failed: ${err.message}`));
				}),
				writeJournalEntry(opts.agentDir, activeBranch, moodCounts, opts.model, opts.env).catch((err) => {
					console.error(dim(`[voice] Journal write failed: ${err.message}`));
				}),
			]);
			pendingShutdownWork.push(p);
		});
	});

	await new Promise<void>((resolve) => {
		httpServer.listen(port, () => resolve());
	});

	console.log(bold(`Voice server running on :${port}`));
	console.log(dim(`[voice] Backend: ${opts.adapter}`));
	console.log(dim(`[voice] Open http://localhost:${port} in your browser`));

	return async () => {
		// Gracefully close WebSocket connections to trigger close handlers (journal, mood, etc.)
		for (const client of wss.clients) {
			client.close(1000, "Server shutting down");
		}
		// Wait for close handlers to fire, then await their async work (journal writes, etc.)
		await new Promise((r) => setTimeout(r, 200));
		if (pendingShutdownWork.length > 0) {
			console.log(dim("[voice] Waiting for journal & mood saves..."));
			await Promise.allSettled(pendingShutdownWork);
		}
		wss.close();
		await new Promise<void>((resolve) => {
			httpServer.close(() => resolve());
		});
		console.log(dim("[voice] Server stopped"));
	};
}

function safeSend(ws: WS, data: string) {
	if (ws.readyState === WS.OPEN) {
		ws.send(data);
	}
}
