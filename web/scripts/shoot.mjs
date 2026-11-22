// Abre la página en Chrome, ejecuta pasos (clics, esperas) y guarda capturas. Uso: node scripts/shoot.mjs out-prefix w h
import { spawn } from "node:child_process";
import { writeFileSync, mkdtempSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

const [prefix, W = "1440", H = "900", url = "http://localhost:5190/#lab"] = process.argv.slice(2);
const chrome = spawn("/Applications/Google Chrome.app/Contents/MacOS/Google Chrome", [
  "--headless=new", "--remote-debugging-port=9333", `--user-data-dir=${mkdtempSync(join(tmpdir(), "cdp-"))}`, "--use-angle=metal", "about:blank",
], { stdio: "ignore" });
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
let tabs;
for (let i = 0; i < 40 && !tabs; i++) { await sleep(500); tabs = await fetch("http://127.0.0.1:9333/json").then((r) => r.json()).catch(() => null); }
const ws = new WebSocket(tabs.find((t) => t.type === "page").webSocketDebuggerUrl);
await new Promise((r) => (ws.onopen = r));
let id = 0;
const waiting = new Map();
ws.onmessage = (m) => { const d = JSON.parse(m.data); if (d.id && waiting.has(d.id)) { waiting.get(d.id)(d); waiting.delete(d.id); } if (d.method === "Runtime.consoleAPICalled") console.log("console:", d.params.args.map((a) => a.value).join(" ")); if (d.method === "Runtime.exceptionThrown") console.log("ERROR:", d.params.exceptionDetails.exception?.description); };
const send = (method, params = {}) => new Promise((r) => { waiting.set(++id, r); ws.send(JSON.stringify({ id, method, params })); });
const ev = async (expr) => (await send("Runtime.evaluate", { expression: expr, awaitPromise: true, returnByValue: true })).result?.result?.value;
const shot = async (name) => { const r = await send("Page.captureScreenshot", { format: "png" }); writeFileSync(`${prefix}-${name}.png`, Buffer.from(r.result.data, "base64")); console.log("captura", name); };
const click = (text) => ev(`(() => { const b = [...document.querySelectorAll('button')].find(b => b.textContent.includes(${JSON.stringify(text)})); if (b) b.click(); return !!b; })()`);

await send("Runtime.enable");
await send("Page.enable");
await send("Emulation.setDeviceMetricsOverride", { width: +W, height: +H, deviceScaleFactor: 1, mobile: +W < 600 });
await send("Page.navigate", { url });
for (let i = 0; i < 60; i++) { await sleep(1000); if (await ev("!!document.querySelector('canvas') && document.querySelectorAll('canvas').length >= 2")) break; }
await sleep(1500);
const steps = JSON.parse(process.env.STEPS ?? "[]");
for (const s of steps) {
  if (s.click) console.log("clic", s.click, await click(s.click));
  if (s.wait) await sleep(s.wait);
  if (s.eval) console.log("eval:", await ev(s.eval));
  if (s.shot) await shot(s.shot);
}
ws.close();
chrome.kill();
