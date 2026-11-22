/// <reference lib="webworker" />
import { Brain, normalize, sample, softmax } from "./core";

type In =
  | { type: "load"; url: string }
  | { type: "write"; prompt: string; book: number; temp: number }
  | { type: "dream" }
  | { type: "lesions"; names: string[] }
  | { type: "speed"; lps: number }
  | { type: "stop" };

let brain: Brain | null = null;
let job = 0;
let lps = 14;
const post = (m: unknown, t: Transferable[] = []) => (self as DedicatedWorkerGlobalScope).postMessage(m, t);

function activity() {
  const b = brain!, a = new Uint8Array(b.N);
  for (let i = 0; i < b.N; i++) { const v = Math.abs(b.x[i]) * 255; a[i] = v > 255 ? 255 : v; }
  return a;
}

const wait = (ms: number) => new Promise((r) => setTimeout(r, ms));

async function paced<T>(id: number, fn: () => T) {
  const t = performance.now();
  const r = fn();
  await wait(Math.max(0, 1000 / lps - (performance.now() - t)));
  return id === job ? r : null;
}

async function write(id: number, prompt: string, book: number, temp: number) {
  const b = brain!;
  b.reset();
  const ids = [...normalize(prompt, b.h.vocab)].map((c) => b.h.vocab.indexOf(c));
  let bits = 0;
  for (const c of ids) {
    const ok = await paced(id, () => {
      const p = softmax(b.logits(book));
      const s = -Math.log2(p[c] + 1e-12);
      bits += s;
      b.step(c);
      const act = activity();
      post({ type: "letter", ch: b.h.vocab[c], prompt: true, bits: s, act }, [act.buffer]);
      return true;
    });
    if (!ok) return;
  }
  if (ids.length) post({ type: "promptBits", bits: bits / ids.length });
  for (let n = 0; n < 700; n++) {
    const ok = await paced(id, () => {
      const c = sample(softmax(b.logits(book), temp));
      b.step(c);
      const act = activity();
      post({ type: "letter", ch: b.h.vocab[c], prompt: false, book, act }, [act.buffer]);
      return true;
    });
    if (!ok) return;
  }
  post({ type: "done" });
}

/** sueño: sin libro delante, escribe y se escucha a sí misma; lo que recuerda de cada libro va y viene */
async function dream(id: number) {
  const b = brain!, K = b.heads.length, V = b.V;
  b.reset();
  const w = Array.from({ length: K }, () => Math.random());
  const tmp = new Float32Array(V), mix = new Float32Array(V);
  for (let n = 0; n < 3000; n++) {
    const ok = await paced(id, () => {
      for (let k = 0; k < K; k++) w[k] = Math.max(-3, Math.min(3, w[k] * 0.995 + (Math.random() - 0.5) * 0.35));
      const e = w.map((v) => Math.exp(v * 2)), s = e.reduce((a, c) => a + c, 0);
      mix.fill(0);
      e.forEach((ek, k) => { b.logits(k, tmp); for (let v = 0; v < V; v++) mix[v] += (ek / s) * tmp[v]; });
      const c = sample(softmax(mix, 0.95));
      b.step(c);
      const act = activity();
      const top = e.indexOf(Math.max(...e));
      post({ type: "letter", ch: b.h.vocab[c], prompt: false, book: top, act }, [act.buffer]);
      return true;
    });
    if (!ok) return;
  }
  post({ type: "done" });
}

async function load(url: string) {
  const r = await fetch(url);
  if (!r.ok || !r.body) throw new Error(`HTTP ${r.status}`);
  const total = +(r.headers.get("content-length") ?? 0);
  const reader = r.body.getReader();
  const parts: Uint8Array[] = [];
  let got = 0, last = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    parts.push(value);
    got += value.length;
    if (performance.now() - last > 80) { post({ type: "progress", loaded: got, total }); last = performance.now(); }
  }
  let bytes = new Uint8Array(got);
  let o = 0;
  for (const p of parts) { bytes.set(p, o); o += p.length; }
  post({ type: "progress", loaded: got, total: got, unpacking: true });
  if (bytes[0] === 0x1f && bytes[1] === 0x8b) {
    bytes = new Uint8Array(await new Response(new Blob([bytes]).stream().pipeThrough(new DecompressionStream("gzip"))).arrayBuffer());
  }
  brain = new Brain(bytes.buffer);
  const pos = brain.pos.slice(), flags = brain.flags.slice();
  post({ type: "ready", N: brain.N, nnz: brain.h.nnz, books: brain.h.books, groups: brain.h.groups, vocab: brain.h.vocab, pos, flags }, [pos.buffer, flags.buffer]);
}

self.onmessage = (ev: MessageEvent<In>) => {
  const m = ev.data;
  if (m.type === "load") load(m.url).catch((e) => post({ type: "error", message: String(e) }));
  if (!brain) return;
  if (m.type === "speed") lps = m.lps;
  else if (m.type === "lesions") brain.setLesions(m.names);
  else if (m.type === "stop") job++;
  else if (m.type === "write") write(++job, m.prompt, m.book, m.temp);
  else if (m.type === "dream") dream(++job);
};
