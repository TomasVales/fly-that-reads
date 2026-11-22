import { useEffect, useMemo, useRef, useState, type ReactNode } from "react";
import { BrainView } from "./BrainView";

type Lang = "en" | "es";
type Res = { bpc: number[]; sample: string; words: number; kcActive: number };
export type LabData = {
  neurons: number; connections: number; inputs: number; kenyon: number; train: number; bpcRandom: number;
  books: { id: string; title: string }[];
  groups: Record<string, number>;
  results: Record<string, Res>;
};
type Seg = { text: string; kind: "prompt" | "fly"; book: number };
type Ready = { N: number; nnz: number; books: string[]; groups: string[]; vocab: string; pos: Int16Array; flags: Uint8Array };

const COLORS = ["#f4f4f2", "#f2c46d", "#7cc3f0", "#f08a8a"];
const PROMPTS: Record<string, string> = { shakespeare: "to be, or not to be, ", quijote: "en un lugar de la mancha, ", fierro: "aquí me pongo a cantar ", dante: "nel mezzo del cammin di " };
const ORDER = ["optic", "apl", "alpn", "inhib", "left", "half"];
const LESION: Record<string, { en: [string, string]; es: [string, string] }> = {
  optic: { en: ["Optic lobes", "The part of the brain that sees."], es: ["Lóbulos ópticos", "La parte del cerebro que ve."] },
  apl: { en: ["APL neuron", "One giant neuron per side that keeps the Kenyon cells in check."], es: ["Neurona APL", "Una neurona gigante por lado que frena a las células de Kenyon."] },
  alpn: { en: ["Olfactory projection", "The bridge from smell to the memory centre."], es: ["Proyección olfativa", "El puente entre el olfato y el centro de memoria."] },
  inhib: { en: ["All inhibitory neurons", "The ones that use GABA or glutamate to quiet others."], es: ["Todas las inhibidoras", "Las que usan GABA o glutamato para frenar a otras."] },
  left: { en: ["Left hemisphere", "Half of the brain, the left side."], es: ["Hemisferio izquierdo", "La mitad izquierda del cerebro."] },
  half: { en: ["Half, at random", "Every other neuron, picked at random."], es: ["La mitad, al azar", "Una de cada dos neuronas, elegidas al azar."] },
};
const SPEEDS = [{ lps: 5, en: "slow", es: "lento" }, { lps: 14, en: "normal", es: "normal" }, { lps: 1000, en: "max", es: "máximo" }];

const T = {
  title: { en: "Lab", es: "Laboratorio" },
  write: { en: "Live writing", es: "Escribe en vivo" },
  lesions: { en: "Lesions", es: "Lesiones" },
  dream: { en: "Dream", es: "Sueño" },
  loading: { en: "Loading the brain", es: "Cargando el cerebro" },
  unpacking: { en: "Wiring it up", es: "Conectando las neuronas" },
  start: { en: "Write", es: "Escribir" },
  stop: { en: "Stop", es: "Parar" },
  sleep: { en: "Fall asleep", es: "Dormir" },
  wake: { en: "Wake up", es: "Despertar" },
  style: { en: "What it learned from", es: "Lo que aprendió de" },
  begin: { en: "Write the beginning, the fly continues", es: "Escribí el comienzo y la mosca sigue" },
  creative: { en: "Creativity", es: "Creatividad" },
  speed: { en: "Speed", es: "Velocidad" },
  empty: { en: "The fly is waiting for its first letter.", es: "La mosca espera su primera letra." },
};

export default function Lab({ lang, onClose }: { lang: Lang; onClose: () => void }) {
  const es = lang === "es";
  const t = (o: { en: string; es: string }) => o[lang];
  const [data, setData] = useState<LabData | null>(null);
  const [ready, setReady] = useState<Ready | null>(null);
  const [prog, setProg] = useState<{ loaded: number; total: number; unpacking?: boolean }>({ loaded: 0, total: 0 });
  const [err, setErr] = useState("");
  const [tab, setTab] = useState<"write" | "lesions" | "dream">("write");
  const [book, setBook] = useState(0);
  const [prompt, setPrompt] = useState(PROMPTS.shakespeare);
  const [temp, setTemp] = useState(0.8);
  const [speed, setSpeed] = useState(1);
  const [lesions, setLesions] = useState<string[]>([]);
  const [segs, setSegs] = useState<Seg[]>([]);
  const [running, setRunning] = useState<"" | "write" | "dream">("");
  const [session, setSession] = useState<"write" | "dream">("write");
  const [promptBits, setPromptBits] = useState<number | null>(null);
  const worker = useRef<Worker | null>(null);
  const view = useRef<BrainView | null>(null);
  const host = useRef<HTMLDivElement>(null);
  const textBox = useRef<HTMLDivElement>(null);
  const pending = useRef<Seg[]>([]);
  const activeEl = useRef<HTMLSpanElement>(null);
  const locale = useRef("en-US");
  locale.current = es ? "es-AR" : "en-US";

  useEffect(() => {
    fetch("data/lab.json").then((r) => r.json()).then(setData).catch(() => {});
    const w = new Worker(new URL("./worker.ts", import.meta.url), { type: "module" });
    worker.current = w;
    let lastAct: Uint8Array | null = null;
    w.onmessage = (e) => {
      const m = e.data;
      if (m.type === "progress") setProg(m);
      else if (m.type === "ready") setReady(m);
      else if (m.type === "error") setErr(m.message);
      else if (m.type === "letter") {
        lastAct = m.act;
        const kind = m.prompt ? "prompt" : "fly";
        const q = pending.current, l = q[q.length - 1];
        if (l && l.kind === kind && l.book === (m.book ?? -1)) l.text += m.ch;
        else q.push({ text: m.ch, kind, book: m.book ?? -1 });
      } else if (m.type === "promptBits") setPromptBits(m.bits);
      else if (m.type === "done") setRunning("");
    };
    // las letras llegan rápido: se juntan y se dibujan una vez por cuadro
    let raf = 0;
    const flush = () => {
      raf = requestAnimationFrame(flush);
      if (lastAct && view.current) {
        view.current.setActivity(lastAct);
        let n = 0;
        for (let i = 0; i < lastAct.length; i++) if (lastAct[i] > 12) n++;
        if (activeEl.current) activeEl.current.textContent = n.toLocaleString(locale.current);
        lastAct = null;
      }
      if (pending.current.length) {
        const add = pending.current;
        pending.current = [];
        setSegs((s) => {
          const out = s.slice();
          for (const a of add) {
            const l = out[out.length - 1];
            if (l && l.kind === a.kind && l.book === a.book) out[out.length - 1] = { ...l, text: (l.text + a.text).slice(-4000) };
            else out.push(a);
          }
          return out.slice(-80);
        });
      }
    };
    raf = requestAnimationFrame(flush);
    w.postMessage({ type: "load", url: new URL("lab/brain.bin", document.baseURI).href });
    return () => { cancelAnimationFrame(raf); w.terminate(); };
  }, []);

  useEffect(() => {
    if (!ready || !host.current) return;
    const v = new BrainView(host.current, ready.pos, ready.flags);
    view.current = v;
    return () => { v.dispose(); view.current = null; };
  }, [ready]);

  useEffect(() => {
    if (!ready) return;
    worker.current?.postMessage({ type: "lesions", names: lesions });
    const bits = lesions.map((n) => 1 << (2 + ready.groups.indexOf(n))).reduce((a, b) => a | b, 0);
    view.current?.setLesioned(ready.flags, bits);
  }, [lesions, ready]);

  useEffect(() => { worker.current?.postMessage({ type: "speed", lps: SPEEDS[speed].lps }); }, [speed, ready]);
  useEffect(() => { if (view.current) view.current.dreaming = running === "dream" ? 1 : 0; }, [running]);
  useEffect(() => { textBox.current?.scrollTo({ top: textBox.current.scrollHeight }); }, [segs]);

  const books = data?.books ?? (ready?.books ?? []).map((id) => ({ id, title: id }));
  const go = (p = prompt, b = book) => {
    setSegs([]);
    setPromptBits(null);
    pending.current = [];
    worker.current?.postMessage({ type: "write", prompt: p, book: b, temp });
    setRunning("write");
    setSession("write");
  };
  const stop = () => { worker.current?.postMessage({ type: "stop" }); setRunning(""); };
  const sleep = () => { setSegs([]); setPromptBits(null); pending.current = []; worker.current?.postMessage({ type: "dream" }); setRunning("dream"); setSession("dream"); };
  const pickBook = (i: number) => {
    const wasDefault = Object.values(PROMPTS).includes(prompt);
    setBook(i);
    if (wasDefault) setPrompt(PROMPTS[books[i]?.id] ?? prompt);
  };

  const pctLoaded = prog.total ? Math.min(100, (prog.loaded / prog.total) * 100) : 0;
  const mb = (n: number) => (n / 1e6).toFixed(1).replace(".", es ? "," : ".");
  const nf = (n: number) => n.toLocaleString(es ? "es-AR" : "en-US");

  const mainButton = running === "write"
    ? <button onClick={stop} className="rounded-full bg-white px-5 py-2 font-medium text-black">{t(T.stop)}</button>
    : <button onClick={() => go()} disabled={!ready} className="rounded-full bg-white px-5 py-2 font-medium text-black disabled:opacity-40">{t(T.start)} ▸</button>;

  const intro = useMemo(() => ({
    write: es
      ? <>Esto no es una grabación: el cerebro de la mosca está corriendo ahora en tu navegador. {ready ? nf(ready.N) : "…"} neuronas y {ready ? nf(ready.nnz) : "…"} conexiones reales, letra por letra.</>
      : <>This is not a recording: the fly's brain is running right now in your browser. {ready ? nf(ready.N) : "…"} neurons and {ready ? nf(ready.nnz) : "…"} real connections, letter by letter.</>,
    lesions: es
      ? <>Como en un laboratorio: la mosca aprendió con el cerebro sano y acá apagás una parte. Las neuronas apagadas se ven en rojo. Los números los medimos sobre {books[book]?.title ?? "el libro"}, en texto que nunca leyó.</>
      : <>Like in a real lab: the fly learned with a healthy brain, and here you switch part of it off. Silenced neurons turn red. Numbers were measured on {books[book]?.title ?? "the book"}, on text it never read.</>,
    dream: es
      ? <>Sin ningún libro delante, la mosca escribe y se escucha a sí misma. Lo que recuerda de cada libro va y viene: el color muestra con cuál está soñando.</>
      : <>With no book in front of it, the fly writes and listens to itself. What it remembers of each book comes and goes: the colour shows which one it is dreaming of.</>,
  }), [es, ready, books, book]);

  const healthy = data?.results.none;
  const output = (
    <>
          {/* lo que escribe */}
          <div ref={textBox} className="thin-scroll min-h-[9rem] flex-1 shrink-0 overflow-y-auto rounded-2xl border border-white/10 bg-black/40 p-4 font-serif text-[19px] leading-relaxed [overflow-wrap:anywhere]">
            {segs.length === 0 ? <span className="text-white/30 italic">{t(T.empty)}</span> : segs.map((s, i) => (
              <span key={i} className={s.kind === "prompt" ? "text-white/45" : ""} style={session === "dream" && s.book >= 0 ? { color: COLORS[s.book] } : undefined}>{s.text}</span>
            ))}
            {running && <span className="animate-pulse">▍</span>}
          </div>
          {promptBits !== null && (
            <p className="text-sm text-white/55">
              {es ? "Tu comienzo la sorprendió " : "Your beginning surprised it by "}<b className="text-white">{promptBits.toFixed(2)} bits</b>{es ? " por letra" : " per letter"}
              {" "}({es ? "al azar" : "random"}: {(data?.bpcRandom ?? 5.55).toFixed(2)}, {es ? "en su libro" : "on its book"}: {healthy ? healthy.bpc[book].toFixed(2) : "—"}).
              {" "}{promptBits < (healthy?.bpc[book] ?? 3.5) + 0.3 ? (es ? "Le resulta familiar." : "It feels familiar.") : (es ? "Le resulta raro." : "It feels strange.")}
            </p>
          )}
    </>
  );

  return (
    <div className="fixed inset-0 z-30 flex flex-col bg-[#0b0b0b] text-[#f4f4f2]">
      <header className="flex items-center justify-between gap-3 border-b border-white/10 px-4 py-3 md:px-7">
        <div className="flex min-w-0 items-center gap-2 sm:gap-4">
          <h2 className="hidden text-xl font-medium tracking-tight sm:block md:text-2xl">{t(T.title)}</h2>
          <nav className="flex gap-1 overflow-x-auto">
            {(["write", "lesions", "dream"] as const).map((k) => (
              <button key={k} onClick={() => setTab(k)} className={`label shrink-0 rounded-full px-3 py-1.5 ${tab === k ? "bg-white/15 text-white" : "text-white/50 hover:text-white"}`}>
                {t(T[k])}{k === "lesions" && lesions.length ? ` · ${lesions.length}` : ""}
              </button>
            ))}
          </nav>
        </div>
        <button onClick={onClose} aria-label="close" className="label shrink-0 rounded-full border border-white/25 px-3 py-1.5 hover:bg-white hover:text-black">✕</button>
      </header>

      <div className="flex min-h-0 flex-1 flex-col md:flex-row">
        {/* cerebro */}
        <div className="relative h-[38svh] shrink-0 md:h-auto md:flex-[1.25]">
          <div ref={host} className="absolute inset-0" />
          {!ready && (
            <div className="absolute inset-0 flex flex-col items-center justify-center gap-3 px-8 text-center">
              {err ? <p className="text-red-300">{err}</p> : <>
                <p className="label text-white/60">{prog.unpacking ? t(T.unpacking) : t(T.loading)}{prog.total ? ` · ${mb(prog.loaded)} / ${mb(prog.total)} MB` : ""}</p>
                <div className="h-1 w-56 overflow-hidden rounded-full bg-white/10"><div className="h-full bg-white transition-[width]" style={{ width: `${prog.unpacking ? 100 : pctLoaded}%` }} /></div>
              </>}
            </div>
          )}
          {ready && (
            <div className="label pointer-events-none absolute bottom-3 left-4 flex flex-wrap gap-x-4 gap-y-1 text-white/50 md:bottom-5 md:left-7">
              <span><span className="mr-1.5 inline-block h-2 w-2 rounded-full bg-[#73b8f2]" />{es ? "olfato" : "smell"}</span>
              <span><span className="mr-1.5 inline-block h-2 w-2 rounded-full bg-[#f2b85a]" />Kenyon</span>
              {lesions.length > 0 && <span><span className="mr-1.5 inline-block h-2 w-2 rounded-full bg-[#e5372b]" />{es ? "apagadas" : "silenced"}</span>}
              <span className="text-white/70">{es ? "activas ahora" : "active now"}: <span ref={activeEl}>0</span></span>
            </div>
          )}
        </div>

        {/* panel */}
        <div className="thin-scroll flex min-h-0 flex-1 flex-col gap-4 overflow-y-auto border-white/10 p-4 md:max-w-[560px] md:border-l md:p-7">
          <p className="text-[15px] leading-relaxed text-white/65">{intro[tab]}</p>

          {tab === "write" && (
            <div className="space-y-4">
              <div>
                <p className="label mb-2 text-white/45">{t(T.style)}</p>
                <div className="flex flex-wrap gap-2">
                  {books.map((b, i) => <Btn key={b.id} on={i === book} onClick={() => pickBook(i)}><span className="mr-1.5 inline-block h-2 w-2 rounded-full ring-1 ring-black/30" style={{ background: COLORS[i] }} />{b.title}</Btn>)}
                </div>
              </div>
              <div>
                <p className="label mb-2 text-white/45">{t(T.begin)}</p>
                <div className="flex gap-2">
                  <input value={prompt} onChange={(e) => setPrompt(e.target.value)} onKeyDown={(e) => e.key === "Enter" && ready && go()} maxLength={160}
                    className="min-w-0 flex-1 rounded-full border border-white/20 bg-white/5 px-4 py-2 font-serif text-[17px] italic outline-none focus:border-white/60" />
                  {mainButton}
                </div>
              </div>
              {controls()}
            </div>
          )}

          {tab === "lesions" && (
            <div className="space-y-2">
              <div className="flex flex-wrap items-center gap-2 pb-1">
                {mainButton}
                {lesions.length > 0 && <Btn onClick={() => setLesions([])}>{es ? "Curar todo" : "Heal all"}</Btn>}
              </div>
              <div className="[&>div:first-child]:max-h-[9.5rem] [&>div:first-child]:min-h-[6rem] space-y-2 pb-2">{output}</div>
              {ORDER.filter((k) => !data || data.groups[k] !== undefined).map((k) => {
                const on = lesions.includes(k);
                const r = data?.results[k];
                return (
                  <button key={k} onClick={() => setLesions((l) => (on ? l.filter((x) => x !== k) : [...l, k]))}
                    className={`block w-full rounded-2xl border p-3 text-left transition-colors ${on ? "border-[#e5372b]/70 bg-[#e5372b]/10" : "border-white/10 bg-white/[0.03] hover:border-white/30"}`}>
                    <div className="flex items-baseline justify-between gap-3">
                      <span className="font-medium">{LESION[k][lang][0]} <span className="font-mono text-xs text-white/45">· {data ? nf(data.groups[k]) : ""} {es ? (data?.groups[k] === 1 ? "neurona" : "neuronas") : "neurons"}</span></span>
                      <span className={`label ${on ? "text-[#ff8a80]" : "text-white/40"}`}>{on ? (es ? "apagada" : "off") : (es ? "sana" : "on")}</span>
                    </div>
                    <p className="mt-1 text-sm text-white/55">{LESION[k][lang][1]}</p>
                    {r && healthy && <Effect healthy={healthy} r={r} book={book} es={es} random={data!.bpcRandom} />}
                  </button>
                );
              })}
              <p className="text-xs text-white/40">{es ? "Podés apagar varias a la vez, incluso mientras escribe." : "You can switch off several at once, even while it writes."}</p>
            </div>
          )}

          {tab === "dream" && (
            <div className="space-y-4">
              <div className="flex flex-wrap items-center gap-2">
                {running === "dream"
                  ? <button onClick={stop} className="rounded-full bg-white px-5 py-2 font-medium text-black">{t(T.wake)}</button>
                  : <button onClick={sleep} disabled={!ready} className="rounded-full bg-[#a08cff] px-5 py-2 font-medium text-black disabled:opacity-40">{t(T.sleep)} ☾</button>}
              </div>
              <div className="label flex flex-wrap gap-x-4 gap-y-1 text-white/50">
                {books.map((b, i) => <span key={b.id}><span className="mr-1.5 inline-block h-2 w-2 rounded-full" style={{ background: COLORS[i] }} />{b.title}</span>)}
              </div>
            </div>
          )}

          {tab !== "lesions" && output}
        </div>
      </div>
    </div>
  );

  function controls() {
    return (
      <div className="flex flex-wrap items-center gap-x-6 gap-y-3">
        <label className="flex items-center gap-3">
          <span className="label text-white/45">{t(T.creative)}</span>
          <input type="range" min={0.3} max={1.4} step={0.05} value={temp} onChange={(e) => setTemp(+e.target.value)} className="w-28 accent-white" />
          <span className="font-mono text-xs text-white/60">{temp.toFixed(2)}</span>
        </label>
        <div className="flex items-center gap-2">
          <span className="label text-white/45">{t(T.speed)}</span>
          {SPEEDS.map((s, i) => <button key={s.lps} onClick={() => setSpeed(i)} className={`rounded-full px-2.5 py-1 font-mono text-xs ${i === speed ? "bg-white text-black" : "border border-white/20 text-white/70"}`}>{s[lang]}</button>)}
        </div>
      </div>
    );
  }
}

function Btn({ on, children, onClick }: { on?: boolean; children: ReactNode; onClick: () => void }) {
  return <button onClick={onClick} className={`rounded-full px-3 py-1.5 text-sm transition-colors ${on ? "bg-white text-black" : "border border-white/20 text-white/75 hover:border-white/60"}`}>{children}</button>;
}

function Effect({ healthy, r, book, es, random }: { healthy: Res; r: Res; book: number; es: boolean; random: number }) {
  const h = healthy.bpc[book], l = r.bpc[book];
  const worse = l - h;
  const verdict = Math.abs(worse) < 0.05 ? (es ? "no le cambia nada" : "no change at all")
    : l > random ? (es ? "peor que teclear al azar" : "worse than random typing")
    : worse > 0.5 ? (es ? "escribe mucho peor" : "writes much worse")
    : worse > 0 ? (es ? "escribe un poco peor" : "writes a bit worse") : (es ? "escribe un poco mejor" : "writes a bit better");
  return (
    <div className="mt-2 flex flex-wrap items-baseline gap-x-4 gap-y-1 font-mono text-xs text-white/55">
      <span>{h.toFixed(2)} → <b className="text-white/90">{l.toFixed(2)}</b> bits</span>
      <span>Kenyon {es ? "activas" : "active"} {Math.round(healthy.kcActive * 100)}% → {Math.round(r.kcActive * 100)}%</span>
      <span className="font-sans text-white/70">{verdict}</span>
    </div>
  );
}
