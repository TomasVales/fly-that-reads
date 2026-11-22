import { useEffect, useMemo, useRef, useState, type ReactNode } from "react";
import { FlyScene } from "./scene";
import { LOG10_UNIVERSE, at, firstWhen, fmtDuration, fmtInt, log10YearsToWrite, pct, totalChars, type Book, type Checkpoint, type Run } from "./data";
import { mockRun } from "./mock";

const SPEEDS = [1, 10, 100, 1000, 10000];
const PAGE = 430; // letras por página dibujada
const COLORS: Record<string, string> = { shakespeare: "#f4f4f2", quijote: "#f2c46d", fierro: "#7cc3f0", dante: "#f08a8a" };
const SHORT: Record<string, { en: string; es: string }> = {
  shakespeare: { en: "Complete works", es: "Obras completas" },
};

type Lang = "en" | "es";
type L = Record<Lang, string>;
const T = {
  title: { en: "The fly that reads", es: "La mosca que lee" },
  sub: {
    en: "The real wiring of a fruit-fly brain reads four classics. How far can it get?",
    es: "El cableado real del cerebro de una mosca lee cuatro clásicos. ¿Hasta dónde puede llegar?",
  },
  study: { en: "Reading for", es: "Leyendo hace" },
  read: { en: "Letters read", es: "Letras leídas" },
  surprise: { en: "Surprise per letter", es: "Sorpresa por letra" },
  meter: { en: "Writes like the author", es: "Escribe como el autor" },
  goal: { en: "goal", es: "meta" },
  won: { en: "got there first", es: "llegó primero" },
  demo: { en: "Demo data · not results yet", es: "Datos de prueba · todavía no son resultados" },
  running: { en: "Real experiment running", es: "Experimento real en curso" },
  real: { en: "Real FlyWire brain", es: "Cerebro real de FlyWire" },
  about: { en: "How does it work?", es: "¿Cómo funciona?" },
  exams: { en: "Exams", es: "Exámenes" },
  close: { en: "Close", es: "Cerrar" },
  watching: { en: "Watching", es: "Mirando" },
  random: { en: "random typing", es: "al azar" },
  human: { en: "a person", es: "una persona" },
  curve: { en: "Surprise per letter while reading", es: "Sorpresa por letra mientras lee" },
  shuffled: { en: "same brain, wiring shuffled", es: "mismo cerebro, cables mezclados" },
  notYet: { en: "not yet", es: "todavía no" },
  never: { en: "never", es: "nunca" },
  at: { en: "after", es: "a las" },
};

export default function App() {
  const [lang, setLang] = useState<Lang>(() => (navigator.language.startsWith("es") ? "es" : "en"));
  const t = (o: L) => o[lang];
  const es = lang === "es";
  const [run, setRun] = useState<Run>(mockRun);
  const [focus, setFocus] = useState(0);
  const [speedIdx, setSpeedIdx] = useState(2);
  const [playing, setPlaying] = useState(true);
  const [about, setAbout] = useState(false);
  const [examsOpen, setExamsOpen] = useState(false);
  const [, force] = useState(0);
  const chars = useRef(0);
  const host = useRef<HTMLDivElement>(null);
  const scene = useRef<FlyScene | null>(null);

  // resultados reales si existen (los escribe el experimento mientras corre); se recargan cada minuto
  useEffect(() => {
    let stop = false;
    const load = () =>
      fetch(`data/run.json?t=${Date.now()}`).then((r) => (r.ok ? r.json() : null)).then((j: Run | null) => {
        if (j && !stop) setRun((prev) => (prev.meta.demo || JSON.stringify(prev.meta) !== JSON.stringify(j.meta) ? j : prev));
      }).catch(() => {});
    load();
    const id = setInterval(load, 60_000);
    return () => { stop = true; clearInterval(id); };
  }, []);

  const total = useMemo(() => totalChars(run), [run]);
  // meta de la carrera: la del experimento, o la mejor marca redondeada si nadie la alcanzó
  const target = useMemo(() => {
    const best = Math.max(...run.books.map((b) => pct(run, b.checkpoints[b.checkpoints.length - 1].bpc)));
    return best >= run.meta.target ? run.meta.target : Math.max(5, Math.floor((best - 1) / 5) * 5);
  }, [run]);
  const finish = useMemo(() => run.books.map((b) => {
    for (let i = 1; i < b.checkpoints.length; i++) {
      const a = b.checkpoints[i - 1], c = b.checkpoints[i];
      const pa = pct(run, a.bpc), pc = pct(run, c.bpc);
      if (pc >= target) return a.chars + ((target - pa) / Math.max(1e-9, pc - pa)) * (c.chars - a.chars);
    }
    return Infinity;
  }), [run, target]);
  const winner = finish.indexOf(Math.min(...finish));

  // escena 3D
  useEffect(() => {
    const s = new FlyScene(host.current!);
    scene.current = s;
    s.setBrain(Float32Array.from(run.brain.positions), run.meta.groups);
    const stop = () => s.setAutoRotate(false);
    host.current!.addEventListener("pointerdown", stop, { once: true });
    return () => s.dispose();
  }, [run.brain, run.meta.groups]);

  // bucle de reproducción
  useEffect(() => {
    let raf = 0, last = performance.now(), lastUi = 0, lastPulse = 0;
    const tick = (now: number) => {
      raf = requestAnimationFrame(tick);
      const dt = Math.max(0, Math.min(0.1, (now - last) / 1000));
      last = now;
      const s = scene.current;
      if (!s) return;
      const speed = SPEEDS[speedIdx];
      s.speed = speed;
      s.playing = playing;
      if (playing) chars.current = Math.min(total, chars.current + dt * speed * run.meta.charsPerSecond);
      if (chars.current >= total && playing) setPlaying(false);
      const book = run.books[focus];
      const page = Math.floor(chars.current / PAGE);
      const nPages = Math.max(1, Math.floor(book.excerpt.length / PAGE));
      const chunk = (k: number) => { const i = ((k % nPages) + nPages) % nPages; return book.excerpt.slice(i * PAGE, (i + 1) * PAGE); };
      s.setPageNumber(page);
      if (now - lastUi > 90) {
        s.setPages(chunk(page - 1), chunk(page), Math.floor(chars.current % PAGE), book.title);
        force((x) => x + 1);
        lastUi = now;
      }
      if (playing && now - lastPulse > 140) {
        const { cp } = at(book, chars.current);
        if (cp.active?.length) {
          const k = Math.min(cp.active.length, 18 + Math.round(Math.log10(speed + 1) * 10));
          const pick: number[] = [];
          for (let i = 0; i < k; i++) pick.push(cp.active[Math.floor(Math.random() * cp.active.length)]);
          s.pulseNeurons(pick, 0.9);
        }
        lastPulse = now;
      }
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [run, focus, speedIdx, playing, total]);

  const c = chars.current;
  const seconds = c / run.meta.charsPerSecond;
  const fb = run.books[focus];
  const status = run.meta.demo ? { txt: t(T.demo), cls: "border-amber-300/40 text-amber-200/90" }
    : !run.meta.done ? { txt: `${t(T.running)} · ${Math.round((run.meta.progress ?? 0) * 100)}%`, cls: "border-sky-300/40 text-sky-200/90" }
    : { txt: `${t(T.real)} · ${fmtInt(run.meta.neurons, es)} ${es ? "neuronas" : "neurons"}`, cls: "border-white/25 text-white/70" };

  const examPanel = <ExamPanel run={run} book={fb} chars={c} lang={lang} />;

  return (
    <div className="relative h-[100svh] w-full overflow-hidden bg-[#0b0b0b] text-[#f4f4f2]">
      <div ref={host} className="absolute inset-0" />

      {/* título */}
      <header className="pointer-events-none absolute left-0 top-0 flex w-full items-start justify-between gap-4 p-4 md:p-7">
        <div className="max-w-md">
          <h1 className="text-3xl font-medium tracking-[-0.04em] md:text-5xl">{t(T.title)}</h1>
          <p className="mt-2 hidden text-sm text-white/60 sm:block md:text-base">{t(T.sub)}</p>
          <p className={`label mt-3 inline-block rounded-full border px-3 py-1 ${status.cls}`}>{status.txt}</p>
        </div>
        <div className="pointer-events-auto flex flex-wrap items-center justify-end gap-2 md:gap-3">
          <button onClick={() => setExamsOpen(true)} className="label rounded-full border border-white/25 px-3 py-1.5 hover:bg-white hover:text-black md:hidden">{t(T.exams)}</button>
          <button onClick={() => setAbout(true)} className="label rounded-full border border-white/25 px-3 py-1.5 hover:bg-white hover:text-black">{t(T.about)}</button>
          <div className="label flex gap-2">
            {(["en", "es"] as Lang[]).map((l) => (
              <button key={l} onClick={() => setLang(l)} className={l === lang ? "opacity-100" : "opacity-40 hover:opacity-100"}>{l}</button>
            ))}
          </div>
        </div>
      </header>

      {/* exámenes (escritorio) */}
      <section className="thin-scroll absolute left-7 top-48 hidden max-h-[calc(100svh-17rem)] w-[19rem] overflow-y-auto rounded-2xl border border-white/10 bg-black/45 p-4 backdrop-blur-md md:block">
        <p className="label text-white/50">{t(T.watching)} · {fb.title}</p>
        <dl className="mt-3 space-y-1.5 text-sm">
          <Row k={t(T.study)} v={fmtDuration(seconds, es)} />
          <Row k={t(T.read)} v={fmtInt(c, es)} />
          <Row k={t(T.surprise)} v={`${at(fb, c).bpc.toFixed(2)} bits`} />
          <Row k={t(T.meter)} v={`${pct(run, at(fb, c).bpc).toFixed(1)}%`} />
        </dl>
        <div className="label mt-2 flex justify-between text-white/35"><span>0% = {t(T.random)}</span><span>100% = {t(T.human)}</span></div>
        {run.meta.groups && (
          <div className="label mt-3 flex flex-wrap gap-x-4 gap-y-1 text-white/55">
            <span><span className="mr-1.5 inline-block h-2 w-2 rounded-full bg-[#73b8f2]" />{es ? "olfato (entrada)" : "smell (input)"}</span>
            <span><span className="mr-1.5 inline-block h-2 w-2 rounded-full bg-[#f2b85a]" />{es ? "Kenyon (aprende)" : "Kenyon (learns)"}</span>
          </div>
        )}
        <div className="mt-4 border-t border-white/10 pt-3">{examPanel}</div>
      </section>

      {/* carrera + curva */}
      <aside className="absolute bottom-[7.5rem] right-3 w-[min(400px,calc(100%-24px))] space-y-2 md:bottom-auto md:right-7 md:top-1/2 md:-translate-y-1/2">
        {run.books.map((b, i) => {
          const st = at(b, c);
          const p = pct(run, st.bpc);
          const done = c >= finish[i];
          return (
            <button key={b.id} onClick={() => setFocus(i)}
              className={`block w-full rounded-2xl border p-3 text-left backdrop-blur-md transition-colors ${i === focus ? "border-white/60 bg-white/10" : "border-white/10 bg-black/45 hover:border-white/30"}`}>
              <div className="flex items-baseline justify-between gap-3">
                <span className="flex min-w-0 items-baseline gap-2 truncate text-sm font-medium">
                  <span className="inline-block h-2 w-2 shrink-0 rounded-full" style={{ background: COLORS[b.id] ?? "#fff" }} />
                  {b.title}<span className="truncate font-normal text-white/50">· {SHORT[b.id]?.[lang] ?? b.author}</span>
                </span>
                <span className="font-mono text-sm tabular-nums">{p.toFixed(1)}%{done && i === winner ? " 🏆" : ""}</span>
              </div>
              <div className="relative mt-2 h-1.5 overflow-hidden rounded-full bg-white/10">
                <div className="absolute inset-y-0 left-0 rounded-full transition-[width] duration-200" style={{ width: `${p}%`, background: COLORS[b.id] ?? "#fff" }} />
                <div className="absolute inset-y-0 w-px bg-white/70" style={{ left: `${target}%` }} title={t(T.goal)} />
              </div>
              <p className="mt-2 hidden truncate font-serif text-[15px] italic text-white/80 sm:block">“{st.cp.sample}<span className="animate-pulse">▍</span>”</p>
            </button>
          );
        })}
        <div className="hidden rounded-2xl border border-white/10 bg-black/45 p-3 backdrop-blur-md md:block">
          <p className="label mb-2 text-white/50">{t(T.curve)}</p>
          <Curve run={run} chars={c} focus={focus} lang={lang} />
        </div>
      </aside>

      {/* ganador */}
      {winner >= 0 && c >= finish[winner] && (
        <div className="pointer-events-none absolute left-1/2 top-[22%] hidden -translate-x-1/2 rounded-full border border-white/30 bg-black/60 px-5 py-2 text-center backdrop-blur-md md:block">
          🏆 <b>{run.books[winner].title}</b> {t(T.won)} · {target}% · {fmtDuration(finish[winner] / run.meta.charsPerSecond, es)}
        </div>
      )}

      {/* controles */}
      <footer className="absolute inset-x-0 bottom-0 flex flex-col gap-3 border-t border-white/10 bg-black/60 p-3 backdrop-blur-md md:flex-row md:items-center md:px-7 md:py-4">
        <div className="flex items-center gap-1.5 overflow-x-auto md:gap-2">
          <button onClick={() => { if (c >= total) chars.current = 0; setPlaying(!playing); }} className="h-10 w-10 shrink-0 rounded-full bg-white text-lg text-black" aria-label="play">{playing ? "❚❚" : "▶"}</button>
          {SPEEDS.map((sp, i) => (
            <button key={sp} onClick={() => { setSpeedIdx(i); setPlaying(true); }}
              className={`shrink-0 rounded-full px-3 py-1.5 font-mono text-xs ${i === speedIdx ? "bg-white text-black" : "border border-white/20 text-white/70 hover:border-white/60"}`}>
              ×{fmtInt(sp, es)}
            </button>
          ))}
          <button onClick={() => { chars.current = total; force((x) => x + 1); }} className="shrink-0 rounded-full border border-white/20 px-3 py-1.5 font-mono text-xs text-white/70 hover:border-white/60" aria-label="end">⏭</button>
        </div>
        <input type="range" min={0} max={total} step={1} value={c} onChange={(e) => { chars.current = +e.target.value; force((x) => x + 1); }} className="w-full accent-white md:flex-1" />
        <span className="hidden font-mono text-xs tabular-nums text-white/60 md:block md:w-40 md:text-right">{fmtDuration(seconds, es)}</span>
      </footer>

      {examsOpen && (
        <Modal onClose={() => setExamsOpen(false)} lang={lang} title={`${t(T.exams)} · ${fb.title}`}>
          <dl className="mb-4 space-y-1.5 text-sm">
            <Row k={t(T.study)} v={fmtDuration(seconds, es)} />
            <Row k={t(T.surprise)} v={`${at(fb, c).bpc.toFixed(2)} bits`} />
            <Row k={t(T.meter)} v={`${pct(run, at(fb, c).bpc).toFixed(1)}%`} />
          </dl>
          {examPanel}
          <p className="label mb-2 mt-5 text-white/50">{t(T.curve)}</p>
          <Curve run={run} chars={c} focus={focus} lang={lang} />
        </Modal>
      )}
      {about && <About lang={lang} run={run} onClose={() => setAbout(false)} />}
    </div>
  );
}

function Row({ k, v }: { k: string; v: string }) {
  return <div className="flex justify-between gap-3"><dt className="text-white/60">{k}</dt><dd className="font-mono tabular-nums">{v}</dd></div>;
}

/* ---------------- exámenes ---------------- */
function ExamPanel({ run, book, chars, lang }: { run: Run; book: Book; chars: number; lang: Lang }) {
  const es = lang === "es";
  const { cp, bpc } = at(book, chars);
  const cps = run.meta.charsPerSecond;
  const when = (ok: (c: Checkpoint) => boolean) => {
    const w = firstWhen(book, ok);
    return w === null ? null : w;
  };
  const tag = (w: number | null) =>
    w === null || chars < w
      ? <span className="text-white/40">{T.notYet[lang]}</span>
      : <span className="text-emerald-300">✓ {T.at[lang]} {fmtDuration(w / cps, es)}</span>;

  const wWords = when((c) => (c.words ?? 0) >= 0.5);
  const wRecog = when((c) => (c.recog ?? 0) >= 0.9);
  const wLine = when((c) => (c.lineBits ?? 99) <= Math.log2(1000));
  const attempts = cp.lineBits != null ? Math.pow(2, cp.lineBits) : null;
  const len = book.workLen ?? 0;
  const yrs = len ? log10YearsToWrite(bpc, len, cps) : null;
  const yrsRandom = len ? log10YearsToWrite(run.meta.bpcRandom, len, cps) : null;
  const famous = book.famous;

  return (
    <ol className="space-y-3 text-sm">
      <Exam n="1" title={es ? "¿Escribe palabras reales?" : "Does it write real words?"} status={tag(wWords)}>
        {es ? "De lo que escribe ahora, " : "Of what it writes now, "}<b className="text-white">{Math.round((cp.words ?? 0) * 100)}%</b>{es ? " son palabras que existen en el libro." : " are words that exist in the book."}
      </Exam>
      <Exam n="2" title={es ? "¿Reconoce a su autor?" : "Does it recognise its author?"} status={tag(wRecog)}>
        {es ? "Ante dos fragmentos, elige el de su autor el " : "Given two passages, it picks its author's "}<b className="text-white">{Math.round((cp.recog ?? 0) * 100)}%</b>{es ? " de las veces (al azar: 50%)." : " of the time (by chance: 50%)."}
      </Exam>
      {famous && (
        <Exam n="3" title={es ? "¿Completa la frase famosa?" : "Can it finish the famous line?"} status={tag(wLine)}>
          <span className="font-serif italic text-white/85">“{famous.prefix}<u>{famous.completion}</u>”</span>
          <br />{es ? "Necesitaría unos " : "It would need about "}<b className="text-white">{attempts == null ? "—" : fmtBig(attempts, es)}</b>{es ? " intentos." : " tries."}
        </Exam>
      )}
      {yrs != null && yrsRandom != null && (
        <Exam n="4" title={es ? `¿Escribe ${book.work} de cero?` : `Can it write ${book.work} from scratch?`} status={<span className="text-white/40">{T.never[lang]}*</span>}>
          {es ? "Letra por letra, sin un error, tardaría unos " : "Letter by letter, without a single mistake, it would take about "}<b className="text-white">10<sup>{fmtInt(yrs, es)}</sup></b>{es ? " años" : " years"}
          {" "}<span className="text-white/45">({es ? "al azar: " : "at random: "}10<sup>{fmtInt(yrsRandom, es)}</sup>)</span>.
          <br /><span className="text-white/45">{es ? `El universo tiene 10^${LOG10_UNIVERSE.toFixed(1)} años.` : `The universe is 10^${LOG10_UNIVERSE.toFixed(1)} years old.`}</span>
        </Exam>
      )}
      <Exam n="5" title={es ? "¿Puede resumirlo?" : "Can it summarise it?"} status={<span className="text-white/40">{T.never[lang]}</span>}>
        {es
          ? "Solo aprende qué letra suele venir después. Para resumir hay que entender el significado, y ni todo su cerebro alcanza para eso."
          : "It only learns which letter tends to come next. Summarising needs meaning, and its whole brain isn't enough for that."}
      </Exam>
    </ol>
  );
}

function Exam({ n, title, status, children }: { n: string; title: string; status: ReactNode; children: ReactNode }) {
  return (
    <li className="grid grid-cols-[1.25rem_1fr] gap-x-2">
      <span className="font-mono text-xs text-white/40">{n}</span>
      <div>
        <div className="flex flex-wrap items-baseline justify-between gap-x-2"><span className="font-medium">{title}</span><span className="text-xs">{status}</span></div>
        <p className="mt-0.5 leading-snug text-white/65">{children}</p>
      </div>
    </li>
  );
}

function fmtBig(x: number, es: boolean) {
  if (x < 1e6) return fmtInt(Math.max(1, x), es);
  const e = Math.floor(Math.log10(x));
  return `${(x / 10 ** e).toFixed(1).replace(".", es ? "," : ".")}×10^${e}`;
}

/* ---------------- curva de aprendizaje ---------------- */
function Curve({ run, chars, focus, lang }: { run: Run; chars: number; focus: number; lang: Lang }) {
  const W = 370, H = 120, P = 6;
  const total = totalChars(run) || 1;
  const all = run.books.flatMap((b) => b.checkpoints.map((c) => c.bpc)).concat((run.controls ?? []).flatMap((c) => c.checkpoints.map((k) => k.bpc)));
  const lo = Math.min(...all) - 0.1, hi = Math.max(...all) + 0.05;
  const x = (v: number) => P + (v / total) * (W - 2 * P);
  const yy = (v: number) => P + ((hi - v) / (hi - lo)) * (H - 2 * P);
  const path = (cps: { chars: number; bpc: number }[]) => cps.map((c, i) => `${i ? "L" : "M"}${x(c.chars).toFixed(1)},${yy(c.bpc).toFixed(1)}`).join("");
  return (
    <svg viewBox={`0 0 ${W} ${H + 18}`} className="w-full">
      {[hi, (hi + lo) / 2, lo].map((v) => (
        <g key={v}>
          <line x1={P} x2={W - P} y1={yy(v)} y2={yy(v)} stroke="rgba(255,255,255,0.08)" />
          <text x={P + 2} y={yy(v) - 3} textAnchor="start" fontSize="9" fill="rgba(255,255,255,0.4)" fontFamily="Geist Mono">{v.toFixed(1)}</text>
        </g>
      ))}
      {(run.controls ?? []).map((c) => <path key={c.id} d={path(c.checkpoints)} fill="none" stroke="rgba(255,255,255,0.45)" strokeDasharray="3 3" strokeWidth="1.2" />)}
      {run.books.map((b, i) => (
        <path key={b.id} d={path(b.checkpoints)} fill="none" stroke={COLORS[b.id] ?? "#fff"} strokeWidth={i === focus ? 2.2 : 1.2} opacity={i === focus ? 1 : 0.55} />
      ))}
      <line x1={x(chars)} x2={x(chars)} y1={P} y2={H - P} stroke="rgba(255,255,255,0.5)" />
      <text x={P} y={H + 12} fontSize="9" fill="rgba(255,255,255,0.45)" fontFamily="Geist Mono">
        {lang === "es" ? "menos = aprendió más · línea punteada: " : "lower = learned more · dashed: "}{T.shuffled[lang]}
      </text>
    </svg>
  );
}

/* ---------------- modales ---------------- */
function Modal({ children, onClose, lang, title }: { children: ReactNode; onClose: () => void; lang: Lang; title: string }) {
  return (
    <div className="absolute inset-0 z-20 flex items-center justify-center bg-black/70 p-4 backdrop-blur-sm" onClick={onClose}>
      <div className="max-h-[85svh] w-full max-w-xl overflow-y-auto rounded-3xl border border-white/15 bg-[#111] p-6 md:p-8" onClick={(e) => e.stopPropagation()}>
        <h2 className="mb-4 text-2xl font-medium tracking-tight">{title}</h2>
        {children}
        <button onClick={onClose} className="label mt-6 rounded-full border border-white/25 px-4 py-2 hover:bg-white hover:text-black">{T.close[lang]}</button>
      </div>
    </div>
  );
}

function About({ lang, run, onClose }: { lang: Lang; run: Run; onClose: () => void }) {
  const es = lang === "es";
  const m = run.meta;
  const n = (v: number | undefined) => fmtInt(v ?? 0, es);
  return (
    <Modal onClose={onClose} lang={lang} title={T.about[lang]}>
      <div className="space-y-3 text-[15px] leading-relaxed text-white/75">
        {es ? (
          <>
            <p>Usamos el <b>cableado real</b> del cerebro de una mosca de la fruta, mapeado por el proyecto FlyWire: {n(m.neurons)} neuronas, {n(m.connections)} conexiones y {n(m.synapses)} sinapsis. Cada conexión pesa según cuántas sinapsis tiene, y excita o inhibe según su neurotransmisor.</p>
            <p><b>Cómo lee:</b> cada letra activa un grupo de sus {n(m.inputs)} neuronas olfativas (la mosca "huele" las letras, el camino principal hacia su centro de memoria). La actividad recorre el cerebro capa por capa, así que las capas profundas recuerdan las letras anteriores.</p>
            <p><b>Cómo aprende:</b> como en una mosca real, el cableado no cambia. Lo único que se ajusta son las conexiones que salen de sus {n(m.kenyon)} células de Kenyon, en el cuerpo fungiforme, donde las moscas guardan lo que aprenden.</p>
            <p><b>Cómo lo medimos:</b> en textos del libro que nunca leyó, cuánto la sorprende cada letra. 0% es teclear al azar y 100% es lo que logra una persona (~1 bit por letra). La línea punteada es el mismo cerebro con los cables mezclados al azar.</p>
            <p><b>Lo que no es:</b> una mosca real no lee. Esto mide qué tan bien el cableado de un cerebro de mosca puede aprender texto. El experimento corrió una vez y la página lo reproduce; el reloj supone que lee a ritmo humano ({m.charsPerSecond} letras por segundo).</p>
            <p className="text-white/45">Datos: FlyWire FAFB v783 (Dorkenwald et al., Schlegel et al., Nature 2024). Libros: Project Gutenberg. * "Nunca" en la práctica: el número de años es finito, pero supera por mucho la edad del universo.</p>
          </>
        ) : (
          <>
            <p>We use the <b>real wiring</b> of a fruit-fly brain, mapped by the FlyWire project: {n(m.neurons)} neurons, {n(m.connections)} connections and {n(m.synapses)} synapses. Each connection is weighted by its number of synapses and excites or inhibits according to its neurotransmitter.</p>
            <p><b>How it reads:</b> each letter activates a group of its {n(m.inputs)} olfactory neurons (the fly "smells" the letters, the main route to its memory centre). Activity flows through the brain layer by layer, so deeper layers remember the previous letters.</p>
            <p><b>How it learns:</b> as in a real fly, the wiring never changes. Only the connections leaving its {n(m.kenyon)} Kenyon cells are adjusted, in the mushroom body, where flies store what they learn.</p>
            <p><b>How we measure it:</b> on passages of the book it never read, how surprised it is by each letter. 0% is random typing and 100% is what a person achieves (~1 bit per letter). The dashed line is the same brain with its wires shuffled at random.</p>
            <p><b>What it isn't:</b> a real fly doesn't read. This measures how well the wiring of a fly brain can learn text. The experiment ran once and the page replays it; the clock assumes human reading speed ({m.charsPerSecond} letters per second).</p>
            <p className="text-white/45">Data: FlyWire FAFB v783 (Dorkenwald et al., Schlegel et al., Nature 2024). Books: Project Gutenberg. * "Never" in practice: the number of years is finite, but far beyond the age of the universe.</p>
          </>
        )}
      </div>
    </Modal>
  );
}
