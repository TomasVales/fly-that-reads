/** Formato del experimento (lo genera el pipeline en Python: pipeline/run.py) */
export type Checkpoint = {
  /** letras leídas hasta este punto */
  chars: number;
  /** bits de sorpresa por letra en texto que nunca vio (menos = mejor) */
  bpc: number;
  /** lo que escribe la mosca en este punto, partiendo de un comienzo del libro */
  sample: string;
  /** índices (en brain.positions) de neuronas más activas mientras leía */
  active: number[];
  /** fracción de palabras escritas que existen en el libro */
  words?: number;
  /** % de veces que distingue un fragmento de su autor de uno de otro (azar = 0.5) */
  recog?: number;
  /** bits para completar la frase famosa (intentos ≈ 2^lineBits) */
  lineBits?: number;
};

export type Book = {
  id: string;
  title: string;
  author: string;
  lang: "en" | "es" | "it";
  /** fragmento real del libro para dibujar las páginas */
  excerpt: string;
  checkpoints: Checkpoint[];
  /** veces que relee el libro */
  reads?: number;
  /** obra a escribir de cero y su largo en letras */
  work?: string;
  workLen?: number;
  famous?: { prefix: string; completion: string };
};

export type Run = {
  meta: {
    /** true mientras se muestran datos inventados para probar la página */
    demo: boolean;
    neurons: number;
    synapses: number;
    /** bits/letra de teclear al azar (0 %) y de una persona (100 %) */
    bpcRandom: number;
    bpcHuman: number;
    /** % de la escala que cuenta como "escribe como su autor" */
    target: number;
    /** letras por segundo a ritmo de lectura humano (para el reloj) */
    charsPerSecond: number;
    note?: string;
    done?: boolean;
    progress?: number;
    kenyon?: number;
    inputs?: number;
    connections?: number;
    groups?: { inputs?: [number, number]; kenyon?: [number, number] };
  };
  brain: { positions: number[] };
  books: Book[];
  /** controles: mismo experimento con cableado al azar e IA chica (opcional) */
  controls?: { id: string; label: string; checkpoints: Pick<Checkpoint, "chars" | "bpc">[] }[];
};

export const pct = (run: Run, bpc: number) =>
  Math.max(0, Math.min(100, ((run.meta.bpcRandom - bpc) / (run.meta.bpcRandom - run.meta.bpcHuman)) * 100));

/** estado de un libro en un momento (letras leídas) */
export function at(book: Book, chars: number) {
  const cps = book.checkpoints;
  let i = 0;
  while (i < cps.length - 1 && cps[i + 1].chars <= chars) i++;
  const a = cps[i], b = cps[Math.min(i + 1, cps.length - 1)];
  const f = b.chars === a.chars ? 0 : Math.max(0, Math.min(1, (chars - a.chars) / (b.chars - a.chars)));
  return { bpc: a.bpc + (b.bpc - a.bpc) * f, cp: a, index: i };
}

export const totalChars = (run: Run) => Math.max(...run.books.map((b) => b.checkpoints[b.checkpoints.length - 1].chars));

export function fmtDuration(sec: number, es: boolean) {
  const d = Math.floor(sec / 86400), h = Math.floor((sec % 86400) / 3600), m = Math.floor((sec % 3600) / 60), s = Math.floor(sec % 60);
  if (d) return `${d} ${es ? (d === 1 ? "día" : "días") : d === 1 ? "day" : "days"} ${h} h`;
  if (h) return `${h} h ${m} min`;
  if (m) return `${m} min ${s} s`;
  return `${s} s`;
}

export const fmtInt = (n: number, es: boolean) => Math.round(n).toLocaleString(es ? "es-AR" : "en-US");

/** log10 de los años que tardaría en escribir `len` letras exactas, con `bpc` bits de sorpresa por letra */
export function log10YearsToWrite(bpc: number, len: number, cps: number) {
  const attempts = bpc * len * Math.LOG10E * Math.LN2; // log10(2^(bpc·len))
  return attempts + Math.log10(len / cps / 31_557_600);
}
export const LOG10_UNIVERSE = Math.log10(13.8e9);

/** primer momento (letras leídas) en que se cumple una condición */
export function firstWhen(book: Book, ok: (c: Checkpoint) => boolean) {
  const c = book.checkpoints.find(ok);
  return c ? c.chars : null;
}
