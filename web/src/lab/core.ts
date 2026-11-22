/**
 * El cerebro de la mosca corriendo en el navegador: misma cuenta que pipeline/live.py.
 * Cada letra activa un grupo fijo de neuronas olfativas; la actividad recorre el cableado real
 * (x ← tanh(W·x + entrada)) durante `ticks` pasos; las células de Kenyon deciden la próxima letra.
 */

export type Header = {
  N: number; nnz: number; V: number; vocab: string; ticks: number; gain: number; scale: number;
  books: string[]; groups: string[]; inputsPerLetter: number; nKC: number;
  ref: { text: string; kc: number[]; p: number[] };
  arrays: Record<string, [number, string, number, number?]>;
};

const TYPES: Record<string, (b: ArrayBuffer, o: number, n: number) => ArrayLike<number>> = {
  u4: (b, o, n) => new Uint32Array(b, o, n),
  i2: (b, o, n) => new Int16Array(b, o, n),
  i1: (b, o, n) => new Int8Array(b, o, n),
  u1: (b, o, n) => new Uint8Array(b, o, n),
  f4: (b, o, n) => new Float32Array(b, o, n),
};

export function parse(buf: ArrayBuffer) {
  const hl = new DataView(buf).getUint32(0, true);
  const header = JSON.parse(new TextDecoder().decode(new Uint8Array(buf, 4, hl))) as Header;
  const base = 4 + hl;
  const SIZE: Record<string, number> = { u4: 4, i2: 2, i1: 1, u1: 1, f4: 4 };
  const get = <T,>(k: string) => {
    const [o, t, n, shuf] = header.arrays[k];
    if (!shuf) return TYPES[t](buf, base + o, n) as unknown as T;
    // bytes guardados por planos (todos los primeros bytes, después los segundos…) → reordenar
    const sz = SIZE[t], src = new Uint8Array(buf, base + o, n * sz), dst = new Uint8Array(n * sz);
    for (let b = 0; b < sz; b++) for (let i = 0, off = b * n; i < n; i++) dst[i * sz + b] = src[off + i];
    return TYPES[t](dst.buffer, 0, n) as unknown as T;
  };
  return { header, get };
}

export const PUNCT = " .,;:!?'-";
const FOLD: Record<string, string> = { "â": "a", "ä": "a", "æ": "ae", "ê": "e", "ë": "e", "î": "i", "ï": "i", "ö": "o", "ô": "o", "œ": "oe", "ç": "c", "û": "u" };

/** igual que books.normalize + quedarse solo con letras del alfabeto de la mosca */
export function normalize(s: string, vocab: string) {
  let t = s.normalize("NFC").toLowerCase();
  t = [...t].map((c) => FOLD[c] ?? c).join("");
  t = t.replace(/[’‘]/g, "'").replace(/[—–]/g, "-").replace(/\s/g, " ");
  t = [...t].filter((c) => vocab.includes(c)).join("");
  return t.replace(/ +/g, " ");
}

export class Brain {
  h: Header;
  N: number;
  V: number;
  rowptr: Uint32Array;
  cols: Uint32Array;
  vals: Float32Array;
  win: Uint32Array;
  kc: Uint32Array;
  pos: Int16Array;
  flags: Uint8Array;
  heads: { w: Int16Array; s: Float32Array; mu: Float32Array; sd: Float32Array }[];
  x: Float32Array;
  y: Float32Array;
  u: Float32Array;
  keep: Float32Array;
  z: Float32Array;

  constructor(buf: ArrayBuffer) {
    const { header, get } = parse(buf);
    this.h = header;
    this.N = header.N;
    this.V = header.V;
    this.rowptr = get<Uint32Array>("rowptr");
    // columnas guardadas como diferencias dentro de cada fila → absolutas
    const d = get<Uint32Array>("cols");
    this.cols = new Uint32Array(d.length);
    for (let i = 0; i < this.N; i++) {
      let c = 0;
      for (let e = this.rowptr[i]; e < this.rowptr[i + 1]; e++) {
        c = e === this.rowptr[i] ? d[e] : c + d[e];
        this.cols[e] = c;
      }
    }
    // peso = signo × log(1 + sinapsis) × escala (igual que en Python)
    const cnt = get<Int16Array>("cnt");
    this.vals = new Float32Array(cnt.length);
    for (let e = 0; e < cnt.length; e++) this.vals[e] = Math.sign(cnt[e]) * Math.log1p(Math.abs(cnt[e])) * header.scale;
    this.win = get<Uint32Array>("win");
    this.kc = get<Uint32Array>("kc");
    this.pos = get<Int16Array>("pos");
    this.flags = get<Uint8Array>("flags");
    this.heads = header.books.map((b) => ({ w: get<Int16Array>(`w_${b}`), s: get<Float32Array>(`s_${b}`), mu: get<Float32Array>(`mu_${b}`), sd: get<Float32Array>(`sd_${b}`) }));
    this.x = new Float32Array(this.N);
    this.y = new Float32Array(this.N);
    this.u = new Float32Array(this.N);
    this.keep = new Float32Array(this.N).fill(1);
    this.z = new Float32Array(header.nKC);
  }

  reset() { this.x.fill(0); }

  /** apaga los grupos lesionados (bit 2+i de flags) */
  setLesions(names: string[]) {
    const bits = names.map((n) => 1 << (2 + this.h.groups.indexOf(n))).reduce((a, b) => a | b, 0);
    for (let i = 0; i < this.N; i++) this.keep[i] = this.flags[i] & bits ? 0 : 1;
  }

  /** la mosca "huele" una letra */
  step(c: number) {
    const { N, rowptr, cols, vals, keep, u } = this;
    const m = this.h.inputsPerLetter, g = this.h.gain;
    for (let k = 0; k < m; k++) u[this.win[c * m + k]] = g;
    for (let t = 0; t < this.h.ticks; t++) {
      const x = this.x, y = this.y;
      for (let i = 0; i < N; i++) {
        let s = u[i];
        for (let e = rowptr[i], end = rowptr[i + 1]; e < end; e++) s += vals[e] * x[cols[e]];
        y[i] = Math.tanh(s) * keep[i];
      }
      this.x = y;
      this.y = x;
    }
    for (let k = 0; k < m; k++) u[this.win[c * m + k]] = 0;
  }

  /** logits de la próxima letra según lo que aprendió con el libro `b` */
  logits(b: number, out = new Float32Array(this.V)) {
    const hd = this.heads[b], V = this.V, nK = this.h.nKC, z = this.z;
    for (let i = 0; i < nK; i++) {
      const v = (this.x[this.kc[i]] - hd.mu[i]) / hd.sd[i];
      z[i] = v > 5 ? 5 : v < -5 ? -5 : v;
    }
    out.fill(0);
    for (let i = 0; i < nK; i++) {
      const zi = z[i];
      if (zi === 0) continue;
      const row = i * V;
      for (let v = 0; v < V; v++) out[v] += zi * hd.w[row + v];
    }
    for (let v = 0; v < V; v++) out[v] = (out[v] + hd.w[nK * V + v]) * hd.s[v];
    return out;
  }
}

export function softmax(l: ArrayLike<number>, temp = 1) {
  let mx = -Infinity;
  for (let i = 0; i < l.length; i++) mx = Math.max(mx, l[i]);
  const p = new Float32Array(l.length);
  let s = 0;
  for (let i = 0; i < l.length; i++) s += p[i] = Math.exp((l[i] - mx) / temp);
  for (let i = 0; i < l.length; i++) p[i] /= s;
  return p;
}

export function sample(p: Float32Array, r = Math.random()) {
  let acc = 0;
  for (let i = 0; i < p.length; i++) { acc += p[i]; if (r < acc) return i; }
  return p.length - 1;
}
