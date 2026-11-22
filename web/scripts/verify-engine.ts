// Compara el motor del navegador con los números de Python (header.ref) y mide su velocidad.
import { readFileSync } from "node:fs";
import { gunzipSync } from "node:zlib";
import { Brain, normalize, softmax } from "../src/lab/core.ts";

const raw = gunzipSync(readFileSync(new URL("../public/lab/brain.bin", import.meta.url)));
const buf = raw.buffer.slice(raw.byteOffset, raw.byteOffset + raw.byteLength);
let t = performance.now();
const b = new Brain(buf);
console.log(`cargado en ${(performance.now() - t).toFixed(0)} ms · ${b.N} neuronas · ${b.h.nnz} conexiones`);
const ids = [...normalize(b.h.ref.text, b.h.vocab)].map((c) => b.h.vocab.indexOf(c));
t = performance.now();
for (const c of ids) b.step(c);
const ms = (performance.now() - t) / ids.length;
let dk = 0;
b.h.ref.kc.forEach((v, i) => (dk = Math.max(dk, Math.abs(v - b.x[b.kc[i]]))));
const p = softmax(b.logits(0));
let dp = 0;
b.h.ref.p.forEach((v, i) => (dp = Math.max(dp, Math.abs(v - p[i]))));
console.log(`${ms.toFixed(1)} ms por letra (${(1000 / ms).toFixed(0)} letras/s) · diferencia Kenyon ${dk.toExponential(1)} · diferencia prob ${dp.toExponential(1)}`);
