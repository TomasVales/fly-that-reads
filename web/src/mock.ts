import type { Book, Run } from "./data";

/**
 * DATOS DE PRUEBA para diseñar la página mientras corre el experimento real.
 * Nada de esto es un resultado: la página lo marca con una etiqueta "demo".
 */
const EXCERPTS: Record<string, string> = {
  shakespeare:
    "To be, or not to be, that is the question: Whether 'tis nobler in the mind to suffer The slings and arrows of outrageous fortune, Or to take arms against a sea of troubles And by opposing end them. To die, to sleep; No more; and by a sleep to say we end The heart-ache and the thousand natural shocks That flesh is heir to: 'tis a consummation Devoutly to be wish'd. To die, to sleep; To sleep, perchance to dream—ay, there's the rub: For in that sleep of death what dreams may come, When we have shuffled off this mortal coil, Must give us pause.",
  quijote:
    "En un lugar de la Mancha, de cuyo nombre no quiero acordarme, no ha mucho tiempo que vivía un hidalgo de los de lanza en astillero, adarga antigua, rocín flaco y galgo corredor. Una olla de algo más vaca que carnero, salpicón las más noches, duelos y quebrantos los sábados, lantejas los viernes, algún palomino de añadidura los domingos, consumían las tres partes de su hacienda.",
  fierro:
    "Aquí me pongo a cantar al compás de la vigüela, que el hombre que lo desvela una pena estrordinaria, como la ave solitaria con el cantar se consuela. Pido a los santos del cielo que ayuden mi pensamiento; les pido en este momento que voy a cantar mi historia me refresquen la memoria y aclaren mi entendimiento.",
  dante:
    "Nel mezzo del cammin di nostra vita mi ritrovai per una selva oscura, ché la diritta via era smarrita. Ahi quanto a dir qual era è cosa dura esta selva selvaggia e aspra e forte che nel pensier rinova la paura! Tant'è amara che poco è più morte; ma per trattar del ben ch'i' vi trovai, dirò de l'altre cose ch'i' v'ho scorte.",
};

function rng(seed: number) {
  return () => ((seed = (seed * 16807) % 2147483647) / 2147483647);
}

/** nube de puntos con forma de cerebro de mosca: cerebro central + dos lóbulos ópticos */
function brainCloud(n: number) {
  const r = rng(11);
  const out: number[] = [];
  const gauss = () => { let s = 0; for (let i = 0; i < 4; i++) s += r(); return (s - 2) / 1.1; };
  for (let i = 0; i < n; i++) {
    const part = r();
    let x: number, y: number, z: number;
    if (part < 0.42) { x = gauss() * 0.28; y = gauss() * 0.32; z = gauss() * 0.3; }
    else {
      const s = part < 0.71 ? -1 : 1;
      const a = r() * Math.PI * 2, b = (r() - 0.5) * Math.PI;
      const rad = 0.75 + r() * 0.25;
      x = s * (0.62 + Math.cos(b) * Math.abs(Math.cos(a)) * 0.28 * rad);
      y = Math.sin(b) * 0.55 * rad;
      z = Math.cos(b) * Math.sin(a) * 0.3 * rad;
    }
    out.push(x, y, z);
  }
  return out;
}

function scramble(text: string, quality: number, r: () => number) {
  // mezcla texto real con letras al azar según la "calidad" (solo para la demo)
  const abc = "abcdefghijklmnopqrstuvwxyz      ";
  let s = "";
  for (const ch of text.slice(0, 64)) s += r() < quality ? ch : abc[Math.floor(r() * abc.length)];
  return s;
}

export function mockRun(): Run {
  const r = rng(7);
  const books: [string, string, string, Book["lang"], number][] = [
    ["shakespeare", "Shakespeare", "William Shakespeare", "en", 2.15],
    ["quijote", "Don Quijote", "Miguel de Cervantes", "es", 2.3],
    ["fierro", "Martín Fierro", "José Hernández", "es", 2.45],
    ["dante", "Divina Commedia", "Dante Alighieri", "it", 2.55],
  ];
  const N = 400_000;
  const steps = 40;
  return {
    meta: { demo: true, neurons: 139255, synapses: 3732460, bpcRandom: 5.7, bpcHuman: 1.0, target: 50, charsPerSecond: 15 },
    brain: { positions: brainCloud(5000) },
    books: books.map(([id, title, author, lang, floor]) => ({
      id, title, author, lang,
      excerpt: EXCERPTS[id], work: title, workLen: 175000,
      famous: { prefix: "to be, or not to be, ", completion: "that is the question" },
      checkpoints: Array.from({ length: steps + 1 }, (_, k) => {
        const chars = Math.round((k / steps) ** 2 * N);
        const bpc = floor + (5.7 - floor) * Math.exp(-chars / (N * 0.18));
        const q = (5.7 - bpc) / 4.7;
        const active = Array.from({ length: 160 }, () => Math.floor(r() * 5000));
        return { chars, bpc, sample: scramble(EXCERPTS[id], q, r), active, words: q * 0.9, recog: 0.5 + q * 0.5, lineBits: 90 - q * 70 };
      }),
    })),
  };
}
