# The fly that reads · La mosca que lee

**🇬🇧 [English](#english) · 🇦🇷 [Español](#español)**

---

## English

What happens when the **real wiring of a fruit-fly brain** reads Shakespeare, Cervantes, Hernández and Dante?

This project takes the complete connectome of an adult *Drosophila* brain (FlyWire FAFB v783: ~139,000 neurons, ~3.7 million connections, ~50 million synapses) and lets it read four classics letter by letter. An interactive 3D page replays the experiment: a fly with reading glasses turns the pages while its brain lights up, four books race each other, and you can speed time up from ×1 to ×10,000.

### How it works

- **Wiring:** every real connection is used as-is. Its weight grows with the number of synapses and its sign comes from the predicted neurotransmitter (GABA and glutamate inhibit, the rest excite).
- **Reading:** each letter activates a group of olfactory receptor neurons, the main route into the fly's memory centre. The letter is "smelled" for a few time steps, so activity travels layer by layer and deeper layers keep a trace of previous letters.
- **Learning:** the wiring never changes. Only the connections leaving the 5,177 **Kenyon cells** of the mushroom body are trained (a linear softmax readout with Adam), just like in real flies, where learning happens at Kenyon-cell output synapses.
- **Control:** the same brain with its connections shuffled at random reads Shakespeare in parallel.
- **Exams**, on passages the fly never read:
  - surprise per letter (bits; 0% = random typing, 100% ≈ a person at ~1 bit/letter)
  - how many of the words it writes are real words from the book
  - whether it can tell its author apart from the other three
  - how many tries it would need to finish the book's most famous line
  - how long it would take to write the work exactly, from scratch (spoiler: longer than the universe)

### Run it

```bash
python3 -m venv .venv && .venv/bin/pip install numpy scipy pandas torch
# put the FlyWire FAFB v783 files in data/ (connections, classification, coordinates, neurotransmitters)
# put the Project Gutenberg books in books/ (ids 100, 2000, 14765, 1012)
.venv/bin/python pipeline/run.py --train 100000     # writes web/public/data/run.json
cd web && npm install && npm run dev
```

The experiment runs on the Apple GPU (Metal) through PyTorch when available, otherwise on the CPU.

### Caveats

A real fly doesn't read. This measures how well the wiring of a fly brain, used as a fixed recurrent network, can learn the statistics of text. The input encoding, time steps and readout are modelling choices.

---

## Español

¿Qué pasa si el **cableado real del cerebro de una mosca de la fruta** lee a Shakespeare, Cervantes, Hernández y Dante?

El proyecto toma el conectoma completo del cerebro de una *Drosophila* adulta (FlyWire FAFB v783: ~139.000 neuronas, ~3,7 millones de conexiones, ~50 millones de sinapsis) y lo pone a leer cuatro clásicos, letra por letra. Una página 3D interactiva reproduce el experimento: una mosca con anteojos pasa las páginas mientras su cerebro se ilumina, los cuatro libros compiten en una carrera y podés acelerar el tiempo de ×1 a ×10.000.

### Cómo funciona

- **Cableado:** se usa cada conexión real tal cual. Su peso crece con la cantidad de sinapsis y su signo sale del neurotransmisor predicho (GABA y glutamato inhiben, el resto excita).
- **Lectura:** cada letra activa un grupo de neuronas receptoras olfativas, el camino principal hacia el centro de memoria de la mosca. La letra se "huele" durante algunos pasos, así que la actividad viaja capa por capa y las capas profundas guardan el rastro de las letras anteriores.
- **Aprendizaje:** el cableado nunca cambia. Solo se entrenan las conexiones que salen de las 5.177 **células de Kenyon** del cuerpo fungiforme (una lectura lineal softmax con Adam), igual que en las moscas reales.
- **Control:** el mismo cerebro con las conexiones mezcladas al azar lee Shakespeare en paralelo.
- **Exámenes**, sobre partes del libro que nunca leyó:
  - sorpresa por letra (bits; 0% = teclear al azar, 100% ≈ una persona, ~1 bit por letra)
  - qué parte de lo que escribe son palabras reales del libro
  - si distingue a su autor de los otros tres
  - cuántos intentos necesitaría para completar la frase más famosa del libro
  - cuánto tardaría en escribir la obra exacta desde cero (spoiler: más que la edad del universo)

### Advertencias

Una mosca real no lee. Esto mide qué tan bien el cableado de un cerebro de mosca, usado como red recurrente fija, puede aprender las estadísticas de un texto. La forma de codificar las letras, los pasos de tiempo y la lectura de salida son decisiones de modelado.

---

### Data & credits

- **Connectome:** FlyWire, FAFB v783. Please cite:
  - Dorkenwald, S. et al. *Neuronal wiring diagram of an adult brain.* Nature 634, 124–138 (2024).
  - Schlegel, P. et al. *Whole-brain annotation and multi-connectome cell typing of Drosophila.* Nature 634, 139–152 (2024).
  - Eckstein, N. et al. *Neurotransmitter classification from electron microscopy images at synaptic sites in Drosophila melanogaster.* Cell 187, 2574–2594 (2024).
- **Books:** Project Gutenberg (public domain): *The Complete Works of William Shakespeare* (#100), *Don Quijote* (#2000), *El Gaucho Martín Fierro* (#14765), *La Divina Commedia* (#1012).

Made by [Tomas Vales](https://tomasvales.github.io/).
