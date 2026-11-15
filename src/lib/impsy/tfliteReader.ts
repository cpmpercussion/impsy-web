// TFLite reader — recovers MDRNN weights from an IMPSY `.tflite` file, the
// inverse of tfliteWriter.ts. Reads both graph shapes in circulation: models
// from `tf.lite.TFLiteConverter` (each LSTM layer inside a WHILE subgraph) and
// models from tfliteWriter.ts (cells inline in the main subgraph).
//
// Weights are found by graph structure, not tensor names (the converter names
// most constants `arith.constantN`): each LSTM cell is the 4-way SPLIT of the
// summed input and recurrent projections, and the MDN head is the 3-input
// CONCATENATION that produces a graph output.

import { ByteBuffer } from "flatbuffers";
import { validateWeights, type DenseWeights, type MDRNNWeights } from "./mdrnnWeights";
import {
  BuiltinOperator,
  BuiltinOptions,
  Slot,
  TFLITE_IDENTIFIER,
  TensorType,
} from "./tfliteSchema";

export interface TfliteTensor {
  name: string;
  shape: number[];
  type: number;
  /** Constant data (a view into the file), or null for inputs and computed tensors. */
  data: Uint8Array | null;
}

export interface TfliteOperator {
  /** `BuiltinOperator` code. */
  code: number;
  inputs: number[];
  outputs: number[];
  /** For WHILE ops, the index of the body subgraph. */
  bodySubgraph?: number;
}

export interface TfliteSubgraph {
  name: string;
  tensors: TfliteTensor[];
  inputs: number[];
  outputs: number[];
  operators: TfliteOperator[];
}

export interface TfliteModel {
  version: number;
  description: string;
  subgraphs: TfliteSubgraph[];
}

/** Parse the graph structure of a `.tflite` file (the parts tfliteSchema.ts covers). */
export function parseTflite(bytes: Uint8Array): TfliteModel {
  const bb = new ByteBuffer(bytes);
  const identifier = String.fromCharCode(...bytes.subarray(4, 8));
  if (bytes.length < 8 || identifier !== TFLITE_IDENTIFIER) {
    throw new Error("Not a TFLite model (missing TFL3 file identifier)");
  }

  // FlatBuffer accessors: a table field's position (0 if absent), then typed reads.
  const field = (table: number, slot: number): number => {
    const o = bb.__offset(table, 4 + 2 * slot);
    return o ? table + o : 0;
  };
  const int32 = (table: number, slot: number, fallback = 0): number => {
    const p = field(table, slot);
    return p ? bb.readInt32(p) : fallback;
  };
  const int8 = (table: number, slot: number): number => {
    const p = field(table, slot);
    return p ? bb.readInt8(p) : 0;
  };
  const string = (table: number, slot: number): string => {
    const p = field(table, slot);
    return p ? (bb.__string(p) as string) : "";
  };
  const int32s = (table: number, slot: number): number[] => {
    const p = field(table, slot);
    if (!p) return [];
    const start = bb.__vector(p);
    return Array.from({ length: bb.__vector_len(p) }, (_, i) => bb.readInt32(start + i * 4));
  };
  const tables = (table: number, slot: number): number[] => {
    const p = field(table, slot);
    if (!p) return [];
    const start = bb.__vector(p);
    return Array.from({ length: bb.__vector_len(p) }, (_, i) => bb.__indirect(start + i * 4));
  };
  const table = (parent: number, slot: number): number => {
    const p = field(parent, slot);
    return p ? bb.__indirect(p) : 0;
  };

  const model = bb.__indirect(0);
  const buffers = tables(model, Slot.Model.buffers).map((buffer) => {
    const p = field(buffer, Slot.Buffer.data);
    if (!p) return null;
    const length = bb.__vector_len(p);
    return length ? bytes.subarray(bb.__vector(p), bb.__vector(p) + length) : null;
  });
  const codes = tables(model, Slot.Model.operatorCodes).map((code) =>
    // Files from before the int32 field existed only have the int8 one.
    Math.max(
      int32(code, Slot.OperatorCode.builtinCode),
      int8(code, Slot.OperatorCode.deprecatedBuiltinCode),
    ),
  );

  const subgraphs = tables(model, Slot.Model.subgraphs).map((sg): TfliteSubgraph => {
    const tensors = tables(sg, Slot.SubGraph.tensors).map((t): TfliteTensor => ({
      name: string(t, Slot.Tensor.name),
      shape: int32s(t, Slot.Tensor.shape),
      type: int8(t, Slot.Tensor.type),
      data: buffers[int32(t, Slot.Tensor.buffer)] ?? null,
    }));
    const operators = tables(sg, Slot.SubGraph.operators).map((op): TfliteOperator => {
      const code = codes[int32(op, Slot.Operator.opcodeIndex)];
      const parsed: TfliteOperator = {
        code,
        inputs: int32s(op, Slot.Operator.inputs),
        outputs: int32s(op, Slot.Operator.outputs),
      };
      const typeField = field(op, Slot.Operator.builtinOptionsType);
      const optionsType = typeField ? bb.readUint8(typeField) : BuiltinOptions.NONE;
      const options = table(op, Slot.Operator.builtinOptions);
      if (code === BuiltinOperator.WHILE && optionsType === BuiltinOptions.WhileOptions) {
        parsed.bodySubgraph = int32(options, Slot.WhileOptions.bodySubgraphIndex);
      }
      return parsed;
    });
    return {
      name: string(sg, Slot.SubGraph.name),
      tensors,
      inputs: int32s(sg, Slot.SubGraph.inputs),
      outputs: int32s(sg, Slot.SubGraph.outputs),
      operators,
    };
  });

  return {
    version: int32(model, Slot.Model.version),
    description: string(model, Slot.Model.description),
    subgraphs,
  };
}

/** A constant float32 tensor's values, copied out of the file. */
function floatData(t: TfliteTensor, what: string): Float32Array {
  if (!t.data) throw new Error(`TFLite reader: ${what} ("${t.name}") is not a constant`);
  if (t.type !== TensorType.FLOAT32) {
    throw new Error(
      `TFLite reader: ${what} ("${t.name}") is not float32 — quantised models are not supported`,
    );
  }
  // slice() gives a fresh, 4-byte-aligned buffer.
  return new Float32Array(t.data.slice().buffer);
}

/** TFLite FULLY_CONNECTED `[outputs, inputs]` weights → Keras `[inputs, outputs]` kernel. */
function kerasKernel(t: TfliteTensor, what: string): Float32Array {
  if (t.shape.length !== 2) throw new Error(`TFLite reader: ${what} is not a matrix`);
  const [outputs, inputs] = t.shape;
  const data = floatData(t, what);
  const out = new Float32Array(data.length);
  for (let i = 0; i < inputs; i++) {
    for (let j = 0; j < outputs; j++) out[i * outputs + j] = data[j * inputs + i];
  }
  return out;
}

/** The op in `sg` that produces each tensor. */
function producers(sg: TfliteSubgraph): Map<number, TfliteOperator> {
  const map = new Map<number, TfliteOperator>();
  for (const op of sg.operators) for (const out of op.outputs) map.set(out, op);
  return map;
}

/** Every LSTM cell (a 4-way SPLIT) reachable from `sg`, in execution order, following WHILE bodies. */
function findCells(
  model: TfliteModel,
  sg: TfliteSubgraph,
): { sg: TfliteSubgraph; split: TfliteOperator }[] {
  return sg.operators.flatMap((op) => {
    if (op.code === BuiltinOperator.WHILE && op.bodySubgraph !== undefined) {
      return findCells(model, model.subgraphs[op.bodySubgraph]);
    }
    if (op.code === BuiltinOperator.SPLIT && op.outputs.length === 4) return [{ sg, split: op }];
    return [];
  });
}

/**
 * Read one LSTM cell's weights by walking back from its gate SPLIT through
 * the ADDs (and the converter's RESHAPE) to the two FULLY_CONNECTED ops. The
 * recurrent projection is the one fed directly by a subgraph input (the h
 * state); biases may sit on a FULLY_CONNECTED or be added as constants.
 */
function readCell(sg: TfliteSubgraph, split: TfliteOperator, layer: number) {
  const producedBy = producers(sg);
  const projections: TfliteOperator[] = [];
  const biases: Float32Array[] = [];
  const what = `LSTM layer ${layer}`;

  const walk = (tensor: number): void => {
    const op = producedBy.get(tensor);
    if (!op) {
      biases.push(floatData(sg.tensors[tensor], `${what} bias`));
    } else if (op.code === BuiltinOperator.ADD) {
      op.inputs.forEach(walk);
    } else if (op.code === BuiltinOperator.RESHAPE) {
      walk(op.inputs[0]);
    } else if (op.code === BuiltinOperator.FULLY_CONNECTED) {
      projections.push(op);
      if (op.inputs[2] >= 0) biases.push(floatData(sg.tensors[op.inputs[2]], `${what} bias`));
    } else {
      throw new Error(`TFLite reader: unexpected op ${op.code} in ${what}`);
    }
  };
  walk(split.inputs[1]); // SPLIT inputs are [axis, value]

  const recurrent = projections.filter((op) => sg.inputs.includes(op.inputs[0]));
  const input = projections.filter((op) => !sg.inputs.includes(op.inputs[0]));
  if (projections.length !== 2 || recurrent.length !== 1 || input.length !== 1) {
    throw new Error(`TFLite reader: could not identify the kernels of ${what}`);
  }
  const kernelTensor = sg.tensors[input[0].inputs[1]];
  const recurrentTensor = sg.tensors[recurrent[0].inputs[1]];
  const gates = kernelTensor.shape[0];
  const bias = new Float32Array(gates);
  for (const b of biases) {
    if (b.length !== gates) throw new Error(`TFLite reader: bad bias length in ${what}`);
    for (let i = 0; i < gates; i++) bias[i] += b[i];
  }
  return {
    inputSize: kernelTensor.shape[1],
    hiddenUnits: gates / 4,
    weights: {
      kernel: kerasKernel(kernelTensor, `${what} kernel`),
      recurrentKernel: kerasKernel(recurrentTensor, `${what} recurrent kernel`),
      bias,
    },
  };
}

/** Read the dense layer that produces `tensor`, looking back through the sigma activation (ELU, ADD). */
function readDense(sg: TfliteSubgraph, tensor: number, what: string) {
  const producedBy = producers(sg);
  let op = producedBy.get(tensor);
  while (op && (op.code === BuiltinOperator.ADD || op.code === BuiltinOperator.ELU)) {
    // The activation's data input is the operand another op computes.
    op = op.inputs.map((t) => producedBy.get(t)).find((p) => p !== undefined);
  }
  if (!op || op.code !== BuiltinOperator.FULLY_CONNECTED || op.inputs[2] < 0) {
    throw new Error(`TFLite reader: could not find the ${what} dense layer`);
  }
  const kernelTensor = sg.tensors[op.inputs[1]];
  const dense: DenseWeights = {
    kernel: kerasKernel(kernelTensor, `${what} kernel`),
    bias: floatData(sg.tensors[op.inputs[2]], `${what} bias`),
  };
  return { outputs: kernelTensor.shape[0], dense };
}

/**
 * Extract the MDRNN weights from an IMPSY `.tflite` file (float32 models
 * only). `writeTflite(readTfliteWeights(bytes))` is a model that computes the
 * same function as `bytes`.
 */
export function readTfliteWeights(bytes: Uint8Array): MDRNNWeights {
  const model = parseTflite(bytes);
  const main = model.subgraphs[0];
  if (!main) throw new Error("TFLite reader: model has no subgraphs");

  const cells = findCells(model, main).map(({ sg, split }, l) => readCell(sg, split, l));
  if (cells.length === 0) throw new Error("TFLite reader: no LSTM layers found");

  const producedBy = producers(main);
  const head = main.outputs
    .map((t) => producedBy.get(t))
    .find((op) => op?.code === BuiltinOperator.CONCATENATION && op.inputs.length === 3);
  if (!head) throw new Error("TFLite reader: no MDN output found");
  const [mus, sigmas, pi] = ["mus", "sigmas", "pi"].map((name, i) =>
    readDense(main, head.inputs[i], `MDN ${name}`),
  );

  const dimension = cells[0].inputSize;
  const weights: MDRNNWeights = {
    config: {
      dimension,
      numLayers: cells.length,
      hiddenUnits: cells[0].hiddenUnits,
      numMixtures: pi.outputs,
    },
    lstm: cells.map((c) => c.weights),
    mdn: { mus: mus.dense, sigmas: sigmas.dense, pi: pi.dense },
  };
  validateWeights(weights);
  return weights;
}
