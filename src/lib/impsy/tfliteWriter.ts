// TFLite writer — serialises MDRNN weights to an IMPSY-contract `.tflite` file
// without TensorFlow. No counterpart in AUv3 or Python, where
// `tf.lite.TFLiteConverter` does this (../impsy/impsy/tflite_converter.py).
//
// This is not a converter: it emits the one graph IMPSY uses, a single-step
// stacked LSTM with an MDN head, sized by `ModelConfig`. The converter wraps
// each LSTM layer in a WHILE loop that runs once; here the cell is written
// inline, so the file is one subgraph of plain builtin ops (no Flex ops).
//
// The interface matches converter-made models exactly (see CLAUDE.md, "Model
// interface contract"): inputs `inputs (1,1,D)`, `state_h_N`/`state_c_N
// (1,H)` in that order; outputs `Identity (1, M·(2D+1))` then `Identity_1…`,
// the updated states as `[h0, c0, h1, c1, …]` in ascending tensor index.

import { Builder } from "flatbuffers";
import { SIGMA_OFFSET, validateWeights, type MDRNNWeights } from "./mdrnnWeights";
import {
  BuiltinOperator,
  BuiltinOptions,
  Slot,
  TFLITE_IDENTIFIER,
  TFLITE_SCHEMA_VERSION,
  TensorType,
} from "./tfliteSchema";

type Offset = number;

interface TensorDef {
  name: string;
  shape: number[];
  type: number;
  /** Constant data; absent for inputs and computed tensors. */
  data?: Uint8Array;
}

interface OpDef {
  code: number;
  inputs: number[];
  outputs: number[];
  optionsType: number;
  /** Builds the op's options table, when it has one. */
  options?: (b: Builder) => Offset;
}

/** Constant weight data is aligned to this many bytes, as the schema asks (`force_align: 16`). */
const DATA_ALIGNMENT = 16;

function bytesOf(a: Float32Array | Int32Array): Uint8Array {
  return new Uint8Array(a.buffer, a.byteOffset, a.byteLength);
}

/** Keras `[inputs, outputs]` kernel → TFLite FULLY_CONNECTED `[outputs, inputs]` weights. */
function transpose(kernel: Float32Array, inputs: number, outputs: number): Float32Array {
  const out = new Float32Array(kernel.length);
  for (let i = 0; i < inputs; i++) {
    for (let j = 0; j < outputs; j++) out[j * inputs + i] = kernel[i * outputs + j];
  }
  return out;
}

/** Accumulates the tensors and operators of the model's single subgraph. */
class Graph {
  readonly tensors: TensorDef[] = [];
  readonly ops: OpDef[] = [];

  tensor(name: string, shape: number[], data?: Float32Array | Int32Array): number {
    const type = data instanceof Int32Array ? TensorType.INT32 : TensorType.FLOAT32;
    this.tensors.push({ name, shape, type, data: data && bytesOf(data) });
    return this.tensors.length - 1;
  }

  private op(
    code: number,
    inputs: number[],
    outputs: number[],
    optionsType: number = BuiltinOptions.NONE,
    options?: (b: Builder) => Offset,
  ): void {
    this.ops.push({ code, inputs, outputs, optionsType, options });
  }

  /** `out = input · weightsᵀ (+ bias)`; `bias` -1 means none. */
  fullyConnected(input: number, weights: number, bias: number, out: number): void {
    this.op(
      BuiltinOperator.FULLY_CONNECTED,
      [input, weights, bias],
      [out],
      BuiltinOptions.FullyConnectedOptions,
      (b) => {
        b.startObject(3); // all defaults: no fused activation, default weights format
        return b.endObject();
      },
    );
  }

  add(a: number, b: number, out: number): void {
    this.op(BuiltinOperator.ADD, [a, b], [out], BuiltinOptions.AddOptions, (bld) => {
      bld.startObject(2);
      return bld.endObject();
    });
  }

  mul(a: number, b: number, out: number): void {
    this.op(BuiltinOperator.MUL, [a, b], [out], BuiltinOptions.MulOptions, (bld) => {
      bld.startObject(1);
      return bld.endObject();
    });
  }

  unary(code: number, input: number, out: number): void {
    this.op(code, [input], [out]);
  }

  reshape(input: number, shapeTensor: number, shape: number[], out: number): void {
    this.op(
      BuiltinOperator.RESHAPE,
      [input, shapeTensor],
      [out],
      BuiltinOptions.ReshapeOptions,
      (b) => {
        const newShape = int32Vector(b, shape);
        b.startObject(1);
        b.addFieldOffset(Slot.ReshapeOptions.newShape, newShape, 0);
        return b.endObject();
      },
    );
  }

  split(axisTensor: number, input: number, outs: number[]): void {
    this.op(BuiltinOperator.SPLIT, [axisTensor, input], outs, BuiltinOptions.SplitOptions, (b) => {
      b.startObject(1);
      b.addFieldInt32(Slot.SplitOptions.numSplits, outs.length, 0);
      return b.endObject();
    });
  }

  concatenation(inputs: number[], axis: number, out: number): void {
    this.op(
      BuiltinOperator.CONCATENATION,
      inputs,
      [out],
      BuiltinOptions.ConcatenationOptions,
      (b) => {
        b.startObject(2);
        b.addFieldInt32(Slot.ConcatenationOptions.axis, axis, 0);
        return b.endObject();
      },
    );
  }
}

function int32Vector(b: Builder, values: number[]): Offset {
  b.startVector(4, values.length, 4);
  for (let i = values.length - 1; i >= 0; i--) b.addInt32(values[i]);
  return b.endVector();
}

function offsetVector(b: Builder, offsets: Offset[]): Offset {
  b.startVector(4, offsets.length, 4);
  for (let i = offsets.length - 1; i >= 0; i--) b.addOffset(offsets[i]);
  return b.endVector();
}

function alignedByteVector(b: Builder, bytes: Uint8Array): Offset {
  b.startVector(1, bytes.length, DATA_ALIGNMENT);
  for (let i = bytes.length - 1; i >= 0; i--) b.addInt8(bytes[i]);
  return b.endVector();
}

/** Lay out the MDRNN graph for `weights`; returns it with its input and output tensor indices. */
function buildGraph(weights: MDRNNWeights): { graph: Graph; inputs: number[]; outputs: number[] } {
  const { dimension: D, numLayers, hiddenUnits: H, numMixtures: M } = weights.config;
  const g = new Graph();

  // Graph inputs first, so their tensor indices are in feed order.
  const inputs = [g.tensor("inputs", [1, 1, D])];
  const stateIn: { h: number; c: number }[] = [];
  for (let l = 0; l < numLayers; l++) {
    stateIn.push({
      h: g.tensor(`state_h_${l}`, [1, H]),
      c: g.tensor(`state_c_${l}`, [1, H]),
    });
    inputs.push(stateIn[l].h, stateIn[l].c);
  }

  // (1,1,D) → (1,D): the single timestep.
  const stepShape = g.tensor("step_shape", [2], Int32Array.of(1, D));
  let x = g.tensor("step", [1, D]);
  g.reshape(inputs[0], stepShape, [1, D], x);

  const gateAxis = g.tensor("gate_axis", [], Int32Array.of(1));
  const stateOutputs: number[] = [];
  weights.lstm.forEach((layer, l) => {
    const p = `lstm_${l}`;
    const inSize = l === 0 ? D : H;
    const kernel = g.tensor(`${p}/kernel`, [4 * H, inSize], transpose(layer.kernel, inSize, 4 * H));
    const recurrent = g.tensor(
      `${p}/recurrent_kernel`,
      [4 * H, H],
      transpose(layer.recurrentKernel, H, 4 * H),
    );
    const bias = g.tensor(`${p}/bias`, [4 * H], layer.bias);
    // Allocate the state outputs h then c, so ascending tensor index is
    // [h0, c0, h1, c1, …] — readers pair state outputs with inputs by index.
    const hOut = g.tensor(`Identity_${stateOutputs.length + 1}`, [1, H]);
    const cOut = g.tensor(`Identity_${stateOutputs.length + 2}`, [1, H]);
    stateOutputs.push(hOut, cOut);

    // z = x·W + b + h·U, split into the gates i, f, g (cell candidate), o.
    const zx = g.tensor(`${p}/input_projection`, [1, 4 * H]);
    const zh = g.tensor(`${p}/recurrent_projection`, [1, 4 * H]);
    const z = g.tensor(`${p}/gates`, [1, 4 * H]);
    g.fullyConnected(x, kernel, bias, zx);
    g.fullyConnected(stateIn[l].h, recurrent, -1, zh);
    g.add(zx, zh, z);
    const [zi, zf, zg, zo] = ["i", "f", "g", "o"].map((n) => g.tensor(`${p}/split_${n}`, [1, H]));
    g.split(gateAxis, z, [zi, zf, zg, zo]);

    const [i, f, cand, o] = ["i", "f", "g", "o"].map((n) => g.tensor(`${p}/gate_${n}`, [1, H]));
    g.unary(BuiltinOperator.LOGISTIC, zi, i);
    g.unary(BuiltinOperator.LOGISTIC, zf, f);
    g.unary(BuiltinOperator.TANH, zg, cand);
    g.unary(BuiltinOperator.LOGISTIC, zo, o);

    // c' = f·c + i·g ;  h' = o·tanh(c')
    const keep = g.tensor(`${p}/forget_cell`, [1, H]);
    const write = g.tensor(`${p}/input_candidate`, [1, H]);
    const cTanh = g.tensor(`${p}/cell_tanh`, [1, H]);
    g.mul(f, stateIn[l].c, keep);
    g.mul(i, cand, write);
    g.add(keep, write, cOut);
    g.unary(BuiltinOperator.TANH, cOut, cTanh);
    g.mul(o, cTanh, hOut);
    x = hOut;
  });

  // MDN head: [mus | elu(sigmas) + 1 + eps | pi logits].
  const dense = (name: string, outputs: number) => {
    const w = weights.mdn[name as keyof MDRNNWeights["mdn"]];
    const kernel = g.tensor(`mdn_${name}/kernel`, [outputs, H], transpose(w.kernel, H, outputs));
    const bias = g.tensor(`mdn_${name}/bias`, [outputs], w.bias);
    const out = g.tensor(`mdn_${name}`, [1, outputs]);
    g.fullyConnected(x, kernel, bias, out);
    return out;
  };
  const mus = dense("mus", M * D);
  const sigmaLinear = dense("sigmas", M * D);
  const pi = dense("pi", M);
  const sigmaElu = g.tensor("mdn_sigmas/elu", [1, M * D]);
  const sigmaOffset = g.tensor("mdn_sigmas/offset", [], Float32Array.of(SIGMA_OFFSET));
  const sigmas = g.tensor("mdn_sigmas/positive", [1, M * D]);
  g.unary(BuiltinOperator.ELU, sigmaLinear, sigmaElu);
  g.add(sigmaElu, sigmaOffset, sigmas);
  const mdn = g.tensor("Identity", [1, M * (2 * D + 1)]);
  g.concatenation([mus, sigmas, pi], -1, mdn);

  return { graph: g, inputs, outputs: [mdn, ...stateOutputs] };
}

/**
 * Serialise `weights` as a `.tflite` model following the IMPSY model interface
 * contract, loadable by IMPSY's Python, AUv3 and web front ends. Float32 only
 * (no quantisation).
 */
export function writeTflite(weights: MDRNNWeights): Uint8Array {
  validateWeights(weights);
  const { dimension: D, numLayers, hiddenUnits: H, numMixtures: M } = weights.config;
  const { graph, inputs, outputs } = buildGraph(weights);
  const b = new Builder(1024);

  // Buffers: index 0 is the conventional empty buffer every non-constant tensor points at.
  const bufferOffsets: Offset[] = [];
  const addBuffer = (data?: Uint8Array): number => {
    const vector = data ? alignedByteVector(b, data) : 0;
    b.startObject(1);
    if (vector) b.addFieldOffset(Slot.Buffer.data, vector, 0);
    bufferOffsets.push(b.endObject());
    return bufferOffsets.length - 1;
  };
  addBuffer();

  const tensorOffsets = graph.tensors.map((t) => {
    const buffer = t.data ? addBuffer(t.data) : 0;
    const shape = int32Vector(b, t.shape);
    const name = b.createString(t.name);
    b.startObject(4);
    b.addFieldOffset(Slot.Tensor.shape, shape, 0);
    b.addFieldInt8(Slot.Tensor.type, t.type, 0);
    b.addFieldInt32(Slot.Tensor.buffer, buffer, 0);
    b.addFieldOffset(Slot.Tensor.name, name, 0);
    return b.endObject();
  });

  // One OperatorCode per distinct builtin op, in order of first use.
  const codes = [...new Set(graph.ops.map((op) => op.code))];
  const operatorOffsets = graph.ops.map((op) => {
    const opInputs = int32Vector(b, op.inputs);
    const opOutputs = int32Vector(b, op.outputs);
    const options = op.options ? op.options(b) : 0;
    b.startObject(5);
    b.addFieldInt32(Slot.Operator.opcodeIndex, codes.indexOf(op.code), 0);
    b.addFieldOffset(Slot.Operator.inputs, opInputs, 0);
    b.addFieldOffset(Slot.Operator.outputs, opOutputs, 0);
    b.addFieldInt8(Slot.Operator.builtinOptionsType, op.optionsType, 0);
    if (options) b.addFieldOffset(Slot.Operator.builtinOptions, options, 0);
    return b.endObject();
  });

  const tensorsVector = offsetVector(b, tensorOffsets);
  const inputsVector = int32Vector(b, inputs);
  const outputsVector = int32Vector(b, outputs);
  const operatorsVector = offsetVector(b, operatorOffsets);
  const subgraphName = b.createString("main");
  b.startObject(5);
  b.addFieldOffset(Slot.SubGraph.tensors, tensorsVector, 0);
  b.addFieldOffset(Slot.SubGraph.inputs, inputsVector, 0);
  b.addFieldOffset(Slot.SubGraph.outputs, outputsVector, 0);
  b.addFieldOffset(Slot.SubGraph.operators, operatorsVector, 0);
  b.addFieldOffset(Slot.SubGraph.name, subgraphName, 0);
  const subgraph = b.endObject();

  const codeOffsets = codes.map((code) => {
    b.startObject(4);
    // Codes below 127 are also written to the pre-2.4 int8 field, as the converter does.
    b.addFieldInt8(Slot.OperatorCode.deprecatedBuiltinCode, code, 0);
    b.addFieldInt32(Slot.OperatorCode.version, 1, 1);
    b.addFieldInt32(Slot.OperatorCode.builtinCode, code, 0);
    return b.endObject();
  });

  const codesVector = offsetVector(b, codeOffsets);
  const subgraphsVector = offsetVector(b, [subgraph]);
  const description = b.createString(
    `IMPSY MDRNN dim${D} layers${numLayers} units${H} mixtures${M} (impsy-web tfliteWriter)`,
  );
  const buffersVector = offsetVector(b, bufferOffsets);
  b.startObject(5);
  b.addFieldInt32(Slot.Model.version, TFLITE_SCHEMA_VERSION, 0);
  b.addFieldOffset(Slot.Model.operatorCodes, codesVector, 0);
  b.addFieldOffset(Slot.Model.subgraphs, subgraphsVector, 0);
  b.addFieldOffset(Slot.Model.description, description, 0);
  b.addFieldOffset(Slot.Model.buffers, buffersVector, 0);
  b.finish(b.endObject(), TFLITE_IDENTIFIER);

  // Copy out of the builder's (larger) backing buffer.
  return b.asUint8Array().slice();
}
