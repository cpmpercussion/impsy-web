// The slice of TFLite's FlatBuffer schema (tensorflow/compiler/mlir/lite/schema/
// schema.fbs, file identifier "TFL3") that IMPSY MDRNN models use. Shared by
// tfliteWriter.ts and tfliteReader.ts.
//
// `Slot` values are each table's field indices in schema order; they are the
// file format, so they never change. Numbers were taken from TensorFlow 2.21's
// generated schema bindings.

export const TFLITE_IDENTIFIER = "TFL3";
export const TFLITE_SCHEMA_VERSION = 3;

export const BuiltinOperator = {
  ADD: 0,
  CONCATENATION: 2,
  FULLY_CONNECTED: 9,
  LOGISTIC: 14,
  MUL: 18,
  RESHAPE: 22,
  TANH: 28,
  SPLIT: 49,
  ELU: 111,
  WHILE: 119,
} as const;

/** Tags of the `BuiltinOptions` union. */
export const BuiltinOptions = {
  NONE: 0,
  FullyConnectedOptions: 8,
  ConcatenationOptions: 10,
  AddOptions: 11,
  ReshapeOptions: 17,
  MulOptions: 21,
  SplitOptions: 35,
  WhileOptions: 93,
} as const;

export const TensorType = { FLOAT32: 0, INT32: 2 } as const;

export const Slot = {
  Model: { version: 0, operatorCodes: 1, subgraphs: 2, description: 3, buffers: 4 },
  OperatorCode: { deprecatedBuiltinCode: 0, version: 2, builtinCode: 3 },
  SubGraph: { tensors: 0, inputs: 1, outputs: 2, operators: 3, name: 4 },
  Tensor: { shape: 0, type: 1, buffer: 2, name: 3 },
  Operator: { opcodeIndex: 0, inputs: 1, outputs: 2, builtinOptionsType: 3, builtinOptions: 4 },
  Buffer: { data: 0 },
  FullyConnectedOptions: { fusedActivationFunction: 0, weightsFormat: 1, keepNumDims: 2 },
  ConcatenationOptions: { axis: 0, fusedActivationFunction: 1 },
  AddOptions: { fusedActivationFunction: 0, potScaleInt16: 1 },
  MulOptions: { fusedActivationFunction: 0 },
  ReshapeOptions: { newShape: 0 },
  SplitOptions: { numSplits: 0 },
  WhileOptions: { condSubgraphIndex: 0, bodySubgraphIndex: 1 },
} as const;
