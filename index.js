export const ort = require("onnxruntime-web");

let session;
window.runOnnxRuntime = async (methodParameter) => {
  if (session == null) {
    session = await ort.InferenceSession.create("/onnx/mnist.onnx");
  }
  const dataIn = Float32Array.from(methodParameter);
  const tensorIn = new ort.Tensor("float32", dataIn, [1, 28, 28, 1]);

  const results = await session.run({ conv2d_2_input: tensorIn });
  const resultData = Array.from(results["dense_3"].data);
  var currentMax = resultData[0];
  var resultValue = 0;
  resultData.forEach((v, i) => {
    if (v > currentMax) {
      currentMax = v;
      resultValue = i;
    }
  });
  return resultValue;
};
