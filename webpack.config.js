const path = require("path");
const CopyPlugin = require("copy-webpack-plugin");

module.exports = () => {
  return {
    target: ["web"],
    entry: path.resolve(__dirname, "index.js"),
    output: {
      path: path.resolve(__dirname, "wwwroot", "js", "ort"),
      filename: "bundle.min.js",
      library: {
        type: "umd",
      },
    },
    plugins: [
      new CopyPlugin({
        patterns: [
          {
            from: "node_modules/onnxruntime-web/dist/*.wasm",
            to: "[name][ext]",
          },
        ],
      }),
    ],
    mode: "production",
  };
};
