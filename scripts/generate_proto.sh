#!/usr/bin/env bash
# Generate Python gRPC code from proto/fl.proto for the coordinator and the CLI.
# (Node.js loads the .proto at runtime, so it needs no generated code.)
set -e
cd "$(dirname "$0")/.."
for dir in coordinator client; do
  mkdir -p "$dir"
  python3 -m grpc_tools.protoc -I proto --python_out="$dir" --grpc_python_out="$dir" proto/fl.proto
done
echo "Generated fl_pb2.py and fl_pb2_grpc.py in coordinator/ and client/"
