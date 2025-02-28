const std = @import("std");
const zant = @import("zant");
const onnx = zant.onnx;
const Tensor = zant.core.tensor.Tensor;
const tensorMath = zant.core.tensor.math_standard;
const allocator = zant.utils.allocator.allocator;
const codeGen = @import("codeGen_skeleton.zig");
const codeGen_utils = @import("codeGen_utils.zig");
const codeGen_init = @import("codeGen_initializers.zig");
const codeGen_mathHandl = @import("codeGen_math_handler.zig");
const codeGen_predict = @import("codeGen_predict.zig");

const codegen_options = @import("codegen_options");
const globals = @import("globals.zig");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const gpa_allocator = gpa.allocator();

    var model = try onnx.parseFromFile(gpa_allocator, "datasets/models/mnist-8/mnist-8.onnx");
    defer model.deinit(gpa_allocator);

    //onnx.printStructure(&model);

    const file_path = "src/codeGen/static_lib.zig";
    var file = try std.fs.cwd().createFile(file_path, .{});
    std.debug.print("\n .......... file created, path:{s}", .{file_path});
    defer file.close();

    //create the hashMap
    try globals.populateReadyTensorHashMap(model);

    //DEBUG
    //utils.printTensorHashMap(tensorHashMap);

    //DEBUG
    //try utils.printOperations(model.graph.?);

    //create the ReadyGraph
    try globals.populateReadyGraph(model);

    //DEBUG
    //try utils.printNodeList(readyGraph);

    try codeGen.writeZigFile(file, model);
}
