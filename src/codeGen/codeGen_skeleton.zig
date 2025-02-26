const std = @import("std");
const zant = @import("zant");

const Tensor = zant.core.tensor.Tensor;
const tensorMath = zant.core.tensor.math_standard;
const onnx = zant.onnx;
const ModelOnnx = onnx.ModelProto;
const DataType = onnx.DataType;
const TensorProto = onnx.TensorProto;
const allocator = zant.utils.allocator.allocator;
const codeGenInitializers = @import("codeGen_initializers.zig");
const coddeGenPredict = @import("codeGen_predict.zig");
const codegen_options = @import("codegen_options");

/// Writes a Zig source file containing the generated code for an ONNX model.
///
/// This function generates the necessary Zig code to initialize tensors and
/// define the prediction logic based on the given ONNX model.
///
/// # Parameters
/// - `file`: The file where the generated Zig code will be written.
/// - `model`: The ONNX model from which to generate the Zig code.
///
/// # Errors
/// This function may return an error if writing to the file fails.
pub fn writeZigFile(file: std.fs.File, model: ModelOnnx) !void {
    const writer = file.writer();

    // Write the necessary library imports to the generated Zig file
    try write_libraries(writer);

    if (codegen_options.log) {
        //log function setting
        try write_logFunction(writer);
    }

    //Fixed Buffer Allocator
    try write_FBA(writer);

    try write_type_T(writer);

    // Generate tensor initialization code
    try codeGenInitializers.writeTensorsInit(writer, model);

    //try write_debug(writer);

    // Generate prediction function code
    try coddeGenPredict.writePredict(writer);
}

/// Writes the required library imports to the generated Zig file.
///
/// This function ensures that the necessary standard and package libraries are
/// imported into the generated Zig source file.
///
/// # Parameters
/// - `writer`: A file writer used to write the import statements.
///
/// # Errors
/// This function may return an error if writing to the file fails.
fn write_libraries(writer: std.fs.File.Writer) !void {
    _ = try writer.print(
        \\
        \\ const std = @import("std");
        \\ const zant = @import("zant");
        \\ const Tensor = zant.core.tensor.Tensor;
        \\ const tensMath = zant.core.tensor.math_standard;
        \\ const pkgAllocator = zant.utils.allocator;
        \\ const allocator = pkgAllocator.allocator;
        \\ const utils = @import("codeGen_utils.zig");
        \\
    , .{});
}

fn write_logFunction(writer: std.fs.File.Writer) !void {
    _ = try writer.print(
        \\
        \\var log_function: ?*const fn ([*c]u8) callconv(.C) void = null;
        \\
        \\pub export fn setLogFunction(func: ?*const fn ([*c]u8) callconv(.C) void) void {{
        \\    log_function = func;
        \\}}
        \\
    , .{});
}
fn write_FBA(writer: std.fs.File.Writer) !void {
    _ = try writer.print(
        \\
        \\
        \\ var buf: [4096 * 10]u8 = undefined;
        \\ var fba_state = @import("std").heap.FixedBufferAllocator.init(&buf);
        \\ const fba = fba_state.allocator();
    , .{});
}

fn write_type_T(writer: std.fs.File.Writer) !void {
    _ = try writer.print( //TODO: get the type form the onnx model
        \\
        \\ const T = f32;
    , .{});
}

fn write_debug(writer: std.fs.File.Writer) !void {
    _ = try writer.print(
        \\
        \\
        \\export fn debug() void {{
        \\      std.debug.print("\n#############################################################", .{{}});
        \\      std.debug.print("\n+                      DEBUG                     +", .{{}});
        \\      std.debug.print("\n#############################################################", .{{}});
        \\}}
    , .{});
}
