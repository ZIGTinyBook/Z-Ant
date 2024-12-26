const std = @import("std");
const tensor = @import("tensor");
const layer = @import("layer");
const denselayer = @import("denselayer").DenseLayer;
const convlayer = @import("convLayer").ConvolutionalLayer;
const flattenlayer = @import("flattenLayer").FlattenLayer;
const activationlayer = @import("activationlayer").ActivationLayer;
const Model = @import("model").Model;
const loader = @import("dataloader");
const ActivationType = @import("activation_function").ActivationType;
const LossType = @import("loss").LossType;
const Trainer = @import("trainer");
const InfoAllocator = @import("info_allocator");
const model_import_export = @import("model_import_export");
const build_options = @import("build_options");

pub const std_options = .{
    // Set the log level to info
    .log_level = .info,

    // Define logFn to override the std implementation
    .logFn = myLogFn,
};

pub fn myLogFn(
    comptime level: std.log.Level,
    comptime scope: @Type(.EnumLiteral),
    comptime format: []const u8,
    args: anytype,
) void {
    _ = level;
    _ = scope;
    _ = format;
    _ = args;
}

export fn infer() i16 {
    return internal();
}

fn internal() i16 {
    var buffer: [200000]u8 = undefined;
    var fba = std.heap.FixedBufferAllocator.init(&buffer);
    const allocator = fba.allocator();

    var model = model_import_export.importModelFromPointer(f64, &allocator, @embedFile("model_binary")) catch return -1;

    defer model.deinit();

    var inputArray: [2][3]f64 = [_][3]f64{
        [_]f64{ 1.0, 2.0, 3.0 },
        [_]f64{ 4.0, 5.0, 6.0 },
    };
    var shape: [2]usize = [_]usize{ 2, 3 };

    var input_tensor = tensor.Tensor(f64).fromArray(&allocator, &inputArray, &shape) catch return -2;
    defer input_tensor.deinit();

    _ = model.forward(&input_tensor) catch return -3;

    return 0;
}
