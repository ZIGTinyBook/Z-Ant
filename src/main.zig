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

pub const std_options = .{
    // Set the log level to info
    .log_level = .debug,
};

const stdout = std.io.getStdOut().writer();
const stderr = std.io.getStdErr().writer();

pub fn main() !void {
    const allocator = @import("pkgAllocator").allocator;

    const args = try std.process.argsWithAllocator(allocator);
    defer args.deinit();

    var target: ?[]const u8 = null;
    var cpu: ?[]const u8 = null;
    var weights_file: ?[]const u8 = null;
    var output_folder: []const u8 = "libpredict.a";

    const executable_name = args.next() orelse unreachable;

    while (args.next()) |arg| {
        // Print help
        if (std.mem.eql(u8, arg, "--help") or std.mem.eql(u8, arg, "-h")) {
            printHelp(executable_name);
            std.process.exit(0);
        }
        // output folder
        else if (std.mem.startsWith(u8, arg, "--output ")) {
            output_folder = arg[9..];
        }
        // compilation target
        else if (std.mem.startsWith(u8, arg, "--target ")) {
            target = arg[9..];
        }
        // compilation CPU
        else if (std.mem.startsWith(u8, arg, "--cpu ")) {
            cpu = arg[6..];
        }
        // Weights file path was not previously specified
        else if (weights_file == null) {
            weights_file = try allocator.dupe(u8, arg);
        }
        // Weights file path was already specified, error
        else {
            try stderr.print("Excess argument: {s}", .{arg});

            printHelp(executable_name);
            std.process.exit(-1);
        }
    }

    if (weights_file) |weights_file_path| {
        var model = model_import_export.importModel(f64, &allocator, weights_file_path) catch {
            try stderr.print("Invalid weights file!", .{});
        };
        defer model.deinit();

        var file = try std.fs.cwd().createFile("/tmp/predict.zig", .{});
        defer file.close();
        const writer = file.writer();

        writer.write(
            \\const tensor = @import("tensor");
            \\const model_import_export = @import("model_import_export");
            \\const build_options = @import("build_options");
            \\
            \\pub const std_options = .{
            \\    // Set the log level to info
            \\    .log_level = .info,
            \\
            \\    // Define logFn to override the std implementation
            \\    .logFn = customLogFn,
            \\};
            \\
            \\var log_function: ?*const fn (string: [*]const u8) callconv(.C) void = null;
            \\var timestamp_function: ?*const fn () callconv(.C) u64 = null;
            \\
            \\pub fn customLogFn(
            \\    comptime level: std.log.Level,
            \\    comptime scope: @Type(.EnumLiteral),
            \\    comptime format: []const u8,
            \\    args: anytype,
            \\) void {
            \\    _ = level;
            \\    _ = scope;
            \\    if (log_function) |unwrapped_log_function| {
            \\        var buf: [256]u8 = [_]u8{0} ** 256;
            \\        unwrapped_log_function((std.fmt.bufPrint(buf[0..250], format, args) catch return).ptr);
            \\    }
            \\}
            \\
            \\export fn setLogFunction(function: *const fn (string: [*]const u8) callconv(.C) void) void {
            \\    log_function = function;
            \\}
            \\
            \\export fn setTimestampFunction(function: *const fn () callconv(.C) u64) void {
            \\    timestamp_function = function;
            \\}
            \\
            \\export fn infer() i16 {
            \\    return internal();
            \\}
            \\
            \\fn getTimestamp() u64 {
            \\    if (timestamp_function) |unwrapped_timestamp_function| {
            \\        return unwrapped_timestamp_function();
            \\    } else {
            \\        return 0;
            \\    }
            \\}
            \\
            \\var buffer: [400000]u8 = undefined;
            \\
            \\fn internal() i16 {
            \\    var fba = std.heap.FixedBufferAllocator.init(&buffer);
            \\    const allocator = fba.allocator();
            \\
            \\    var model = model_import_export.importModelFromPointer(f64, &allocator, @embedFile("model_binary")) catch return -1;
            \\    defer model.deinit();
            \\
            \\    var inputArray: [1][1][28][28]f64 = [1][1][28][28]f64{
            \\        [_]f64{
            \\            [_]f64{
            \\                [_]f64{} ** 28,
            \\            } ** 28,
            \\        },
            \\    };
            \\    var shape: [2]usize = [_]usize{ 2, 3 };
            \\
            \\    var input_tensor = tensor.Tensor(f64).fromArray(&allocator, &inputArray, &shape) catch return -2;
            \\    defer input_tensor.deinit();
            \\
            \\    std.log.info("[{}] starting forward", .{getTimestamp()});
            \\    const result = model.forward(&input_tensor) catch return -3;
            \\    std.log.info("[{}] {any}", .{ getTimestamp(), result.data });
            \\
            \\    return 0;
            \\}
        );
    }
    // No weights file was specified
    else {
        try stderr.print("Missing weights file!", .{});

        printHelp(executable_name);
        std.process.exit(-1);
    }
}

fn printHelp(executable_name: []const u8) !void {
    try stdout.print(
        \\Usage:
        \\    {s} [options] [file]
        \\
        \\Options:
        \\ -h, --help           Print this message
        \\ -Dtarget=<target>    Target option, to be passed verbatim to the Zig compiler
        \\ -Dcpu=<cpu>          Cpu option, to be passed verbatim to the Zig compiler
    , .{executable_name});
}

fn prepareZantProject(path: []const u8, allocator: *std.mem.Allocator) !void {
    const cwd = std.fs.cwd();
    try cwd.makeDir(path);

    const result = try std.process.Child.run(.{ .allocator = allocator, .cwd = path, .argv = .{ "zig", "init" } });
    allocator.free(result.stderr);
    allocator.free(result.stdout);
}
