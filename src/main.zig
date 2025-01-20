const std = @import("std");
const tensor = @import("tensor");
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
const Codegen = @import("codegen");

pub const std_options = .{
    // Set the log level to info
    .log_level = .debug,
};

const stdout = std.io.getStdOut().writer();
const stderr = std.io.getStdErr().writer();

pub fn main() !void {
    var allocator = @import("pkgAllocator").allocator;

    var args = try std.process.argsWithAllocator(allocator);
    defer args.deinit();

    var target: ?[]const u8 = null;
    var cpu: ?[]const u8 = null;
    var weights_file: ?[]const u8 = null;
    var output_folder: []const u8 = "zant-predict";

    const executable_name = args.next() orelse unreachable;

    while (args.next()) |arg| {
        // Print help
        if (std.mem.eql(u8, arg, "--help") or std.mem.eql(u8, arg, "-h")) {
            try printHelp(executable_name);
            std.process.exit(0);
        }
        // output folder
        else if (std.mem.startsWith(u8, arg, "--output")) {
            output_folder = args.next() orelse {
                try stderr.print("Missing output\n", .{});
                try printHelp(executable_name);
                std.process.exit(1);
            };
        }
        // compilation target
        else if (std.mem.startsWith(u8, arg, "--target")) {
            target =
                args.next() orelse {
                try stderr.print("Missing target\n", .{});
                try printHelp(executable_name);
                std.process.exit(1);
            };
        }
        // compilation CPU
        else if (std.mem.startsWith(u8, arg, "--cpu")) {
            cpu =
                args.next() orelse {
                try stderr.print("Missing cpu\n", .{});
                try printHelp(executable_name);
                std.process.exit(1);
            };
        }
        // Weights file path was not previously specified
        else if (weights_file == null) {
            weights_file = try allocator.dupe(u8, arg);
        }
        // Weights file path was already specified, error
        else {
            try stderr.print("Excess argument: {s}\n", .{arg});

            try printHelp(executable_name);
            std.process.exit(1);
        }
    }

    if (weights_file) |weights_file_path| {
        // TODO: the base unit should be dynamically parametrizable here
        var model = model_import_export.importModel(f64, &allocator, weights_file_path) catch {
            try stderr.print("Invalid weights file!\n", .{});
            std.process.exit(1);
        };
        defer model.deinit();

        const nn_zig = try prepareZantProject(output_folder, allocator);
        try writeZigFile(nn_zig.file, model);

        const result = try std.process.Child.run(.{ .allocator = allocator, .argv = &.{ "zig", "fmt", nn_zig.path } });
        allocator.free(result.stderr);
        allocator.free(result.stdout);
        allocator.free(nn_zig.path);
    }
    // No weights file was specified
    else {
        try stderr.print("Missing weights file!\n", .{});

        try printHelp(executable_name);
        std.process.exit(1);
    }
}

fn printHelp(executable_name: []const u8) !void {
    try stdout.print(
        \\
        \\Usage:
        \\    {s} [options] [file]
        \\
        \\Options:
        \\ -h, --help           Print this message
        \\ -Dtarget=<target>    Target option, to be passed verbatim to the Zig compiler
        \\ -Dcpu=<cpu>          Cpu option, to be passed verbatim to the Zig compiler
        \\ --output <path>      Output folder
        \\
        \\
    , .{executable_name});
}

fn prepareZantProject(path: []const u8, allocator: std.mem.Allocator) !struct { file: std.fs.File, path: []u8 } {
    {
        const result = try std.process.Child.run(.{ .allocator = allocator, .argv = &.{ "rm", "-rf", path } });
        allocator.free(result.stderr);
        allocator.free(result.stdout);
    }

    {
        const result = try std.process.Child.run(.{ .allocator = allocator, .argv = &.{ "cp", "-r", "/home/maldus/Projects/Maldus512/zant-template/", path } });
        allocator.free(result.stderr);
        allocator.free(result.stdout);
    }

    const nn_zig = try std.fs.path.join(allocator, &.{ path, "src/nn.zig" });
    errdefer allocator.free(nn_zig);

    return .{ .file = try std.fs.cwd().openFile(nn_zig, .{ .mode = .write_only }), .path = nn_zig };
}

fn writeZigFile(file: std.fs.File, model: Model(f64)) !void {
    defer file.close();
    const writer = file.writer();

    if (model.layers.items.len == 0) {
        return;
    }

    _ = try writer.write(
        \\const std = @import("std");
        \\const tensor = @import("tensor");
        \\const model_import_export = @import("model_import_export");
        \\
        \\pub fn predict(input: *anyopaque, output: *anyopaque) !i16 {
        \\
    );

    {
        const first_layer = model.layers.items[0];
        const n_inputs = first_layer.get_n_inputs();
        std.log.info("input tensor size {}", .{n_inputs});

        _ = try writer.write("const input_tensor = blk: {\n");
        try Codegen.writeAllocator(writer, "allocator", n_inputs);
        try writer.print(
            \\const input_ptr: [*]const f64 = @ptrCast(@alignCast(input));
            \\var shape: [2]usize = [_]usize{{1, {}}};
            \\break :blk try tensor.Tensor(f64).fromArray(&allocator, input_ptr, &shape);
        , .{n_inputs});
        _ = try writer.write("};\n");

        _ = try writer.write(
            \\input_tensor.print();
            \\
            \\_ = output;
            \\
        );
    }

    for (model.layers.items) |layer| {
        //try layer.codegen(writer);
        _ = layer;
    }

    _ = try writer.write(
        \\   return 0;
        \\}
    );
}
