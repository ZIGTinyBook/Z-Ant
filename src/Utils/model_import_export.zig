const std = @import("std");
const cwd = std.fs.cwd();
const Model = @import("model").Model;
const Layer = @import("layer");
const DenseLayer = @import("denselayer").DenseLayer;
const ActivationLayer = @import("activationlayer").ActivationLayer;

const LayerType = @import("layer").LayerType;
const Tensor = @import("tensor").Tensor;
const ActivationType = @import("activation_function").ActivationType;

// ----------------- Export -----------------
pub fn exportModel(
    comptime T: type,
    model: Model(T),
    file_path: []const u8,
) !void {
    std.log.debug("\n ..... EXPORTING THE MODEL ......", .{});
    var file = try std.fs.cwd().createFile(file_path, .{});
    std.log.debug("\n ..... file created ......", .{});
    defer file.close();

    const writer = file.writer();
    std.log.debug("\n ..... writer created ......", .{});

    try writer.writeInt(usize, model.layers.items.len, std.builtin.Endian.big);
    for (model.layers.items) |*l| {
        try exportLayer(T, l.*, writer);
    }
    return;
}

pub fn exportLayer(
    comptime T: type,
    layer: Layer.Layer(T),
    writer: std.fs.File.Writer,
) !void {
    std.log.debug("\n ... export layer... ", .{});

    //TODO: handle Default layer and null layer
    if (layer.layer_type == LayerType.DenseLayer) {
        _ = try writer.write("Dense.....");
        const denseLayer: *DenseLayer(T) = @alignCast(@ptrCast(layer.layer_ptr));
        try exportLayerDense(T, denseLayer.*, writer);
    } else if (layer.layer_type == LayerType.ActivationLayer) {
        _ = try writer.write("Activation");
        const activationLayer: *ActivationLayer(T) = @alignCast(@ptrCast(layer.layer_ptr));
        try exportLayerActivation(T, activationLayer.*, writer);
    }
}

pub fn exportLayerDense(
    comptime T: type,
    layer: DenseLayer(T),
    writer: std.fs.File.Writer,
) !void {
    std.log.debug(" dense ", .{});

    try exportTensor(T, layer.weights, writer);
    try exportTensor(T, layer.bias, writer);
    try writer.writeInt(usize, layer.n_inputs, std.builtin.Endian.big);
    try writer.writeInt(usize, layer.n_neurons, std.builtin.Endian.big);
    try exportTensor(T, layer.w_gradients, writer);
    try exportTensor(T, layer.b_gradients, writer);
}

pub fn exportLayerActivation(
    comptime T: type,
    layer: ActivationLayer(T),
    writer: std.fs.File.Writer,
) !void {
    std.log.debug(" activation ", .{});

    try writer.writeInt(usize, layer.n_inputs, std.builtin.Endian.big);
    try writer.writeInt(usize, layer.n_neurons, std.builtin.Endian.big);

    if (layer.activationFunction == ActivationType.ReLU) {
        _ = try writer.write("ReLU......");
    } else if (layer.activationFunction == ActivationType.Sigmoid) {
        _ = try writer.write("Sigmoid...");
    } else if (layer.activationFunction == ActivationType.Softmax) {
        _ = try writer.write("Softmax...");
    } else if (layer.activationFunction == ActivationType.None) {
        _ = try writer.write("None......");
    }
}

pub fn exportTensor(
    comptime T: type,
    tensor: Tensor(T),
    writer: std.fs.File.Writer,
) !void {
    std.log.debug("\n ... export tensor... ", .{});

    // Write tensor size and shape
    try writer.writeInt(usize, tensor.size, std.builtin.Endian.big);
    try writer.writeInt(usize, tensor.shape.len, std.builtin.Endian.big);
    for (tensor.shape) |dim| {
        try writer.writeInt(usize, dim, std.builtin.Endian.big);
    }

    // Write tensor data
    for (tensor.data) |value| {
        try writeNumber(T, value, writer);
    }
}

pub fn writeNumber(
    comptime T: type,
    number: T,
    writer: std.fs.File.Writer,
) !void {
    const size = @sizeOf(T);
    var buffer: [size]u8 = @bitCast(number);
    try writer.writeAll(&buffer);
}

// ----------------- Import -----------------

pub fn importModelFromPointer(
    comptime T: type,
    allocator: *const std.mem.Allocator,
    model_buffer: []const u8,
) !Model(T) {
    const MemoryReader = struct {
        const Self = @This();

        index: usize,
        contents: []const u8,

        fn readFn(context: *Self, buffer: []u8) error{}!usize {
            const available = context.contents.len - context.index;
            const len = if (buffer.len > available) available else buffer.len;
            @memcpy(buffer, context.contents[context.index .. context.index + len]);
            context.index += len;
            return len;
        }
    };

    var memory_reader = MemoryReader{
        .index = 0,
        .contents = model_buffer,
    };
    const reader = std.io.GenericReader(*MemoryReader, error{}, MemoryReader.readFn){ .context = &memory_reader };

    var model: Model(T) = Model(T){
        .layers = undefined,
        .allocator = allocator,
        .input_tensor = undefined,
    };
    try model.init();

    const n_layers = try reader.readInt(usize, std.builtin.Endian.big);
    for (0..n_layers) |_| {
        const newLayer: Layer.Layer(T) = try importLayer(T, allocator, reader);

        try model.addLayer(newLayer);
    }
    return model;
}

pub fn importModel(
    comptime T: type,
    allocator: *const std.mem.Allocator,
    file_path: []const u8,
) !Model(T) {
    std.log.debug("\n ..... IMPORTING THE MODEL ......", .{});

    var file = try std.fs.cwd().openFile(file_path, .{});
    std.log.debug("\n ..... file created ......", .{});

    defer file.close();
    const reader = file.reader();
    std.log.debug("\n ..... reader created ......", .{});

    var model: Model(T) = Model(T){
        .layers = undefined,
        .allocator = allocator,
        .input_tensor = undefined,
    };
    try model.init();

    const n_layers = try reader.readInt(usize, std.builtin.Endian.big);
    for (0..n_layers) |_| {
        const newLayer: Layer.Layer(T) = try importLayer(T, allocator, reader);

        try model.addLayer(newLayer);
    }
    return model;
}

pub fn importLayer(
    comptime T: type,
    allocator: *const std.mem.Allocator,
    reader: anytype, // TODO: should be an AnyReader eventually
) !Layer.Layer(T) {
    std.log.debug("\n ... import layer... ", .{});

    var layer_type_string: [10]u8 = undefined;
    _ = try reader.read(&layer_type_string);
    std.log.debug("{s}", .{layer_type_string});

    //TODO: handle Default layer and null layer
    if (std.mem.eql(u8, &layer_type_string, "Dense.....")) {
        const denseLayerPtr = try allocator.create(DenseLayer(T));

        denseLayerPtr.* = try importLayerDense(T, allocator, reader);
        // Transfer ownership to the Layer
        const newLayer = DenseLayer(T).create(denseLayerPtr);
        defer {} // Cancel previous defer since ownership is transferred

        return newLayer;
    } else if (std.mem.eql(u8, &layer_type_string, "Activation")) {
        return ActivationLayer(T).create(
            @constCast(
                &try importLayerActivation(T, allocator, reader),
            ),
        );
    } else {
        return error.impossibleLayer;
    }
}

pub fn importLayerDense(
    comptime T: type,
    allocator: *const std.mem.Allocator,
    reader: anytype,
) !DenseLayer(T) {
    const weights_tens: Tensor(T) = try importTensor(T, allocator, reader);
    const bias_tens: Tensor(T) = try importTensor(T, allocator, reader);
    const n_inputs = try reader.readInt(usize, std.builtin.Endian.big);
    const n_neurons = try reader.readInt(usize, std.builtin.Endian.big);
    const w_grad_tens = try importTensor(T, allocator, reader);
    const b_grad_tens = try importTensor(T, allocator, reader);

    return DenseLayer(f64){
        .weights = weights_tens,
        .bias = bias_tens,
        .input = undefined,
        .output = undefined,
        .n_inputs = n_inputs,
        .n_neurons = n_neurons,
        .w_gradients = w_grad_tens,
        .b_gradients = b_grad_tens,
        .allocator = allocator,
    };
}

pub fn importLayerActivation(
    comptime T: type,
    allocator: *const std.mem.Allocator,
    reader: anytype,
) !ActivationLayer(T) {
    const n_inputs = try reader.readInt(usize, std.builtin.Endian.big);
    const n_neurons = try reader.readInt(usize, std.builtin.Endian.big);

    var activation_type_string: [10]u8 = undefined;
    _ = try reader.read(&activation_type_string);

    var layerActiv = ActivationLayer(T){
        .input = undefined, //input_tens,
        .output = undefined, //output_tens,
        .n_inputs = n_inputs,
        .n_neurons = n_neurons,
        .activationFunction = undefined,
        .allocator = allocator,
    };

    if (std.mem.eql(u8, &activation_type_string, "ReLU......")) {
        layerActiv.activationFunction = ActivationType.ReLU;
    } else if (std.mem.eql(u8, &activation_type_string, "Sigmoid...")) {
        layerActiv.activationFunction = ActivationType.Sigmoid;
    } else if (std.mem.eql(u8, &activation_type_string, "Softmax...")) {
        layerActiv.activationFunction = ActivationType.Softmax;
    } else if (std.mem.eql(u8, &activation_type_string, "None......")) {
        layerActiv.activationFunction = ActivationType.None;
    }

    return layerActiv;
}

pub fn importTensor(
    comptime T: type,
    allocator: *const std.mem.Allocator,
    reader: anytype,
) !Tensor(T) {
    std.log.debug("\n ... import tensor... ", .{});

    // Read tensor size
    const tensor_size: usize = try reader.readInt(usize, std.builtin.Endian.big);

    // Read tensor shape lenght
    const tensor_shapeLen: usize = try reader.readInt(usize, std.builtin.Endian.big);

    // Read tensor shape
    const tensor_shape = try allocator.alloc(usize, tensor_shapeLen);
    for (0..tensor_shapeLen) |i| {
        tensor_shape[i] = try reader.readInt(usize, std.builtin.Endian.big);
    }

    const tensor_data = try allocator.alloc(T, tensor_size);
    // Read tensor data
    for (0..tensor_size) |i| {
        tensor_data[i] = try readNumber(T, reader);
    }

    return Tensor(T){
        .data = tensor_data,
        .size = tensor_size,
        .shape = tensor_shape,
        .allocator = allocator,
    };
}

inline fn readNumber(
    comptime T: type,
    reader: anytype,
) !T {
    const size = @sizeOf(T);
    var buffer: [size]u8 = undefined;
    _ = try reader.readAll(&buffer);
    return @bitCast(buffer);
}
