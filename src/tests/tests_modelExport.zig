const std = @import("std");
const model_import_export = @import("model_import_export");
const Model = @import("model").Model;
const layer = @import("layer");
const denselayer = @import("denselayer");
const actlayer = @import("activationlayer");
const Tensor = @import("tensor").Tensor;
const ActivationType = @import("activation_function").ActivationType;
const Trainer = @import("trainer");

test "Import/Export of a tensor" {
    std.debug.print("\n     test: Import/Export of a tensor", .{});

    const allocator = std.heap.page_allocator; // Known memory leak, avoid testing allocator
    const file_path = "importExportTensorTestFile.bin";
    //EXPORT
    var file = try std.fs.cwd().createFile(file_path, .{});
    const writer = file.writer();

    var inputArray: [2][2]f32 = [_][2]f32{
        [_]f32{ 1.0, 2.0 },
        [_]f32{ 4.0, 5.0 },
    };

    var shape: [2]usize = [_]usize{ 2, 2 }; // 2x2 matrix

    var t1 = try Tensor(f32).fromArray(&allocator, &inputArray, &shape);
    defer t1.deinit();

    try model_import_export.exportTensor(f32, t1, writer);
    file.close();

    //IMPORT
    file = try std.fs.cwd().openFile(file_path, .{});
    const reader = file.reader();
    var t2: Tensor(f32) = try model_import_export.importTensor(f32, &allocator, reader);
    defer t2.deinit();

    file.close();

    //same data
    try std.testing.expect(t1.data.len == t2.data.len);
    for (0..t1.data.len) |i| {
        try std.testing.expect(t1.data[i] == t2.data[i]);
    }

    //same size
    try std.testing.expect(t1.size == t2.size);

    //same shape
    try std.testing.expect(t1.shape.len == t2.shape.len);
    for (0..t1.shape.len) |i| {
        try std.testing.expect(t1.shape[i] == t2.shape[i]);
    }

    try std.fs.cwd().deleteFile(file_path);
}

test "Import/Export of dense layer" {
    std.debug.print("\n     test: Import/Export of dense layer", .{});
    const allocator = std.heap.page_allocator; //OSS!! denseLayerPtr in importLayer() is not freed
    const file_path = "importExportDenseLayerTestFile.bin";
    //EXPORT
    var file = try std.fs.cwd().createFile(file_path, .{});
    const writer = file.writer();

    var dense_layer1 = denselayer.DenseLayer(f64){
        .weights = undefined,
        .input = undefined,
        .bias = undefined,
        .output = undefined,
        .n_inputs = 0,
        .n_neurons = 0,
        .w_gradients = undefined,
        .b_gradients = undefined,
        .allocator = undefined,
    };
    var layer1_ = denselayer.DenseLayer(f64).create(&dense_layer1);
    try layer1_.init(
        &allocator,
        @constCast(&struct {
            n_inputs: usize,
            n_neurons: usize,
        }{
            .n_inputs = 3,
            .n_neurons = 2,
        }),
    );
    defer layer1_.deinit();

    try model_import_export.exportLayer(f64, layer1_, writer);

    file.close();

    //IMPORT
    file = try std.fs.cwd().openFile(file_path, .{});
    const reader = file.reader();

    var layer_imported = try model_import_export.importLayer(f64, &allocator, reader);
    defer layer_imported.deinit();

    file.close();

    //same type
    std.debug.print("\n same type: {any}={any}", .{ layer_imported.layer_type, layer1_.layer_type });
    try std.testing.expect(layer_imported.layer_type == layer1_.layer_type);

    //same n_neurons
    std.debug.print("\n same n_neurons: {any}={any}", .{ layer_imported.get_n_neurons(), layer1_.get_n_neurons() });
    try std.testing.expect(layer_imported.get_n_neurons() == layer1_.get_n_neurons());

    //same n_inputs
    std.debug.print("\n same n_inputs  {any}={any}", .{ layer_imported.get_n_inputs(), layer1_.get_n_inputs() });
    try std.testing.expect(layer_imported.get_n_inputs() == layer1_.get_n_inputs());

    //check layer
    const denseImportedPtr: *denselayer.DenseLayer(f64) = @alignCast(@ptrCast(layer_imported.layer_ptr));
    //same weights len
    try std.testing.expect(denseImportedPtr.weights.data.len == dense_layer1.weights.data.len);
    //same weights
    for (0..denseImportedPtr.weights.data.len) |i| {
        try std.testing.expect(denseImportedPtr.weights.data[i] == dense_layer1.weights.data[i]);
    }
    //same bias len
    try std.testing.expect(denseImportedPtr.bias.data.len == dense_layer1.bias.data.len);
    //same weights
    for (0..denseImportedPtr.bias.data.len) |i| {
        try std.testing.expect(denseImportedPtr.bias.data[i] == dense_layer1.bias.data[i]);
    }

    try std.fs.cwd().deleteFile(file_path);
}

test "Import/Export of activation layer" {
    std.debug.print("\n     test: Import/Export of activation layer", .{});
    const allocator = std.heap.page_allocator;
    const file_path = "importExportTestFile.bin";
    //EXPORT
    var file = try std.fs.cwd().createFile(file_path, .{});
    const writer = file.writer();

    var activ_layer = actlayer.ActivationLayer(f64){
        .input = undefined,
        .output = undefined,
        .n_inputs = 0,
        .n_neurons = 0,
        .activationFunction = ActivationType.ReLU,
        .allocator = &allocator,
    };
    const layer1_ = actlayer.ActivationLayer(f64).create(&activ_layer);
    // n_input = 5, n_neurons= 4
    try layer1_.init(
        &allocator,
        @constCast(&struct {
            n_inputs: usize,
            n_neurons: usize,
        }{
            .n_inputs = 5,
            .n_neurons = 4,
        }),
    );

    //defer layer1_.deinit();

    try model_import_export.exportLayer(f64, layer1_, writer);

    file.close();

    //IMPORT
    file = try std.fs.cwd().openFile(file_path, .{});
    const reader = file.reader();

    var layer_imported = try model_import_export.importLayer(f64, &allocator, reader);
    //defer layer_imported.deinit();

    file.close();

    //same type layer
    std.debug.print("\n same type: {any}={any}", .{ layer_imported.layer_type, layer1_.layer_type });
    try std.testing.expect(layer_imported.layer_type == layer1_.layer_type);

    const actImportedPtr: *actlayer.ActivationLayer(f64) = @alignCast(@ptrCast(layer_imported.layer_ptr));
    //same type activagtion
    std.debug.print("\n same type: {any}={any}", .{ actImportedPtr.activationFunction, activ_layer.activationFunction });
    try std.testing.expect(actImportedPtr.activationFunction == activ_layer.activationFunction);

    //same n_neurons
    std.debug.print("\n same n_neurons: {any}={any}", .{ layer_imported.get_n_neurons(), layer1_.get_n_neurons() });
    try std.testing.expect(layer_imported.get_n_neurons() == layer1_.get_n_neurons());

    //same n_inputs
    std.debug.print("\n same n_inputs  {any}={any}", .{ layer_imported.get_n_inputs(), layer1_.get_n_inputs() });
    try std.testing.expect(layer_imported.get_n_inputs() == layer1_.get_n_inputs());

    try std.fs.cwd().deleteFile(file_path);
}

// test "Import/Export of a tensor" {
//     std.debug.print("\n     test: Import/Export of a tensor", .{});
//     const allocator = std.heap.page_allocator;
//     const file_path = "importExportTestFile.bin";
//     //EXPORT
//     var file = try std.fs.cwd().createFile(file_path, .{});
//     const writer = file.writer();

//     file.close();

//     //IMPORT
//     file = try std.fs.cwd().openFile(file_path, .{});
//     const reader = file.reader();

//     file.close();

//     try std.fs.cwd().deleteFile(file_path);
// }

test "Export of a complex model" {
    std.debug.print("\n     test: Export of a 2D Tensor", .{});

    const allocator = std.heap.page_allocator;

    var model = try Model(f64).init(&allocator);
    defer model.deinit();

    //layer 1: 3 inputs, 2 neurons
    var layer1 = denselayer.DenseLayer(f64){
        .weights = undefined,
        .bias = undefined,
        .input = undefined,
        .output = undefined,
        .n_inputs = 0,
        .n_neurons = 0,
        .w_gradients = undefined,
        .b_gradients = undefined,
        .allocator = undefined,
    };
    var layer1_ = denselayer.DenseLayer(f64).create(&layer1);
    try layer1_.init(
        &allocator,
        @constCast(&struct {
            n_inputs: usize,
            n_neurons: usize,
        }{
            .n_inputs = 3,
            .n_neurons = 2,
        }),
    );
    try model.addLayer(layer1_);

    //layer 1: 3 inputs, 2 neurons
    var layer1Activ = actlayer.ActivationLayer(f64){
        .input = undefined,
        .output = undefined,
        .n_inputs = 0,
        .n_neurons = 0,
        .activationFunction = ActivationType.ReLU,
        .allocator = &allocator,
    };
    var layer1Activ_ = actlayer.ActivationLayer(f64).create(&layer1Activ);
    try layer1Activ_.init(
        &allocator,
        @constCast(&struct {
            n_inputs: usize,
            n_neurons: usize,
        }{
            .n_inputs = 2,
            .n_neurons = 2,
        }),
    );
    try model.addLayer(layer1Activ_);

    //layer 2: 2 inputs, 5 neurons
    var layer2 = denselayer.DenseLayer(f64){
        .weights = undefined,
        .bias = undefined,
        .input = undefined,
        .output = undefined,
        .n_inputs = 0,
        .n_neurons = 0,
        .w_gradients = undefined,
        .b_gradients = undefined,
        .allocator = undefined,
    };
    var layer2_ = denselayer.DenseLayer(f64).create(&layer2);
    try layer2_.init(
        &allocator,
        @constCast(&struct {
            n_inputs: usize,
            n_neurons: usize,
        }{
            .n_inputs = 2,
            .n_neurons = 5,
        }),
    );
    try model.addLayer(layer2_);

    var layer2Activ = actlayer.ActivationLayer(f64){
        .input = undefined,
        .output = undefined,
        .n_inputs = 0,
        .n_neurons = 0,
        .activationFunction = ActivationType.Softmax,
        .allocator = &allocator,
    };
    var layer2Activ_ = actlayer.ActivationLayer(f64).create(&layer2Activ);
    try layer2Activ_.init(
        &allocator,
        @constCast(&struct {
            n_inputs: usize,
            n_neurons: usize,
        }{
            .n_inputs = 2,
            .n_neurons = 5,
        }),
    );
    try model.addLayer(layer2Activ_);

    var inputArray: [2][3]f64 = [_][3]f64{
        [_]f64{ 1.0, 2.0, 3.0 },
        [_]f64{ 4.0, 5.0, 6.0 },
    };
    var shape: [2]usize = [_]usize{ 2, 3 };

    var targetArray: [2][5]f64 = [_][5]f64{
        [_]f64{ 1.0, 2.0, 3.0, 4.0, 5.0 },
        [_]f64{ 4.0, 5.0, 6.0, 4.0, 5.0 },
    };
    var targetShape: [2]usize = [_]usize{ 2, 5 };

    var input_tensor = try Tensor(f64).fromArray(&allocator, &inputArray, &shape);
    defer {
        input_tensor.deinit();
        std.debug.print("\n -.-.-> input_tensor deinitialized", .{});
    }

    var target_tensor = try Tensor(f64).fromArray(&allocator, &targetArray, &targetShape);
    defer {
        target_tensor.deinit();
        std.debug.print("\n -.-.-> target_tensor deinitialized", .{});
    }

    try Trainer.trainTensors(
        f64, //type
        &allocator, //allocator
        &model, //model
        &input_tensor, //input
        &target_tensor, //target
        5, //epochs
        0.5, //learning rate
    );

    layer1Activ_.printLayer(1);

    try model_import_export.exportModel(f64, model, "exportTryModel.bin");

    _ = try model_import_export.importModel(f64, &allocator, "exportTryModel.bin");
}
