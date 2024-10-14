const std = @import("std");
const tensor = @import("tensor");
const layer = @import("layers");
const Model = @import("model").Model;
const loader = @import("dataloader");
const ActivationType = @import("activation_function").ActivationType;
const LossType = @import("loss").LossType;
const Trainer = @import("trainer");

pub fn main() !void {
    const allocator = std.heap.page_allocator;

    var model = Model(f64, &allocator){
        .layers = undefined,
        .allocator = &allocator,
        .input_tensor = undefined,
    };
    try model.init();
    var rng = std.Random.Xoshiro256.init(12345);

    var layer1 = layer.DenseLayer(f64, &allocator){
        .weights = undefined,
        .bias = undefined,
        .input = undefined,
        .output = undefined,
        .outputActivation = undefined,
        .n_inputs = 0,
        .n_neurons = 0,
        .w_gradients = undefined,
        .b_gradients = undefined,
        .allocator = undefined,
        .activationFunction = ActivationType.None,
    };
    var layer1_ = layer.Layer(f64, &allocator){
        .denseLayer = &layer1,
    };
    try layer1_.init(784, 32, &rng);
    try model.addLayer(&layer1_);

    var layer2 = layer.DenseLayer(f64, &allocator){
        .weights = undefined,
        .bias = undefined,
        .input = undefined,
        .output = undefined,
        .outputActivation = undefined,
        .n_inputs = 0,
        .n_neurons = 0,
        .w_gradients = undefined,
        .b_gradients = undefined,
        .allocator = undefined,
        .activationFunction = ActivationType.None,
    };
    //layer 2: 2 inputs, 5 neurons
    var layer2_ = layer.Layer(f64, &allocator){
        .denseLayer = &layer2,
    };
    try layer2_.init(32, 32, &rng);
    try model.addLayer(&layer2_);

    var layer3 = layer.DenseLayer(f64, &allocator){
        .weights = undefined,
        .bias = undefined,
        .input = undefined,
        .output = undefined,
        .outputActivation = undefined,
        .n_inputs = 0,
        .n_neurons = 0,
        .w_gradients = undefined,
        .b_gradients = undefined,
        .allocator = undefined,
        .activationFunction = ActivationType.Softmax,
    };
    var layer3_ = layer.Layer(f64, &allocator){
        .denseLayer = &layer3,
    };
    try layer3_.init(32, 10, &rng);
    try model.addLayer(&layer3_);

    var load = loader.DataLoader(f64, u8, u8, 16){
        .X = undefined,
        .y = undefined,
        .xTensor = undefined,
        .yTensor = undefined,
        .XBatch = undefined,
        .yBatch = undefined,
    };

    // const file_name: []const u8 = "dataset_regressione.csv";
    // const features = [_]usize{ 0, 1, 2, 3, 4 };
    // const featureCols: []const usize = &features;
    // const labelCol: usize = 5;
    // try load.fromCSV(&allocator, file_name, featureCols, labelCol);

    const image_file_name: []const u8 = "t10k-images-idx3-ubyte";
    const label_file_name: []const u8 = "t10k-labels-idx1-ubyte";

    try load.loadMNISTDataParallel(&allocator, image_file_name, label_file_name);

    try Trainer.TrainDataLoader(f64, u8, u8, &allocator, 16, 784, &model, &load, 100, LossType.CCE, 0.05);

    //std.debug.print("Output tensor shape: {any}\n", .{output.shape});
    //std.debug.print("Output tensor data: {any}\n", .{output.data});

    model.deinit();
}
