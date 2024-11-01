const std = @import("std");
const tensor = @import("tensor");
const layer = @import("layers");
const Model = @import("model").Model;
const loader = @import("dataloader");
const ActivationType = @import("activation_function").ActivationType;
const LossType = @import("loss").LossType;
const Trainer = @import("trainer");

pub fn main() !void {
    const allocator = std.heap.raw_c_allocator;

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
        .activationFunction = ActivationType.ReLU,
    };
    var layer1_ = layer.Layer(f64, &allocator){
        .denseLayer = &layer1,
    };
    try layer1_.init(784, 64, &rng);
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
        .activationFunction = ActivationType.ReLU,
    };
    //layer 2: 2 inputs, 5 neurons
    var layer2_ = layer.Layer(f64, &allocator){
        .denseLayer = &layer2,
    };
    try layer2_.init(64, 64, &rng);
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
    try layer3_.init(64, 10, &rng);
    try model.addLayer(&layer3_);

    var load = loader.DataLoader(f64, u8, u8, 100){
        .X = undefined,
        .y = undefined,
        .xTensor = undefined,
        .yTensor = undefined,
        .XBatch = undefined,
        .yBatch = undefined,
    };

    const image_file_name: []const u8 = "t10k-images-idx3-ubyte";
    const label_file_name: []const u8 = "t10k-labels-idx1-ubyte";

    try load.loadMNISTDataParallel(&allocator, image_file_name, label_file_name);

    try Trainer.TrainDataLoader(
        f64, //The data type for the tensor elements in the model
        u8, //The data type for the input tensor (X)
        u8, //The data type for the output tensor (Y)
        &allocator, //Memory allocator for dynamic allocations during training
        100, //The number of samples in each batch
        784, //The number of features in each input sample
        &model, //A pointer to the model to be trained
        &load, //A pointer to the `DataLoader` that provides data batches
        10, //The total number of epochs to train for
        LossType.CCE, //The type of loss function used during training
        0.005, //The learning rate for model optimization
    );

    model.deinit();
}
