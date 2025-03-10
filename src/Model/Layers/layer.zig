//! This file contains the definition of the layers that can be used in the neural network.
//! There are function to initiialize random weigths, initialization right now is completely random but in the future
//! it will possible to use proper initialization techniques.
//! Layer can be stacked in a model and they implement proper forward and backward methods.

// All existing layers are imported so that the layer module can be used as a layer library by other modules.
// New layers should be imported here.
pub const ActivationLayer = @import("activationLayer.zig").ActivationLayer;
pub const ActivationType = @import("activationLayer.zig").ActivationType;
pub const BatchNormLayer = @import("batchNormLayer.zig").BatchNormLayer;
pub const ConvolutionalLayer = @import("convLayer.zig").ConvolutionalLayer;
pub const DenseLayer = @import("denseLayer.zig").DenseLayer;
pub const FlattenLayer = @import("flattenLayer.zig").FlattenLayer;
pub const PoolingLayer = @import("poolingLayer.zig").PoolingLayer;
pub const poolingLayer = @import("poolingLayer.zig");

const std = @import("std");
const zant = @import("../../zant.zig");
const Tensor = zant.core.tensor.Tensor;
const TensMath = zant.core.tensor.math_standard;
const TensorError = TensMath.TensorError;
const ArchitectureError = TensMath.ArchitectureError;

//import error libraries
const LayerError = zant.utils.error_handler.LayerError;

pub const LayerType = enum {
    DenseLayer,
    DefaultLayer,
    ConvolutionalLayer,
    PoolingLayer,
    ActivationLayer,
    FlattenLayer,
    BatchNormLayer,
    null,
};

/// ------------------------------------------------------------------------------------------------------
/// Initialize a vector of random values with a normal distribution
/// TODO: improve the random number generation, at the moment is generated by an hardcoded seed. it would be great if the user has a custom generation.
pub fn randn(comptime T: type, allocator: *const std.mem.Allocator, n_inputs: usize, n_neurons: usize) ![]T {
    var rng = std.Random.Xoshiro256.init(@intCast(std.time.timestamp()));
    const vector = try allocator.alloc(T, n_inputs * n_neurons);

    // He initialization: sqrt(2/n_inputs)
    const scale = @sqrt(2.0 / @as(T, @floatFromInt(n_inputs)));

    for (0..vector.len) |i| {
        vector[i] = rng.random().floatNorm(T) * scale;
    }
    return vector;
}

/// Function used to initialize a vector of zeros used for bias.
/// Pay attention, when using Tensor.fromShape() already initialize a tensor of zeros. Do not use zeros() and then Tensor.fromArray() because is redundant.
pub fn zeros(comptime T: type, allocator: *const std.mem.Allocator, n_inputs: usize, n_neurons: usize) ![]T {
    const vector = try allocator.alloc(T, n_inputs * n_neurons);
    for (0..n_inputs) |i| {
        for (0..n_neurons) |j| {
            vector[i * n_neurons + j] = 0; // TODO: fix me!! why +1 ??
        }
    }
    return vector;
}

//------------------------------------------------------------------------------------------------------
/// INTERFACE LAYER
///
/// Layer() is the superclass for all the possible implementation of a layer (Activation, Dense, Conv ... see /Layer folder).
///
/// @param T:comptime type of the values in the layer
pub fn Layer(comptime T: type) type {
    return struct {
        layer_type: LayerType,
        layer_ptr: *anyopaque,
        layer_impl: *const Basic_Layer_Interface,

        const Self = @This();

        /// Interface methods to be implemented
        pub const Basic_Layer_Interface = struct {
            init: *const fn (ctx: *anyopaque, allocator: *const std.mem.Allocator, args: *anyopaque) anyerror!void,
            deinit: *const fn (ctx: *anyopaque) void,
            forward: *const fn (ctx: *anyopaque, input: *Tensor(T)) anyerror!Tensor(T),
            backward: *const fn (ctx: *anyopaque, dValues: *Tensor(T)) anyerror!Tensor(T),
            printLayer: *const fn (ctx: *anyopaque, choice: u8) void,
            get_n_inputs: *const fn (ctx: *anyopaque) usize,
            get_n_neurons: *const fn (ctx: *anyopaque) usize,
            get_input: *const fn (ctx: *anyopaque) *const Tensor(T),
            get_output: *const fn (ctx: *anyopaque) *Tensor(T),
        };

        pub fn init(self: Self, alloc: *const std.mem.Allocator, args: *anyopaque) anyerror!void {
            return self.layer_impl.init(self.layer_ptr, alloc, args);
        }

        /// When deinit() pay attention to:
        /// - Double-freeing memory.
        /// - Using uninitialized or already-deallocated pointers.
        /// - Incorrect allocation or deallocation logic.
        ///
        pub fn deinit(self: Self) void {
            return self.layer_impl.deinit(self.layer_ptr);
        }
        pub fn forward(self: Self, input: *Tensor(T)) !Tensor(T) {
            return self.layer_impl.forward(self.layer_ptr, input);
        }
        pub fn backward(self: Self, dValues: *Tensor(T)) !Tensor(T) {
            return self.layer_impl.backward(self.layer_ptr, dValues);
        }
        pub fn printLayer(self: Self, choice: u8) void {
            return self.layer_impl.printLayer(self.layer_ptr, choice);
        }
        pub fn get_n_inputs(self: Self) usize {
            return self.layer_impl.get_n_inputs(self.layer_ptr);
        }
        pub fn get_n_neurons(self: Self) usize {
            return self.layer_impl.get_n_neurons(self.layer_ptr);
        }
        pub fn get_input(self: Self) *const Tensor(T) {
            return self.layer_impl.get_input(self.layer_ptr);
        }
        pub fn get_output(self: Self) *Tensor(T) {
            return self.layer_impl.get_output(self.layer_ptr);
        }
    };
}
