//! This file contains the definition of the layers that can be used in the neural network.
//! There are function to initiialize random weigths, initialization right now is completely random but in the future
//! it will possible to use proper initialization techniques.
//! Layer can be stacked in a model and they implement proper forward and backward methods.

const std = @import("std");
const tensor = @import("tensor");
const TensMath = @import("tensor_m");
const Architectures = @import("architectures").Architectures;
const TensorError = @import("tensor_m").TensorError;
const ArchitectureError = @import("tensor_m").ArchitectureError;
const ActivLib = @import("activation_function");
const ActivationType = @import("activation_function").ActivationType;
//import error libraries
const LayerError = @import("errorHandler").LayerError;

pub const LayerType = enum {
    DenseLayer,
    DefaultLayer,
    ConvolutionalLayer,
    PoolingLayer,
    ActivationLayer,
    FlattenLayer,
    null,
};

//------------------------------------------------------------------------------------------------------
/// UTILS
/// Initialize a vector of random values with a normal distribution
pub fn randn(comptime T: type, allocator: *const std.mem.Allocator, n_inputs: usize, n_neurons: usize) ![]T {
    var rng = std.Random.Xoshiro256.init(12345);

    const vector = try allocator.alloc(T, n_inputs * n_neurons);
    for (0..n_inputs) |i| {
        for (0..n_neurons) |j| {
            vector[i * n_neurons + j] = rng.random().floatNorm(T) + 1; // TODO: fix me!! why +1 ??
        }
    }
    return vector;
}
///Function used to initialize a vector of zeros used for bias
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
// INTERFACE LAYER

pub fn Layer(comptime T: type) type {
    return struct {
        layer_type: LayerType,
        layer_ptr: *anyopaque,
        layer_impl: *const Basic_Layer_Interface,

        const Self = @This();

        pub const Basic_Layer_Interface = struct {
            init: *const fn (ctx: *anyopaque, allocator: *const std.mem.Allocator, args: *anyopaque) anyerror!void,
            deinit: *const fn (ctx: *anyopaque) void,
            forward: *const fn (ctx: *anyopaque, input: *tensor.Tensor(T)) anyerror!tensor.Tensor(T),
            backward: *const fn (ctx: *anyopaque, dValues: *tensor.Tensor(T)) anyerror!tensor.Tensor(T),
            printLayer: *const fn (ctx: *anyopaque, choice: u8) void,
            get_n_inputs: *const fn (ctx: *anyopaque) usize,
            get_n_neurons: *const fn (ctx: *anyopaque) usize,
            get_input: *const fn (ctx: *anyopaque) *const tensor.Tensor(T),
            get_output: *const fn (ctx: *anyopaque) *tensor.Tensor(T),
            codegen: *const fn (ctx: *anyopaque, writer: std.fs.File.Writer) std.posix.WriteError!void,
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
        pub fn forward(self: Self, input: *tensor.Tensor(T)) !tensor.Tensor(T) {
            return self.layer_impl.forward(self.layer_ptr, input);
        }
        pub fn backward(self: Self, dValues: *tensor.Tensor(T)) !tensor.Tensor(T) {
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
        pub fn get_input(self: Self) *const tensor.Tensor(T) {
            return self.layer_impl.get_input(self.layer_ptr);
        }
        pub fn get_output(self: Self) *tensor.Tensor(T) {
            return self.layer_impl.get_output(self.layer_ptr);
        }
        pub fn codegen(self: Self, writer: std.fs.File.Writer) !void {
            return self.layer_impl.codegen(self.layer_ptr, writer);
        }
    };
}
