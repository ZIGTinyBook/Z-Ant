const std = @import("std");
const zant = @import("../../../zant.zig");

const Tensor = zant.core.tensor.Tensor; // Import Tensor type
const pkg_allocator = zant.utils.allocator.allocator;
const error_handler = zant.utils.error_handler;
const TensorMathError = error_handler.TensorMathError;
const TensorError = error_handler.TensorError;
const ArchitectureError = error_handler.ArchitectureError;
const Converter = zant.utils.type_converter;

/// ReLU (Rectified Linear Unit).
/// It outputs the input directly if it's positive, but returns zero for any negative input.
pub inline fn ReLU_standard(comptime T: anytype, tensor: *Tensor(T)) !Tensor(T) {
    var output = try Tensor(T).fromShape(&pkg_allocator, tensor.shape);
    if (tensor.size <= 0) return TensorError.ZeroSizeTensor;
    try lean_ReLU(T, tensor, &output);

    return output;
}

pub inline fn lean_ReLU(comptime T: anytype, input_tensor: *Tensor(T), output_tensor: *Tensor(T)) !void {
    //apply ReLU
    //OSS: can be improved, see how did I parallelized CPU Tensor Sum
    for (0..input_tensor.size) |i| {
        if (input_tensor.data[i] > 0) output_tensor.data[i] = input_tensor.data[i];
    }
}

pub inline fn ReLU_backward(comptime T: anytype, gradient: *Tensor(T), act_relu_input: *Tensor(T)) !void {

    //checks
    if (gradient.size <= 0 or act_relu_input.size <= 0) return TensorError.ZeroSizeTensor;
    if (gradient.size != act_relu_input.size) return TensorMathError.InputTensorDifferentSize;

    //apply ReLU derivative: f'(x) = 0 if x <= 0, 1 if x > 0
    for (0..gradient.size) |i| {
        if (act_relu_input.data[i] <= 0) {
            gradient.data[i] = 0;
        }
        // else gradient remains unchanged since f'(x) = 1 for x > 0
    }
}

/// Leaky ReLU is a variant of ReLU that allows a small, positive gradient when the input is negative.
/// This can help prevent the dying ReLU problem, where neurons stop learning if they get stuck in the negative side of the ReLU function.
/// The leaky parameter is a small value that determines how much the function leaks into the negative side.
/// A common value for the leaky parameter is 0.01.
/// The Leaky ReLU function is defined as:
/// f(x) = x if x > 0
/// f(x) = alpha * x if x <= 0
/// where alpha is a small constant.
/// The derivative of the Leaky ReLU function is:
/// f'(x) = 1 if x > 0
/// f'(x) = alpha if x <= 0
pub inline fn leakyReLU(comptime T: anytype, tensor: *Tensor(T), slope: T) !Tensor(T) {
    //checks
    if (tensor.size <= 0) return TensorError.ZeroSizeTensor;

    var output_tensor = try Tensor(T).fromShape(&pkg_allocator, tensor.shape);
    errdefer output_tensor.deinit();

    try lean_leakyReLU(T, tensor, slope, &output_tensor);

    return output_tensor;
}

pub inline fn lean_leakyReLU(comptime T: anytype, input_tensor: *Tensor(T), slope: T, output_tensor: *Tensor(T)) !void {
    //apply Leaky ReLU suing relu self.relu() - (-neg_slope*self).relu()
    for (0..input_tensor.size) |i| {
        if (input_tensor.data[i] <= 0) {
            output_tensor.data[i] = slope * input_tensor.data[i];
        } else {
            output_tensor.data[i] = input_tensor.data[i];
        }
    }
}

pub fn leakyReLU_backward(comptime T: anytype, gradient: *Tensor(T), act_relu_input: *Tensor(T), slope: T) !void {

    //checks
    if (gradient.size <= 0 or act_relu_input.size <= 0) return TensorError.ZeroSizeTensor;
    if (gradient.size != act_relu_input.size) return TensorMathError.InputTensorDifferentSize;

    //apply Leaky ReLU derivative: f'(x) = 1 if x > 0, slope if x <= 0
    for (0..gradient.size) |i| {
        gradient.data[i] *= if (act_relu_input.data[i] > 0) 1 else slope;
    }
}

/// The Sigmoid activation function is a smooth, S-shaped function that maps any input
/// to a value between 0 and 1.
/// it can suffer from vanishing gradients, especially for large positive or negative
/// inputs, slowing down training in deep networks.
pub inline fn sigmoid(comptime T: anytype, tensor: *Tensor(T)) !Tensor(T) {
    //checks
    if (tensor.size <= 0) return TensorError.ZeroSizeTensor;

    var output_tensor = try Tensor(T).fromShape(&pkg_allocator, tensor.shape);
    errdefer output_tensor.deinit();

    try sigmoid_lean(T, tensor, &output_tensor);

    return output_tensor;
}

pub inline fn sigmoid_lean(comptime T: anytype, input_tensor: *Tensor(T), output_tensor: *Tensor(T)) !void {
    //std.debug.print("\n[DEBUG] sigmoid_lean:", .{});
    //std.debug.print("\n  Input shape: ", .{});
    //for (input_tensor.shape) |s| std.debug.print("{d} ", .{s});

    //std.debug.print("\n  Output shape: ", .{});
    //for (output_tensor.shape) |s| std.debug.print("{d} ", .{s});

    //apply Sigmoid
    for (0..input_tensor.size) |i| {
        const input_val = input_tensor.data[i];
        output_tensor.data[i] = 1.0 / (1.0 + @exp(-input_val));
        //std.debug.print("\n  sigmoid({d:.6}) = {d:.6}", .{ input_val, output_tensor.data[i] });
    }
    //std.debug.print("\n[DEBUG] sigmoid_lean completed\n", .{});
}

pub fn sigmoid_backward(comptime T: anytype, gradient: *Tensor(T), act_forward_out: *Tensor(T)) !void {
    //checks
    if (gradient.size <= 0 or act_forward_out.size <= 0) return TensorError.ZeroSizeTensor;
    if (gradient.size != act_forward_out.size) return TensorMathError.InputTensorDifferentSize;

    //apply Sigmoid derivative: f'(x) = f(x) * (1 - f(x))
    for (0..gradient.size) |i| {
        const sigmoid_output = act_forward_out.data[i];
        gradient.data[i] *= sigmoid_output * (1 - sigmoid_output);
    }
}

/// The Softmax activation function is used in multi-class classification tasks to convert
/// logits (raw output values) into probabilities that sum to 1.
/// Ideal for output layers in multi-class neural networks.
pub fn softmax(comptime T: anytype, tensor: *Tensor(T)) !Tensor(T) {

    // TODO: add more robust checks
    //checks
    if (tensor.size <= 0) return TensorError.ZeroSizeTensor;

    var output_tensor = try Tensor(T).fromShape(&pkg_allocator, tensor.shape);
    errdefer output_tensor.deinit();

    //compute
    try lean_softmax(T, tensor, &output_tensor);

    return output_tensor;
}

pub inline fn lean_softmax(comptime T: anytype, input: *Tensor(T), output: *Tensor(T)) !void {
    const rows = input.shape[0];
    const cols = input.shape[1];

    var max_val: T = undefined;
    var sum_of_exp: T = 0.0;
    var val: T = undefined;

    // For each row
    for (0..rows) |i| {
        // Find the maximum value in the row to stabilize the computation
        max_val = input.data[i * cols];
        for (0..cols) |j| {
            val = input.data[i * cols + j];
            if (val > max_val) {
                max_val = val;
            }
        }

        // Compute stabilized exponentials and their sum
        sum_of_exp = 0.0;
        for (0..cols) |j| {
            val = input.data[i * cols + j] - max_val; // Stabilization
            val = @exp(val);
            output.data[i * cols + j] = val;
            sum_of_exp += val;
        }

        // Normalize to calculate the softmax
        for (0..cols) |j| {
            output.data[i * cols + j] /= sum_of_exp;
        }
    }
}

pub fn softmax_backward(comptime T: anytype, dL_dX: *Tensor(T), softmax_output: *Tensor(T)) !void {
    //checks
    if (dL_dX.size <= 0) return TensorError.ZeroSizeTensor;
    if (dL_dX.size != softmax_output.size) return TensorMathError.InputTensorDifferentSize;

    const dim = dL_dX.shape.len;
    const rows = dL_dX.shape[dim - 2];
    const cols = dL_dX.shape[dim - 1];

    // Make a copy of incoming gradients since we'll modify dL_dX
    var dL_dS = try dL_dX.copy();
    defer dL_dS.deinit();

    // For each sample in the batch
    for (0..rows) |i| {
        const row_offset = i * cols;
        // For each output neuron j
        for (0..cols) |j| {
            var sum: T = 0;
            const sj = softmax_output.data[row_offset + j];

            // Compute sum of dL/dS_k * S_k for all k
            for (0..cols) |k| {
                const sk = softmax_output.data[row_offset + k];
                const dL_dS_k = dL_dS.data[row_offset + k];
                if (j == k) {
                    sum += dL_dS_k * sk * (1 - sj); // Diagonal term
                } else {
                    sum += dL_dS_k * (-sk * sj); // Off-diagonal term
                }
            }
            dL_dX.data[row_offset + j] = sum;
        }
    }
}
