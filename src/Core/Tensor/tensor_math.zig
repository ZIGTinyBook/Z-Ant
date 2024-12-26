//! Tensor math contains all the functions to perform operations on tensors
const std = @import("std");
const Tensor = @import("tensor").Tensor; // Import Tensor type
const Architectures = @import("architectures").Architectures; //Import Architectures type
const Converter = @import("typeC");
const Layer = @import("Layer");

const PoolingType = @import("poolingLayer").PoolingType;

//import error libraries
const TensorMathError = @import("errorHandler").TensorMathError;
const ArchitectureError = @import("errorHandler").ArchitectureError;
const TensorError = @import("errorHandler").TensorError;

/// Function that add the bias for all the features in the tensor
pub fn add_bias(comptime T: anytype, allocator: *const std.mem.Allocator, tensor: *Tensor(T), bias: *Tensor(T)) !void {
    // Checks:
    if (tensor.size == 0) {
        return TensorError.EmptyTensor;
    }
    if (bias.size == 0) {
        return TensorError.EmptyTensor;
    }
    if (bias.shape.len != 1) {
        return TensorMathError.InputTensorsWrongShape;
    }
    const len = bias.shape[0];
    if (len != tensor.shape[tensor.shape.len - 1]) {
        return TensorMathError.InputTensorDimensionMismatch;
    }

    // Allocate an array for threads, one for each row of the tensor
    //const num_threads = tensor.size / bias.size;

    //var threads = try allocator.alloc(std.Thread, num_threads); //Array to save thread handles

    var index: usize = 0;
    var i: usize = 0;

    _ = allocator;
    // Start a thread for each row of the tensor
    while (index < tensor.size) : (i += 1) {
        //threads[i] = try std.Thread.spawn(.{}, add_bias_thread, .{ T, tensor.data, index, len, bias });
        add_bias_thread(T, tensor.data, index, len, bias);
        index += len;
    }

    // Merges all threads
    //for (threads) |*thread| {
    //thread.join(); // Use try to catch any errors
    //}

    // Free the thread array
    //allocator.free(threads);
}

fn add_bias_thread(comptime T: anytype, array: []T, start: usize, len: usize, bias: *Tensor(T)) void {
    for (0..len) |i| {
        array[start + i] += bias.data[i];
    }
}
/// Performs the mean of a given tensor. It is a reduction operation, collapsing the whole tenosr into a single value.
pub fn mean(comptime T: anytype, tensor: *Tensor(T)) f32 {
    var res: f32 = 0;

    for (tensor.data) |*d| {
        res += Converter.convert(T, f32, d.*);
    }
    res = res / Converter.convert(usize, f32, tensor.size);
    return res;
}

///Returns a Tensor with the same shape pf t1 and t2, where each element --> out[location] = t1[location] + t2[location]
pub fn sum_tensors(comptime arch: Architectures, comptime Tin: anytype, comptime Tout: anytype, t1: *Tensor(Tin), t2: *Tensor(Tin)) !Tensor(Tout) {

    //selecting between all possible architectures
    return switch (arch) {
        Architectures.CPU => return CPU_sum_tensors(Tin, Tout, t1, t2),

        Architectures.GPU => {
            std.log.debug("{} is under developement \n", .{arch});
            return ArchitectureError.UnderDevelopementArchitecture;
        },
        Architectures.SP32 => {
            std.log.debug("{} is under developement \n", .{arch});
            return ArchitectureError.UnderDevelopementArchitecture;
        },
        else => return ArchitectureError.UnknownArchitecture,
    };
}

//Return the sum of the tensors inside another Tensor (t3)
fn CPU_sum_tensors(comptime inputType: anytype, comptime outputType: anytype, t1: *Tensor(inputType), t2: *Tensor(inputType)) !Tensor(outputType) {
    // CHECKS:
    if (t1.size != t2.size) return TensorMathError.InputTensorDifferentSize;

    if (@bitSizeOf(outputType) <= 16) { // quantized
        if (@bitSizeOf(outputType) <= (@bitSizeOf(inputType) * 2)) return TensorMathError.TooSmallOutputType;
    } else { // non-quant
        if (@bitSizeOf(outputType) < @bitSizeOf(inputType)) return TensorMathError.TooSmallOutputType;
    }

    // Allocating the array for the sum
    var out_sum = try t1.allocator.alloc(outputType, t1.size);
    defer t1.allocator.free(out_sum); // Ensure out_sum gets freed in case of error

    var i: usize = 0;
    const unroll_factor: usize = 4;

    // Loop unrolling
    while (i + unroll_factor <= t1.size) : (i += 4) {
        out_sum[i] = t1.data[i] + t2.data[i];
        out_sum[i + 1] = t1.data[i + 1] + t2.data[i + 1];
        out_sum[i + 2] = t1.data[i + 2] + t2.data[i + 2];
        out_sum[i + 3] = t1.data[i + 3] + t2.data[i + 3];
    }

    // Handle any remaining elements
    while (i < t1.size) : (i += 1) {
        out_sum[i] = t1.data[i] + t2.data[i];
    }

    // Create output tensor
    const out_tensor = try Tensor(outputType).fromArray(t1.allocator, out_sum, t1.shape);

    // Remove the defer since the tensor will manage its own memory after creation
    return out_tensor;
}

// DOT PRODUCT -----------------------------------------------------------------------------------------------------------------------

/// Returns the dot product of two tensors. The dot product is the sum of the products of the corresponding entries of the two sequences of numbers.
/// Deprecated: use dot_product_tensor instead
pub fn compute_dot_product(comptime T: type, allocator: *const std.mem.Allocator, input: *Tensor(T), weights: *Tensor(T)) !Tensor(T) {
    return try CPU_dot_product_tensors(T, T, allocator, input, weights);
}

/// Returns the dot product of two tensors. The dot product is the sum of the products of the corresponding entries of the two sequences of numbers.
pub fn dot_product_tensor(comptime arch: Architectures, comptime Tin: anytype, comptime Tout: anytype, allocator: *const std.mem.Allocator, t1: *Tensor(Tin), t2: *Tensor(Tin)) !Tensor(Tout) {
    return switch (arch) {
        Architectures.CPU => return CPU_dot_product_tensors(Tin, Tout, allocator, t1, t2),
        Architectures.GPU => {
            std.log.debug("{} is under development\n", .{arch});
            return ArchitectureError.UnderDevelopmentArchitecture;
        },
        Architectures.SP32 => {
            std.log.debug("{} is under development\n", .{arch});
            return ArchitectureError.UnderDevelopmentArchitecture;
        },
        else => return ArchitectureError.UnknownArchitecture,
    };
}
/// Implementation of dot product for CPU architecture still not parallelized
pub fn CPU_dot_product_tensors(comptime inputType: anytype, comptime outputType: anytype, allocator: *const std.mem.Allocator, t1: *Tensor(inputType), t2: *Tensor(inputType)) !Tensor(outputType) {

    //CHECKS :
    const nDimT1 = t1.shape.len; //number of dimesion of tensor 1
    const nDimT2 = t2.shape.len; //number of dimesion of tensor 2
    // -imput shape:
    if (nDimT1 != nDimT2) return TensorMathError.InputTensorDifferentShape;

    //-dimensional compatibility:
    // If you have two matrices A and B, to compute the product A×B, the number of columns in A must be equal to the number of rows in B.
    // If A is a matrix of dimensions m×n and B is a matrix of dimensions n×p, then the product A×B is defined, and it results in a matrix of dimensions m×p.
    if (t1.shape[nDimT1 - 1] != t2.shape[nDimT1 - 2]) return TensorMathError.InputTensorsWrongShape;

    // -this check is necassary to avoid loss of information/ overflow when working with quantized tensors
    // usually quantization reduce to a maximum of 16bit, to the next check is divided between quant and non-quant data
    //bool (1 bit)
    // u1 (1 bit)
    // i8 (8 bits)
    // u8 (8 bits)
    // i16 (16 bits)
    // u16 (16 bits)
    // f16 (16 bits)
    // i32 (32 bits)
    // u32 (32 bits)
    // f32 (32 bits)
    // i64 (64 bits)
    // u64 (64 bits)
    // f64 (64 bits)
    // i128 (128 bits)
    // u128 (128 bits)
    // f128 (128 bits)
    if (@TypeOf(outputType) == @TypeOf(inputType)) {
        // Se input e output sono dello stesso tipo, non eseguire il controllo
        // Evitiamo l'errore in questo caso
    } else {
        if (@bitSizeOf(outputType) <= 16) { //quantized
            if (@bitSizeOf(outputType) <= (@bitSizeOf(inputType) * 2)) return TensorMathError.TooSmallOutputType;
        } else { //non-quant
            if (@bitSizeOf(outputType) <= @bitSizeOf(inputType)) return TensorMathError.TooSmallOutputType;
        }
    }

    //CREATING output_tensor :
    var out_shape = try allocator.alloc(usize, nDimT1); //I had to use alloc() bacause nDimT1 is not known at comptime
    defer allocator.free(out_shape);
    //defining the resulting shape
    for (0..(nDimT1 - 2)) |i| {
        out_shape[i] = t1.shape[i];
    }
    out_shape[nDimT1 - 2] = t1.shape[nDimT1 - 2];
    out_shape[nDimT1 - 1] = t2.shape[nDimT1 - 1];

    var out_tensor = try Tensor(outputType).fromShape(allocator, out_shape);
    try out_tensor.set(0, 0);
    //initialize the current location to all 0
    const location = try allocator.alloc(usize, nDimT1);
    defer allocator.free(location);
    for (location) |*loc| {
        loc.* = 0;
    }

    //call mutidim_mat_mul to handle multidimensionality
    try multidim_multiplication(
        inputType,
        outputType,
        t1,
        t2,
        &out_tensor,
        0,
        location,
    );
    //print output tensor shape

    return out_tensor;
}
/// Function that performs the multiplication of two tensors used in a recursive way to handle multidimensional tensors
fn multidim_multiplication(comptime inputType: anytype, comptime outputType: anytype, t1: *Tensor(inputType), t2: *Tensor(inputType), t3: *Tensor(outputType), current_depth: usize, location: []usize) !void {
    if (current_depth == (t1.shape.len - 2)) {

        //declaring sum
        var sum: outputType = 0;

        //with the first two for loop I iterate over t3
        for (0..t1.shape[current_depth]) |row| { //for each row of t1

            for (0..t2.shape[current_depth + 1]) |col| { //for each col of t2

                sum = 0;

                for (0..t1.shape[current_depth + 1]) |i| {

                    //compose the location on t1
                    location[t1.shape.len - 1] = i; //location
                    location[t1.shape.len - 2] = row; //location

                    //getting the correct numbers in t1
                    const a = try t1.get_at(location);

                    //compose the location on t2
                    location[t1.shape.len - 1] = col; //location
                    location[t1.shape.len - 2] = i; //location

                    //getting the correct numbers in t2
                    const b = try t2.get_at(location);

                    sum += a * b;
                }

                //compose the location on t3
                location[t1.shape.len - 1] = col; //col on the out tensor matrix
                location[t1.shape.len - 2] = row; //row on the out tensor matrix

                try t3.set_at(location, sum);
            }
        }
    } else {
        for (0..t1.shape[current_depth]) |element_at_current_depth| {
            //print location:
            //std.log.debug("\n depth: {} element_at_current_depth: {}", .{ current_depth, element_at_current_depth });
            location[current_depth] = element_at_current_depth;
            //otherwise I have to go deeper
            try multidim_multiplication(
                inputType,
                outputType,
                t1,
                t2,
                t3,
                current_depth + 1,
                location,
            );
        }
    }
}

// CONVOLVE -----------------------------------------------------------------------------------------------------------------------

pub fn convolve_tensor(comptime arch: Architectures, comptime Tin: anytype, comptime Tout: anytype, allocator: *std.mem.Allocator, input: *Tensor(Tin), kernel: *Tensor(Tin)) !Tensor(Tout) {
    return switch (arch) {
        Architectures.CPU => return CPU_convolve_tensors(Tin, Tout, allocator, input, kernel),
        Architectures.GPU => {
            std.log.debug("{} è in fase di sviluppo\n", .{arch});
            return ArchitectureError.UnderDevelopmentArchitecture;
        },
        Architectures.SP32 => {
            std.log.debug("{} è in fase di sviluppo\n", .{arch});
            return ArchitectureError.UnderDevelopmentArchitecture;
        },
        else => return ArchitectureError.UnknownArchitecture,
    };
}

/// Convolution fro CPU Architecture
pub fn CPU_convolve_tensors(comptime inputType: anytype, comptime outputType: anytype, allocator: *std.mem.Allocator, input: *Tensor(inputType), kernel: *Tensor(inputType)) !Tensor(outputType) {
    // CHECKS:
    const nDimInput = input.shape.len; // Dimension of the input tensor
    const nDimKernel = kernel.shape.len; // Dimension of the kernel tensor

    // Verify that the input and kernel tensors have the same number of dimensions
    if (nDimInput != nDimKernel) return TensorMathError.InputTensorDifferentShape;

    // Verify that the kernel is smaller than the input tensor in all dimensions
    for (0..nDimInput) |i| {
        if (kernel.shape[i] > input.shape[i]) return TensorMathError.InputTensorsWrongShape;
    }

    // Check that the output tensor is large enough to hold the result
    if (@TypeOf(outputType) == @TypeOf(inputType)) {} else {
        if (@bitSizeOf(outputType) <= 16) {
            if (@bitSizeOf(outputType) <= (@bitSizeOf(inputType) * 2)) return TensorMathError.TooSmallOutputType;
        } else { // Non quantizzato
            if (@bitSizeOf(outputType) <= @bitSizeOf(inputType)) return TensorMathError.TooSmallOutputType;
        }
    }

    // Creation of the output tensor
    var out_shape = try allocator.alloc(usize, nDimInput);
    defer allocator.free(out_shape);

    for (0..nDimInput) |i| {
        out_shape[i] = input.shape[i] - kernel.shape[i] + 1;
    }

    var out_tensor = try Tensor(outputType).fromShape(&allocator, out_shape);
    try out_tensor.set(0, 0);

    const location = try allocator.alloc(usize, nDimInput);
    defer allocator.free(location);
    for (location) |*loc| {
        loc.* = 0;
    }

    // Multi-dimensional convolution
    try multidim_convolution(
        inputType,
        outputType,
        allocator,
        input,
        kernel,
        &out_tensor,
        0,
        location,
    );

    return out_tensor;
}

/// Function that performs the convolution of two tensors used in a recursive way to handle multidimensional tensors
fn multidim_convolution(comptime inputType: anytype, comptime outputType: anytype, allocator: *std.mem.Allocator, input: *Tensor(inputType), kernel: *Tensor(inputType), output: *Tensor(outputType), current_dim: usize, location: []usize) !void {
    if (current_dim == input.shape.len) {
        // Base Case: calculate in this location

        var sum: outputType = 0;
        const nDims = input.shape.len;

        const kernel_indices = try allocator.alloc(usize, nDims);
        const input_indices = try allocator.alloc(usize, nDims);

        // SUm over the kernel
        try sum_over_kernel(
            inputType,
            outputType,
            input,
            kernel,
            &sum,
            location,
            kernel_indices,
            input_indices,
            0,
        );

        try output.set_at(location, sum);

        allocator.free(kernel_indices);
        allocator.free(input_indices);
    } else {
        for (0..output.shape[current_dim]) |i| {
            location[current_dim] = i;
            try multidim_convolution(
                inputType,
                outputType,
                allocator,
                input,
                kernel,
                output,
                current_dim + 1,
                location,
            );
        }
    }
}

fn anyOutOfBounds(indices: []usize, shape: []usize) bool {
    for (0..indices.len) |i| {
        if (indices[i] >= shape[i]) {
            return true;
        }
    }
    return false;
}

/// Recursive function to sum over the kernel
fn sum_over_kernel(
    comptime inputType: anytype,
    comptime outputType: anytype,
    input: *Tensor(inputType),
    kernel: *Tensor(inputType),
    sum: *outputType,
    input_location: []usize,
    kernel_indices: []usize,
    input_indices: []usize,
    current_dim: usize,
) !void {
    if (current_dim == kernel.shape.len) {
        if (anyOutOfBounds(input_indices, input.shape)) {
            return error.IndexOutOfBounds;
        }
        if (anyOutOfBounds(kernel_indices, kernel.shape)) {
            return error.IndexOutOfBounds;
        }

        const input_value = try input.get_at(input_indices);
        const kernel_value = try kernel.get_at(kernel_indices);

        sum.* += input_value * kernel_value;
    } else {
        for (0..kernel.shape[current_dim]) |k| {
            kernel_indices[current_dim] = k;

            const input_idx_dim = if (current_dim >= 2) current_dim else current_dim + 2;
            if (input_idx_dim < input_indices.len) {
                input_indices[input_idx_dim] = input_location[input_idx_dim] + k;
            }

            try sum_over_kernel(
                inputType,
                outputType,
                input,
                kernel,
                sum,
                input_location,
                kernel_indices,
                input_indices,
                current_dim + 1,
            );
        }
    }
}

/// Multidim Conv
fn multidim_convolution_with_bias(
    comptime inputType: anytype,
    comptime outputType: anytype,
    allocator: *const std.mem.Allocator,
    input: *Tensor(inputType),
    kernel: *Tensor(inputType),
    output: *Tensor(outputType),
    bias: *Tensor(outputType),
    current_dim: usize,
    location: []usize,
) !void {
    if (current_dim == output.shape.len) {
        var sum: outputType = 0;

        const kernel_indices = try allocator.alloc(usize, kernel.shape.len);
        defer allocator.free(kernel_indices);
        const input_indices = try allocator.alloc(usize, input.shape.len);
        defer allocator.free(input_indices);

        // Inizializza gli indici
        for (0..kernel_indices.len) |i| kernel_indices[i] = 0;
        for (0..input_indices.len) |i| input_indices[i] = 0;

        // Imposta il batch
        input_indices[0] = location[0]; // Batch

        //std.log.debug("multidim_convolution_with_bias: location: {d}, sum: {}\n", .{ location, sum });

        // Itera su tutti i canali
        for (0..kernel.shape[1]) |channel| {
            input_indices[1] = channel; // Canale di input
            kernel_indices[1] = channel; // Canale del kernel

            try sum_over_kernel(
                inputType,
                outputType,
                input,
                kernel,
                &sum,
                location,
                kernel_indices,
                input_indices,
                2, // Partendo dalle dimensioni spaziali
            );
        }

        // Recupera il bias dal tensore bias
        const bias_index = [_]usize{ location[1], 0 }; // Location[1] corrisponde al filtro
        //bias.info();
        const bias_value = try bias.get_at(&bias_index);

        sum += bias_value; // Aggiungi il bias
        //std.log.debug("multidim_convolution_with_bias: Adding bias. New sum: {}\n", .{sum});

        // Imposta il risultato nell'output
        try output.set_at(location, sum);
    } else {
        // Itera lungo la dimensione corrente dell'output
        for (0..output.shape[current_dim]) |i| {
            location[current_dim] = i;

            if (location[current_dim] >= output.shape[current_dim]) {
                std.log.debug("Error: location out of bounds: {d}, shape: {d}\n", .{ location, output.shape });
                return error.IndexOutOfBounds;
            }

            //std.log.debug("multidim_convolution_with_bias: Recursing at dimension {d}, index {d}\n", .{ current_dim, i });

            try multidim_convolution_with_bias(
                inputType,
                outputType,
                allocator,
                input,
                kernel,
                output,
                bias,
                current_dim + 1,
                location,
            );
        }
    }
}

/// Convolution tensor with bias
pub fn CPU_convolve_tensors_with_bias(
    comptime inputType: anytype,
    comptime outputType: anytype,
    allocator: *const std.mem.Allocator,
    input: *Tensor(inputType),
    kernel: *Tensor(inputType),
    bias: *Tensor(outputType),
) !Tensor(outputType) {
    //std.log.debug("CPU_convolve_tensors_with_bias: input shape: {d}, kernel shape: {d}\n", .{ input.shape, kernel.shape });

    const nDimInput = input.shape.len;
    const nDimKernel = kernel.shape.len;

    if (nDimInput != 4 or nDimKernel != 4) {
        std.log.debug("Error: Tensors must have 4 dimensions\n", .{});
        return TensorMathError.InputTensorDifferentShape;
    }

    if (input.shape[1] != kernel.shape[1]) {
        std.log.debug("Error: Mismatched channels. Input: {d}, Kernel: {d}\n", .{ input.shape[1], kernel.shape[1] });
        return TensorMathError.InputTensorsWrongShape;
    }

    for (2..4) |i| {
        if (kernel.shape[i] > input.shape[i]) {
            std.log.debug("Error: Kernel larger than input in dimension {d}\n", .{i});
            return TensorMathError.InputTensorsWrongShape;
        }
    }

    var out_shape: [4]usize = [_]usize{
        input.shape[0], // Batch
        kernel.shape[0], // n filters
        input.shape[2] - kernel.shape[2] + 1, // Height
        input.shape[3] - kernel.shape[3] + 1, // Width
    };

    //std.log.debug("Output tensor shape: {d}\n", .{out_shape});

    var out_tensor = try Tensor(outputType).fromShape(allocator, &out_shape);
    try out_tensor.set(0, 0);

    var location: [4]usize = [_]usize{ 0, 0, 0, 0 };

    try multidim_convolution_with_bias(
        inputType,
        outputType,
        allocator,
        input,
        kernel,
        &out_tensor,
        bias,
        0,
        &location,
    );

    //std.log.debug("Result tensor data: {d}\n", .{out_tensor.data});
    return out_tensor;
}

pub fn convolution_backward_biases(comptime T: type, allocator: *const std.mem.Allocator, dValues: *Tensor(T)) !Tensor(T) {
    // Compute gradients with respect to biases by summing over batch, height, and width dimensions
    // Assumes dValues shape: [batch_size, out_channels, output_height, output_width]

    // Check that dValues has at least 4 dimensions
    if (dValues.shape.len < 4) return TensorMathError.InputTensorsWrongShape;

    const out_channels = dValues.shape[1];
    var bias_gradients_shape = [_]usize{out_channels};

    // Allocate the bias_gradients tensor
    var bias_gradients = try Tensor(T).fromShape(allocator, &bias_gradients_shape);

    // Initialize bias_gradients to zero
    try bias_gradients.set(0, 0);

    const batch_size = dValues.shape[0];
    const output_height = dValues.shape[2];
    const output_width = dValues.shape[3];

    // Sum over batch_size, output_height, output_width dimensions
    for (0..out_channels) |oc| {
        var sum: T = 0;
        for (0..batch_size) |b| {
            for (0..output_height) |h| {
                for (0..output_width) |w| {
                    const index = [_]usize{ b, oc, h, w };
                    const val = try dValues.get_at(&index);
                    sum += val;
                }
            }
        }
        // Set the sum in bias_gradients
        try bias_gradients.set_at(&[_]usize{oc}, sum);
    }

    return bias_gradients;
}

pub fn convolution_backward_weights(comptime T: type, allocator: *const std.mem.Allocator, input: *Tensor(T), dValues: *Tensor(T)) !Tensor(T) {
    // Compute gradients with respect to weights
    // Input shape: [batch_size, in_channels, input_height, input_width]
    // dValues shape: [batch_size, out_channels, output_height, output_width]
    // Weights shape: [out_channels, in_channels, kernel_height, kernel_width]

    const batch_size = input.shape[0];
    const in_channels = input.shape[1];
    const input_height = input.shape[2];
    const input_width = input.shape[3];

    const out_batch_size = dValues.shape[0];
    const out_channels = dValues.shape[1];
    const output_height = dValues.shape[2];
    const output_width = dValues.shape[3];

    std.log.debug("\n batch_size: {} in_channels: {} input_height: {} input_width: {} out_batch_size: {} out_channels: {} output_height: {} output_width: {}\n", .{ batch_size, in_channels, input_height, input_width, out_batch_size, out_channels, output_height, output_width });

    // Check for matching batch sizes
    if (batch_size != out_batch_size) return TensorMathError.InputTensorsWrongShape;

    // Calculate kernel dimensions
    const kernel_height = input_height - output_height + 1;
    const kernel_width = input_width - output_width + 1;

    var w_gradients_shape = [_]usize{ out_channels, in_channels, kernel_height, kernel_width };
    var w_gradients = try Tensor(T).fromShape(allocator, &w_gradients_shape);

    // Initialize w_gradients to zero
    try w_gradients.set(0, 0);

    // Compute gradients
    for (0..out_channels) |oc| {
        for (0..in_channels) |ic| {
            for (0..kernel_height) |kh| {
                for (0..kernel_width) |kw| {
                    var sum: T = 0;
                    for (0..batch_size) |b| {
                        for (0..output_height) |oh| {
                            for (0..output_width) |ow| {
                                const input_h = oh + kh;
                                const input_w = ow + kw;

                                const input_index = [_]usize{ b, ic, input_h, input_w };
                                const dValue_index = [_]usize{ b, oc, oh, ow };

                                const input_val = try input.get_at(&input_index);
                                const dValue = try dValues.get_at(&dValue_index);

                                sum += input_val * dValue;
                            }
                        }
                    }
                    // Set the gradient
                    const w_grad_index = [_]usize{ oc, ic, kh, kw };
                    try w_gradients.set_at(&w_grad_index, sum);
                }
            }
        }
    }

    return w_gradients;
}

pub fn convolution_backward_input(comptime T: type, allocator: *const std.mem.Allocator, dValues: *Tensor(T), weights: *Tensor(T)) !Tensor(T) {
    // Compute gradients with respect to the input
    // dValues shape: [batch_size, out_channels, output_height, output_width]
    // Weights shape: [out_channels, in_channels, kernel_height, kernel_width]
    // Output gradients shape: [batch_size, in_channels, input_height, input_width]

    const batch_size = dValues.shape[0];
    const out_channels = dValues.shape[1];
    const output_height = dValues.shape[2];
    const output_width = dValues.shape[3];

    const weight_out_channels = weights.shape[0];
    const in_channels = weights.shape[1];
    const kernel_height = weights.shape[2];
    const kernel_width = weights.shape[3];

    if (out_channels != weight_out_channels) {
        std.log.debug("Error: Mismatched output channels: dValues {d}, weights {d}\n", .{ out_channels, weight_out_channels });
        return TensorMathError.InputTensorsWrongShape;
    }

    const input_height = output_height + kernel_height - 1;
    const input_width = output_width + kernel_width - 1;

    var input_gradients_shape = [_]usize{ batch_size, in_channels, input_height, input_width };
    var input_gradients = try Tensor(T).fromShape(allocator, &input_gradients_shape);

    // Initialize input_gradients to zero
    try input_gradients.set(0, 0);

    std.log.debug("Backward input gradients initialized with shape: {d}\n", .{input_gradients.shape});

    // Compute input gradients
    for (0..batch_size) |b| {
        for (0..in_channels) |ic| {
            var shape: [4]usize = [_]usize{ 1, 1, input_height, input_width };
            var input_channel_gradient = try Tensor(T).fromShape(allocator, &shape);
            try input_channel_gradient.set(0, 0);

            for (0..out_channels) |oc| {
                //std.log.debug("Processing batch {d}, in_channel {d}, out_channel {d}\n", .{ b, ic, oc });

                // Flip weights along spatial dimensions (rotate 180 degrees)
                var flipped_weights = try flip_kernel(T, allocator, weights, oc, ic);

                // Slice dValues for the current batch and out_channel
                var start_indices = [_]usize{ b, oc, 0, 0 };
                var slice_shape = [_]usize{ 1, 1, output_height, output_width };

                //std.log.debug("Slicing dValues with start indices: {d}, slice shape: {d}\n", .{ start_indices, slice_shape });

                var dValue_slice = try dValues.slice(&start_indices, &slice_shape);

                //std.log.debug("Starting convolution for batch {d}, in_channel {d}, out_channel {d}\n", .{ b, ic, oc });

                //Create 0 array with bias, can be optimized
                const zeros = try Layer.zeros(T, allocator, out_channels, 1);
                defer allocator.free(zeros);
                var shapeBias = [_]usize{ out_channels, 1 };
                var zeroBias = try Tensor(T).fromArray(allocator, zeros, &shapeBias);

                // Convolve dValues[b, oc, :, :] with flipped_weights
                var input_grad = try CPU_convolve_tensors_with_bias(T, T, allocator, &dValue_slice, &flipped_weights, &zeroBias);

                // Add to input_channel_gradient
                for (0..input_grad.shape[2]) |h| {
                    for (0..input_grad.shape[3]) |w| {
                        const input_grad_val = try input_grad.get_at(&[_]usize{ 0, 0, h, w });
                        const index = [_]usize{ 0, 0, h, w };
                        const current_val = try input_channel_gradient.get_at(&index);
                        try input_channel_gradient.set_at(&index, current_val + input_grad_val);
                    }
                }

                //std.log.debug("Completed convolution for batch {d}, in_channel {d}, out_channel {d}\n", .{ b, ic, oc });

                // Clean up temporary tensors
                input_grad.deinit();
                flipped_weights.deinit();
                dValue_slice.deinit();
                zeroBias.deinit();
            }

            // Add input_channel_gradient to input_gradients
            for (0..input_height) |h| {
                for (0..input_width) |w| {
                    const grad_val = try input_channel_gradient.get_at(&[_]usize{ 0, 0, h, w });
                    const index = [_]usize{ b, ic, h, w };
                    const current_val = try input_gradients.get_at(&index);
                    try input_gradients.set_at(&index, current_val + grad_val);
                }
            }

            input_channel_gradient.deinit();
        }
    }

    //std.log.debug("Completed backward input gradients computation. Shape: {d}\n", .{input_gradients.shape});

    return input_gradients;
}

// Helper function to flip the kernel (rotate 180 degrees)
fn flip_kernel(comptime T: type, allocator: *const std.mem.Allocator, weights: *Tensor(T), out_channel: usize, in_channel: usize) !Tensor(T) {
    const kernel_height = weights.shape[2];
    const kernel_width = weights.shape[3];
    var flipped_shape = [_]usize{ 1, 1, kernel_height, kernel_width };

    var flipped_kernel = try Tensor(T).fromShape(allocator, &flipped_shape);

    for (0..kernel_height) |h| {
        for (0..kernel_width) |w| {
            const flipped_index = [_]usize{ 0, 0, kernel_height - h - 1, kernel_width - w - 1 };
            const original_index = [_]usize{ out_channel, in_channel, h, w };

            const value = try weights.get_at(&original_index);
            try flipped_kernel.set_at(&flipped_index, value);
        }
    }

    //std.log.debug("Flipped kernel for out_channel {d}, in_channel {d} generated.\n", .{ out_channel, in_channel });

    return flipped_kernel;
}

// POOLING -----------------------------------------------------------------------------------------------------------------------
//TODO: add padding
pub fn pool_tensor(
    comptime T: type,
    allocator: *const std.mem.Allocator,
    input: *Tensor(T),
    used_windows: *Tensor(u8), // Shape: [W, input_rows, input_cols]
    kernel: []usize,
    stride: []usize,
    poolingType: PoolingType,
) !Tensor(T) {
    // Calcolo output shape
    var outputTensorShape = try allocator.alloc(usize, input.shape.len);
    defer allocator.free(outputTensorShape);
    for (0..input.shape.len - 2) |i| {
        outputTensorShape[i] = input.shape[i];
    }
    const height = input.shape.len - 2;
    const width = input.shape.len - 1;

    outputTensorShape[height] = (input.shape[height] - kernel[0] + 1) / stride[0];
    outputTensorShape[width] = (input.shape[width] - kernel[1] + 1) / stride[1];

    var output = try Tensor(T).fromShape(allocator, outputTensorShape);

    //const output_rows = outputTensorShape[height];
    //const output_cols = outputTensorShape[width];

    // Il tensor used_windows ha shape [W, input_rows, input_cols]
    // W = output_rows * output_cols
    // Assumiamo che used_windows sia già stato allocato e zero-inizializzato prima di chiamare questa funzione.

    // location
    const location = try allocator.alloc(usize, input.shape.len);
    defer allocator.free(location);
    for (location) |*loc| loc.* = 0;

    try multidim_pooling(
        T,
        allocator,
        input,
        used_windows,
        &output,
        0,
        location,
        kernel,
        stride,
        poolingType,
    );

    return output;
}

pub fn multidim_pooling(
    comptime T: anytype,
    allocator: *const std.mem.Allocator,
    input: *Tensor(T),
    used_windows: *Tensor(u8), // Shape: [W, input_rows, input_cols]
    output: *Tensor(T),
    current_depth: usize,
    location: []usize,
    kernel: []usize,
    stride: []usize,
    poolingType: PoolingType,
) !void {
    if (current_depth == output.shape.len - 2) {
        var temp_location = try allocator.alloc(usize, input.shape.len);
        defer allocator.free(temp_location);
        var window_location = try allocator.alloc(usize, input.shape.len);
        defer allocator.free(window_location);
        var window_values = try allocator.alloc(T, kernel[0] * kernel[1]);
        defer allocator.free(window_values);

        var output_row_counter: usize = 0;
        var output_col_counter: usize = 0;

        @memcpy(temp_location, location);
        @memcpy(window_location, location);

        temp_location[current_depth] = 0;
        temp_location[current_depth + 1] = 0;

        const input_rows = input.shape[current_depth];
        const input_cols = input.shape[current_depth + 1];

        const output_cols = output.shape[current_depth + 1];

        while (temp_location[current_depth] + kernel[0] <= input.shape[current_depth]) : (temp_location[current_depth] += stride[0]) {
            while (temp_location[current_depth + 1] + kernel[1] <= input.shape[current_depth + 1]) : (temp_location[current_depth + 1] += stride[1]) {
                window_location[current_depth] = temp_location[current_depth];
                window_location[current_depth + 1] = temp_location[current_depth + 1];

                const kernel_rows = kernel[0];
                const kernel_cols = kernel[1];

                const w = output_row_counter * output_cols + output_col_counter;

                for (0..kernel_rows) |i| {
                    window_location[current_depth] += i;
                    for (0..kernel_cols) |j| {
                        window_location[current_depth + 1] += j;
                        window_values[i * kernel_cols + j] = try input.get_at(window_location);
                    }
                    window_location[current_depth + 1] = temp_location[current_depth + 1];
                }

                switch (poolingType) {
                    .Max => {
                        var max = window_values[0];
                        var max_idx: usize = 0;
                        for (0..window_values.len) |i| {
                            if (window_values[i] > max) {
                                max = window_values[i];
                                max_idx = i;
                            }
                        }

                        const row = max_idx / kernel_cols;
                        const col = max_idx % kernel_cols;

                        const max_r = temp_location[current_depth] + row;
                        const max_c = temp_location[current_depth + 1] + col;

                        used_windows.data[w * (input_rows * input_cols) + max_r * input_cols + max_c] = 1;

                        // Set output
                        window_location[current_depth] = output_row_counter;
                        window_location[current_depth + 1] = output_col_counter;
                        try output.set_at(window_location, max);
                    },
                    .Min => {},
                    .Avg => {},
                }

                output_col_counter += 1;
            }
            temp_location[current_depth + 1] = 0;
            output_row_counter += 1;
            output_col_counter = 0;
        }
    } else {
        for (0..output.shape[current_depth]) |element_at_current_depth| {
            location[current_depth] = element_at_current_depth;
            try multidim_pooling(
                T,
                allocator,
                input,
                used_windows,
                output,
                current_depth + 1,
                location,
                kernel,
                stride,
                poolingType,
            );
        }
    }
}
