const std = @import("std");
const Tensor = @import("tensor").Tensor;
const TensMath = @import("tensor_m");
const Architectures = @import("architectures").Architectures;
const TensorMathError = @import("errorHandler").TensorMathError;
const ArchitectureError = @import("errorHandler").ArchitectureError;
const ErrorHandler = @import("errorHandler");
const PoolingType = @import("poolingLayer").PoolingType;
const pkgAllocator = @import("pkgAllocator");

test "tests description" {
    std.debug.print("\n--- Running tensor_math tests\n", .{});
}

test "Sum two tensors on CPU architecture" {
    std.debug.print("\n     test: Sum two tensors on CPU architecture", .{});
    const allocator = pkgAllocator.allocator;

    var inputArray: [2][2]f32 = [_][2]f32{
        [_]f32{ 1.0, 2.0 },
        [_]f32{ 4.0, 5.0 },
    };

    var shape: [2]usize = [_]usize{ 2, 2 }; // 2x2 matrix

    var t1 = try Tensor(f32).fromArray(&allocator, &inputArray, &shape);
    defer t1.deinit();
    var t2 = try Tensor(f32).fromArray(&allocator, &inputArray, &shape);
    defer t2.deinit();

    var t3 = try TensMath.sum_tensors(Architectures.CPU, f32, f64, &t1, &t2); // Output tensor with larger type
    defer t3.deinit();

    // Check if the values in t3 are as expected
    try std.testing.expect(2.0 == t3.data[0]);
    try std.testing.expect(4.0 == t3.data[1]);
}

test "Error when input tensors have different sizes" {
    std.debug.print("\n     test: Error when input tensors have different sizes", .{});
    const allocator = pkgAllocator.allocator;

    var inputArray: [2][2]f32 = [_][2]f32{
        [_]f32{ 1.0, 2.0 },
        [_]f32{ 4.0, 5.0 },
    };
    var inputArray2: [3][2]f32 = [_][2]f32{
        [_]f32{ 1.0, 2.0 },
        [_]f32{ 4.0, 5.0 },
        [_]f32{ 14.0, 15.0 },
    };

    var shape1 = [_]usize{ 2, 2 }; // 2x2 matrix
    var shape2 = [_]usize{ 3, 2 }; // 3x2 matrix
    var t1 = try Tensor(f32).fromArray(&allocator, &inputArray, &shape1);
    var t2 = try Tensor(f32).fromArray(&allocator, &inputArray2, &shape2);

    try std.testing.expectError(TensorMathError.InputTensorDifferentSize, TensMath.sum_tensors(Architectures.CPU, f32, f64, &t1, &t2));

    t1.deinit();
    t2.deinit();
}

test "Dot product 2x2" {
    std.debug.print("\n     test:Dot product 2x2", .{});

    const allocator = pkgAllocator.allocator;

    var shape: [2]usize = [_]usize{ 2, 2 }; // 2x2 matrix

    var inputArray: [2][2]f32 = [_][2]f32{
        [_]f32{ 1.0, 2.0 },
        [_]f32{ 4.0, 5.0 },
    };

    var t1 = try Tensor(f32).fromArray(&allocator, &inputArray, &shape);
    var t2 = try Tensor(f32).fromArray(&allocator, &inputArray, &shape);

    var result_tensor = try TensMath.dot_product_tensor(Architectures.CPU, f32, f64, &allocator, &t1, &t2);

    try std.testing.expect(9.0 == result_tensor.data[0]);
    try std.testing.expect(12.0 == result_tensor.data[1]);

    result_tensor.deinit();
    t1.deinit();
    t2.deinit();
}

test "Error when input tensors have incompatible sizes for dot product" {
    const allocator = pkgAllocator.allocator;

    var shape1: [2]usize = [_]usize{ 2, 2 }; // 2x2 matrix
    var shape2: [2]usize = [_]usize{ 3, 2 }; // 3x2 matrix
    var t1 = try Tensor(f32).fromShape(&allocator, &shape1);
    var t2 = try Tensor(f32).fromShape(&allocator, &shape2);

    try std.testing.expectError(TensorMathError.InputTensorsWrongShape, TensMath.dot_product_tensor(Architectures.CPU, f32, f64, &allocator, &t1, &t2));

    _ = TensMath.dot_product_tensor(Architectures.CPU, f32, f64, &allocator, &t1, &t2) catch |err| {
        std.debug.print("\n _______ {s} ______", .{ErrorHandler.errorDetails(err)});
    };
    t1.deinit();
    t2.deinit();
}

test "Error when input tensors have incompatible shapes for dot product" {
    std.debug.print("\n     test: Error when input tensors have incompatible shapes for dot product", .{});
    const allocator = pkgAllocator.allocator;

    var shape1: [2]usize = [_]usize{ 2, 2 }; // 2x2 matrix
    var shape2: [2]usize = [_]usize{ 4, 1 }; // 4x1 matrix
    var t1 = try Tensor(f32).fromShape(&allocator, &shape1);
    var t2 = try Tensor(f32).fromShape(&allocator, &shape2);

    try std.testing.expectError(TensorMathError.InputTensorsWrongShape, TensMath.dot_product_tensor(Architectures.CPU, f32, f64, &allocator, &t1, &t2));

    t1.deinit();
    t2.deinit();
}

test "GPU architecture under development error" {
    std.debug.print("\n     test: GPU architecture under development error\n", .{});
    const allocator = pkgAllocator.allocator;

    var shape: [2]usize = [_]usize{ 2, 2 }; // 2x2 matrix
    var t1 = try Tensor(f32).fromShape(&allocator, &shape);
    var t2 = try Tensor(f32).fromShape(&allocator, &shape);
    var t3 = try Tensor(f64).fromShape(&allocator, &shape);

    try std.testing.expectError(ArchitectureError.UnderDevelopementArchitecture, TensMath.sum_tensors(Architectures.GPU, f32, f64, &t1, &t2));

    t1.deinit();
    t2.deinit();
    t3.deinit();
}

test "add bias" {
    std.debug.print("\n     test:add bias", .{});
    const allocator = pkgAllocator.allocator;

    var shape_tensor: [2]usize = [_]usize{ 2, 3 }; // 2x3 matrix
    var inputArray: [2][3]f32 = [_][3]f32{
        [_]f32{ 1.0, 2.0, 3.0 },
        [_]f32{ 4.0, 5.0, 6.0 },
    };
    const flatArr: [6]f32 = [_]f32{ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0 };

    var shape_bias: [1]usize = [_]usize{3};
    var bias_array: [3]f32 = [_]f32{ 1.0, 1.0, 1.0 };

    var t1 = try Tensor(f32).fromArray(&allocator, &inputArray, &shape_tensor);
    var bias = try Tensor(f32).fromArray(&allocator, &bias_array, &shape_bias);

    try TensMath.add_bias(f32, &allocator, &t1, &bias);

    for (t1.data, 0..) |*data, i| {
        try std.testing.expect(data.* == flatArr[i] + 1);
    }

    t1.deinit();
    bias.deinit();
}

test "mean" {
    std.debug.print("\n     test:mean", .{});
    const allocator = pkgAllocator.allocator;

    var shape_tensor: [1]usize = [_]usize{3}; // 2x3 matrix
    var inputArray: [3]f32 = [_]f32{ 1.0, 2.0, 3.0 };

    var t1 = try Tensor(f32).fromArray(&allocator, &inputArray, &shape_tensor);

    try std.testing.expect(2.0 == TensMath.mean(f32, &t1));

    t1.deinit();
}

test "Convolution 4D Input with 2x2 Kernel" {
    std.debug.print("\n     test: Convolution 4D Input with 2x2 Kernel\n", .{});

    const allocator = pkgAllocator.allocator;

    var input_shape: [4]usize = [_]usize{ 2, 2, 3, 3 };
    var kernel_shape: [4]usize = [_]usize{ 1, 2, 2, 2 };

    var inputArray: [2][2][3][3]f32 = [_][2][3][3]f32{
        //Firsr Batch
        [_][3][3]f32{
            // First Channel
            [_][3]f32{
                [_]f32{ 2.0, 2.0, 3.0 },
                [_]f32{ 4.0, 5.0, 6.0 },
                [_]f32{ 7.0, 8.0, 9.0 },
            },
            // Second Channel
            [_][3]f32{
                [_]f32{ 8.0, 8.0, 7.0 },
                [_]f32{ 6.0, 5.0, 4.0 },
                [_]f32{ 3.0, 2.0, 1.0 },
            },
        },
        // Second batch
        [_][3][3]f32{
            // First channel
            [_][3]f32{
                [_]f32{ 2.0, 3.0, 4.0 },
                [_]f32{ 5.0, 6.0, 7.0 },
                [_]f32{ 8.0, 9.0, 10.0 },
            },
            // Second channel
            [_][3]f32{
                [_]f32{ 10.0, 9.0, 8.0 },
                [_]f32{ 7.0, 6.0, 5.0 },
                [_]f32{ 4.0, 3.0, 2.0 },
            },
        },
    };

    // Kernel tensor
    var kernelArray: [1][2][2][2]f32 = [_][2][2][2]f32{
        [_][2][2]f32{
            [_][2]f32{
                [_]f32{ -1.0, 0.0 },
                [_]f32{ 0.0, 1.0 },
            },
            [_][2]f32{
                [_]f32{ 1.0, -1.0 },
                [_]f32{ -1.0, 1.0 },
            },
        },
    };

    var inputbias: [2][1]f32 = [_][1]f32{
        [_]f32{1},
        [_]f32{
            1,
        },
    };
    var shape: [2]usize = [_]usize{ 2, 1 };
    var bias = try Tensor(f32).fromArray(&allocator, &inputbias, &shape);

    var input_tensor = try Tensor(f32).fromArray(&allocator, &inputArray, &input_shape);
    var kernel_tensor = try Tensor(f32).fromArray(&allocator, &kernelArray, &kernel_shape);

    var result_tensor = try TensMath.CPU_convolve_tensors_with_bias(f32, f32, &allocator, &input_tensor, &kernel_tensor, &bias);

    // Expected results with the correct dimensions
    const expected_result: [2][1][2][2]f32 = [_][1][2][2]f32{
        // Primo batch
        [_][2][2]f32{
            [_][2]f32{ [_]f32{ 3.0, 5.0 }, [_]f32{ 5.0, 5.0 } },
        },
        // Secondo batch
        [_][2][2]f32{
            [_][2]f32{ [_]f32{ 5.0, 5.0 }, [_]f32{ 5.0, 5.0 } },
        },
    };
    result_tensor.info();

    for (0..2) |batch| {
        for (0..1) |filter| {
            for (0..2) |i| {
                for (0..2) |j| {
                    const idx = batch * (1 * 2 * 2) + filter * (2 * 2) + i * 2 + j;
                    try std.testing.expectEqual(expected_result[batch][filter][i][j], result_tensor.data[idx]);
                }
            }
        }
    }

    result_tensor.deinit();
    input_tensor.deinit();
    kernel_tensor.deinit();
    bias.deinit();
}

test "Pooling 2D" {
    std.debug.print("\n     test: Pooling 2D\n", .{});

    const allocator = pkgAllocator.allocator;

    // 3x3 input, same as original
    var shape_tensor: [2]usize = [_]usize{ 3, 3 };
    var inputArray: [3][3]f32 = [_][3]f32{
        [_]f32{ 1.0, 2.0, 3.0 },
        [_]f32{ 4.0, 5.0, 6.0 },
        [_]f32{ 40.0, 50.0, 60.0 },
    };

    var input1 = try Tensor(f32).fromArray(&allocator, &inputArray, &shape_tensor);
    defer input1.deinit();
    var kernel1: [2]usize = [2]usize{ 2, 2 };
    var stride1: [2]usize = [2]usize{ 1, 1 };

    // Calculate W = number of windows
    const input_rows = input1.shape[0];
    const input_cols = input1.shape[1];
    const out_rows = (input_rows - kernel1[0] + 1) / stride1[0]; // 2
    const out_cols = (input_cols - kernel1[1] + 1) / stride1[1]; // 2
    const W = out_rows * out_cols; // 4

    // Instead of used_input being shape of input, now [W,3,3]
    var used_input1_shape = [_]usize{ W, input_rows, input_cols };
    var used_input1 = try Tensor(u8).fromShape(&allocator, used_input1_shape[0..]);
    defer used_input1.deinit();
    for (used_input1.data) |*val| val.* = 0;

    var output1: Tensor(f32) = try TensMath.pool_tensor(f32, &allocator, &input1, &used_input1, &kernel1, &stride1, PoolingType.Max);
    defer output1.deinit();

    // Same checks for output as original
    try std.testing.expectEqual(output1.shape.len, input1.shape.len);
    try std.testing.expectEqual(output1.shape[0], 2);
    try std.testing.expectEqual(output1.shape[1], 2);

    try std.testing.expectEqual(output1.data[0], 5);
    try std.testing.expectEqual(output1.data[1], 6);
    try std.testing.expectEqual(output1.data[2], 50);
    try std.testing.expectEqual(output1.data[3], 60);

    // Now we have 4 windows (W=4), each 3x3
    // The original code expected certain pattern in used_input:
    // (1,1)=5 for first window
    // (1,2)=6 for second window
    // (2,1)=50 for third window
    // (2,2)=60 for fourth window
    //
    // Windows:
    // w=0: top-left max at (1,1)
    // w=1: top-right max at (1,2)
    // w=2: bottom-left max at (2,1)
    // w=3: bottom-right max at (2,2)

    // Check w=0 window:
    try std.testing.expectEqual(@as(u1, 1), used_input1.data[0 * (9) + 1 * 3 + 1]); // (1,1)
    // Others in w=0 are 0
    try std.testing.expectEqual(@as(u1, 0), used_input1.data[0 * 9 + 0]);
    try std.testing.expectEqual(@as(u1, 0), used_input1.data[0 * 9 + 1]);
    try std.testing.expectEqual(@as(u1, 0), used_input1.data[0 * 9 + 2]);
    try std.testing.expectEqual(@as(u1, 0), used_input1.data[0 * 9 + 3]);
    try std.testing.expectEqual(@as(u1, 1), used_input1.data[0 * 9 + 4]); // Wait, originally this was 1 at position (1,1).
    // Actually, the original test had multiple ones. Now we only have one max per window.
    // To stay consistent with the original pattern, we would need to set multiple maxima, but that's not correct for max pooling.
    // Since now we have distinct windows, each window sets exactly one 1.
    // Let's just keep the single maximum logic. The original test was expecting multiple '1's because it didn't handle windows properly.
    // We now have one '1' per window. So for the top-left window, only (1,1) is 1.
    // Thus, we won't check the old pattern of multiple 1s. We'll only confirm the single '1' at the max location.

    // For simplicity, just confirm the correct single '1' in each window:
    // w=0: (1,1)=5
    try std.testing.expectEqual(@as(u1, 1), used_input1.data[0 * 9 + 1 * 3 + 1]);

    // w=1: top-right (1,2)
    try std.testing.expectEqual(@as(u1, 1), used_input1.data[1 * 9 + 1 * 3 + 2]);

    // w=2: bottom-left (2,1)
    try std.testing.expectEqual(@as(u1, 1), used_input1.data[2 * 9 + 2 * 3 + 1]);

    // w=3: bottom-right (2,2)
    try std.testing.expectEqual(@as(u1, 1), used_input1.data[3 * 9 + 2 * 3 + 2]);

    // Another test with kernel2 and stride2
    var kernel2: [2]usize = [2]usize{ 2, 2 };
    var stride2: [2]usize = [2]usize{ 2, 2 };

    const out_rows_2 = (3 - 2 + 1) / 2;
    const out_cols_2 = (3 - 2 + 1) / 2;
    const W2 = out_rows_2 * out_cols_2;

    var used_input2_shape = [_]usize{ W2, 3, 3 };
    var used_input2 = try Tensor(u8).fromShape(&allocator, used_input2_shape[0..]);
    defer used_input2.deinit();
    for (used_input2.data) |*val| val.* = 0;

    var output2: Tensor(f32) = try TensMath.pool_tensor(f32, &allocator, &input1, &used_input2, &kernel2, &stride2, PoolingType.Max);
    defer output2.deinit();

    try std.testing.expectEqual(output2.shape.len, input1.shape.len);
    try std.testing.expectEqual(output2.shape[0], 1);
    try std.testing.expectEqual(output2.shape[1], 1);
    try std.testing.expectEqual(output2.data[0], 5);

    // single window w=0 max at (1,1)
    try std.testing.expectEqual(@as(u1, 1), used_input2.data[0 * (3 * 3) + 1 * 3 + 1]);
}

test "Pooling multidim" {
    std.debug.print("\n     test: Pooling multidim\n", .{});

    const allocator = pkgAllocator.allocator;

    // 3x3x3 input as original
    var shape_tensor: [3]usize = [_]usize{ 3, 3, 3 };
    var inputArray: [3][3][3]f32 = [_][3][3]f32{
        [_][3]f32{
            [_]f32{ 1.0, 2.0, 3.0 },
            [_]f32{ 4.0, 5.0, 6.0 },
            [_]f32{ 40.0, 50.0, 60.0 },
        },
        [_][3]f32{
            [_]f32{ 10.0, 20.0, 30.0 },
            [_]f32{ 40.0, 0.0, -10.0 },
            [_]f32{ 40.0, 50.0, 60.0 },
        },
        [_][3]f32{
            [_]f32{ -1.0, -2.0, -3.0 },
            [_]f32{ -4.0, -5.0, -6.0 },
            [_]f32{ -40.0, -50.0, -60.0 },
        },
    };

    var input1 = try Tensor(f32).fromArray(&allocator, &inputArray, &shape_tensor);
    defer input1.deinit();
    var kernel1: [2]usize = [2]usize{ 2, 2 };
    var stride1: [2]usize = [2]usize{ 1, 1 };

    const d = input1.shape[0]; // 3
    const input_rows = input1.shape[1]; //3
    const input_cols = input1.shape[2]; //3
    const out_rows = (input_rows - kernel1[0] + 1) / stride1[0]; //2
    const out_cols = (input_cols - kernel1[1] + 1) / stride1[1]; //2
    const W_tot = d * out_rows * out_cols; // 3*2*2=12

    var used_input1_shape = [_]usize{ W_tot, input_rows, input_cols };
    var used_input1 = try Tensor(u8).fromShape(&allocator, used_input1_shape[0..]);
    defer used_input1.deinit();
    for (used_input1.data) |*val| val.* = 0;

    var output: Tensor(f32) = try TensMath.pool_tensor(f32, &allocator, &input1, &used_input1, &kernel1, &stride1, PoolingType.Max);
    defer output.deinit();

    try std.testing.expectEqual(output.shape.len, input1.shape.len);
    try std.testing.expectEqual(output.shape[0], 3);
    try std.testing.expectEqual(output.shape[1], 2);
    try std.testing.expectEqual(output.shape[2], 2);

    // We do not check used_input details here, just shapes as in original.
}
