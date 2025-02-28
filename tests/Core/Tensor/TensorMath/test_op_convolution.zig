const std = @import("std");
const zant = @import("zant");
const pkgAllocator = zant.utils.allocator;
const TensMath = zant.core.tensor.math_standard;
const Tensor = zant.core.tensor.Tensor;
const TensorMathError = zant.utils.error_handler.TensorMathError;

test "Convolution 4D Input with 2x2x2x2 Kernel shape" {
    std.debug.print("\n     test: Convolution 4D Input with 2x2x2x2 Kernel shape\n", .{});

    const allocator = pkgAllocator.allocator;

    // Input tensor
    var input_shape: [4]usize = [_]usize{ 2, 2, 3, 3 };
    var inputArray: [2][2][3][3]f32 = [_][2][3][3]f32{ //batches:2, channels:2, rows:3, cols:3
        //First Batch
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
    var kernel_shape: [4]usize = [_]usize{ 2, 2, 2, 2 };
    var kernelArray: [2][2][2][2]f32 = [_][2][2][2]f32{ //filters:2, channels:2, rows:2, cols:2
        //first filter
        [_][2][2]f32{
            //first channel
            [_][2]f32{
                [_]f32{ -1.0, 0.0 },
                [_]f32{ 0.0, 1.0 },
            },
            //second channel
            [_][2]f32{
                [_]f32{ 1.0, -1.0 },
                [_]f32{ -1.0, 1.0 },
            },
        },
        //second filter
        [_][2][2]f32{
            //first channel
            [_][2]f32{
                [_]f32{ 0.0, 0.0 },
                [_]f32{ 0.0, 0.0 },
            },
            //second channel
            [_][2]f32{
                [_]f32{ 0.0, 0.0 },
                [_]f32{ 0.0, 0.0 },
            },
        },
    };

    var inputbias: [2]f32 = [_]f32{ 1, 1 }; //batches: 2, filters:2

    var bias_shape: [1]usize = [_]usize{2};
    var bias = try Tensor(f32).fromArray(&allocator, &inputbias, &bias_shape);
    defer bias.deinit();
    var input_tensor = try Tensor(f32).fromArray(&allocator, &inputArray, &input_shape);
    defer input_tensor.deinit();
    var kernel_tensor = try Tensor(f32).fromArray(&allocator, &kernelArray, &kernel_shape);
    defer kernel_tensor.deinit();
    const stride: [2]usize = [_]usize{ 1, 1 };

    var result_tensor = try TensMath.convolve_tensor_with_bias(f32, &input_tensor, &kernel_tensor, &bias, &stride, null);
    defer result_tensor.deinit();

    // Expected results with the correct dimensions
    const expected_result: [2][2][2][2]f32 = [_][2][2][2]f32{
        // Primo batch
        [_][2][2]f32{
            [_][2]f32{
                [_]f32{ 3.0, 5.0 },
                [_]f32{ 5.0, 5.0 },
            },
            [_][2]f32{
                [_]f32{ 1.0, 1.0 },
                [_]f32{ 1.0, 1.0 },
            },
        },
        // Secondo batch
        [_][2][2]f32{
            [_][2]f32{
                [_]f32{ 5.0, 5.0 },
                [_]f32{ 5.0, 5.0 },
            },
            [_][2]f32{
                [_]f32{ 1.0, 1.0 },
                [_]f32{ 1.0, 1.0 },
            },
        },
    };

    // result_tensor.info();
    // result_tensor.print();

    const output_location = try allocator.alloc(usize, 4); //coordinates in the output space, see test below
    defer allocator.free(output_location);
    @memset(output_location, 0);

    for (0..2) |batch| {
        output_location[0] = batch;
        for (0..2) |filter| {
            output_location[1] = filter;
            for (0..2) |row| {
                output_location[2] = row;
                for (0..2) |col| {
                    output_location[3] = col;
                    //std.debug.print("\n get OUTPUT at:{any}", .{output_location});
                    try std.testing.expectEqual(expected_result[batch][filter][row][col], result_tensor.get_at(output_location));
                }
            }
        }
    }
}

test "convolution_backward_biases() " {
    std.debug.print("\n     test: convolution_backward_biases \n", .{});

    const allocator = pkgAllocator.allocator;

    var d_val_shape: [4]usize = [_]usize{ 2, 2, 2, 2 };
    var d_val_array: [2][2][2][2]f32 = [_][2][2][2]f32{
        // Primo batch
        [_][2][2]f32{
            [_][2]f32{
                [_]f32{ 3.0, 5.0 },
                [_]f32{ 5.0, 5.0 },
            },
            [_][2]f32{
                [_]f32{ 1.0, 1.0 },
                [_]f32{ 1.0, 1.0 },
            },
        },
        // Secondo batch
        [_][2][2]f32{
            [_][2]f32{
                [_]f32{ 14.0, 14.0 },
                [_]f32{ 14.0, 14.0 },
            },
            [_][2]f32{
                [_]f32{ 10.0, 10.0 },
                [_]f32{ 10.0, 10.0 },
            },
        },
    };

    var d_val = try Tensor(f32).fromArray(&allocator, &d_val_array, &d_val_shape);
    defer d_val.deinit();

    //compute bias derivate
    var result_tensor = try TensMath.convolution_backward_biases(f32, &d_val);
    defer result_tensor.deinit();

    result_tensor.print();

    var d_bias_shape: [1]usize = [_]usize{2};
    var d_bias_array: [2]f32 = [_]f32{ 74, 44 };

    var d_bias_expected = try Tensor(f32).fromArray(&allocator, &d_bias_array, &d_bias_shape);
    defer d_bias_expected.deinit();

    //check on values
    for (d_bias_expected.data, 0..) |expected, i| {
        try std.testing.expectEqual(expected, result_tensor.data[i]);
    }

    //check on dim
    try std.testing.expectEqual(d_bias_expected.size, result_tensor.size);

    for (d_bias_expected.shape, 0..) |expected, i| {
        try std.testing.expectEqual(expected, result_tensor.shape[i]);
    }
}

test "convolution_backward_weights() " {
    std.debug.print("\n     test: convolution_backward_weights \n", .{});

    const allocator = pkgAllocator.allocator;

    var input_shape: [4]usize = [_]usize{ 2, 2, 3, 4 };
    var inputArray: [2][2][3][4]f32 = [_][2][3][4]f32{ //batches:2, channels:2, rows:3, cols:4
        //First Batch
        [_][3][4]f32{
            // First Channel
            [_][4]f32{
                [_]f32{ 2.0, 2.0, 3.0, 0.0 },
                [_]f32{ 4.0, 5.0, 6.0, 0.0 },
                [_]f32{ 7.0, 8.0, 9.0, 0.0 },
            },
            // Second Channel
            [_][4]f32{
                [_]f32{ 8.0, 8.0, 7.0, 0.0 },
                [_]f32{ 6.0, 5.0, 4.0, 0.0 },
                [_]f32{ 3.0, 2.0, 1.0, 0.0 },
            },
        },
        // Second batch
        [_][3][4]f32{
            // First channel
            [_][4]f32{
                [_]f32{ 2.0, 3.0, 4.0, 0.0 },
                [_]f32{ 5.0, 6.0, 7.0, 0.0 },
                [_]f32{ 8.0, 9.0, 10.0, 0.0 },
            },
            // Second channel
            [_][4]f32{
                [_]f32{ 10.0, 9.0, 8.0, 0.0 },
                [_]f32{ 7.0, 6.0, 5.0, 0.0 },
                [_]f32{ 4.0, 3.0, 2.0, 0.0 },
            },
        },
    };
    var input_tensor = try Tensor(f32).fromArray(&allocator, &inputArray, &input_shape);
    defer input_tensor.deinit();

    // Kernel tensor
    var kernel_shape: [4]usize = [_]usize{ 2, 2, 2, 2 };
    var kernelArray: [2][2][2][2]f32 = [_][2][2][2]f32{ //filters:2, channels:2, rows:2, cols:2
        //first filter
        [_][2][2]f32{
            //first channel
            [_][2]f32{
                [_]f32{ -1.0, 0.0 },
                [_]f32{ 0.0, 1.0 },
            },
            //second channel
            [_][2]f32{
                [_]f32{ 1.0, -1.0 },
                [_]f32{ -1.0, 1.0 },
            },
        },
        //second filter
        [_][2][2]f32{
            //first channel
            [_][2]f32{
                [_]f32{ 0.0, 0.0 },
                [_]f32{ 0.0, 0.0 },
            },
            //second channel
            [_][2]f32{
                [_]f32{ 0.0, 0.0 },
                [_]f32{ 0.0, 0.0 },
            },
        },
    };
    var kernel_tensor = try Tensor(f32).fromArray(&allocator, &kernelArray, &kernel_shape);
    defer kernel_tensor.deinit();

    // d_val
    var d_val_shape: [4]usize = [_]usize{ 2, 2, 2, 3 };
    var d_val_array: [2][2][2][3]f32 = [_][2][2][3]f32{
        // Primo batch
        [_][2][3]f32{
            [_][3]f32{
                [_]f32{ 0, 0, 0 },
                [_]f32{ 0, 0, 0 },
            },
            [_][3]f32{
                [_]f32{ 0, 0, 0 },
                [_]f32{ 0, 0, 0 },
            },
        },
        // Secondo batch
        [_][2][3]f32{
            [_][3]f32{
                [_]f32{ 1.0, 1.0, 1.0 },
                [_]f32{ 1.0, 1.0, 1.0 },
            },
            [_][3]f32{
                [_]f32{ 1.0, 1.0, 1.0 },
                [_]f32{ 1.0, 1.0, 1.0 },
            },
        },
    };

    var d_val_tensor = try Tensor(f32).fromArray(&allocator, &d_val_array, &d_val_shape);
    defer d_val_tensor.deinit();

    //create stride
    const stride: [2]usize = [_]usize{ 1, 1 };

    //creating all zero bias
    var bias_array: [2]f32 = [_]f32{ 1, 1 }; //batches: 2, filters:2
    var bias_shape: [1]usize = [_]usize{2};
    var bias = try Tensor(f32).fromArray(&allocator, &bias_array, &bias_shape);
    defer bias.deinit();

    //generating an output
    var output_tensor = try TensMath.convolution_backward_weights(
        f32,
        &input_tensor,
        &d_val_tensor,
        kernel_tensor.shape[0..],
        stride,
    );
    defer output_tensor.deinit();

    output_tensor.info();
    output_tensor.print();

    //compute bias derivate
    var d_weights = try TensMath.convolution_backward_weights(
        f32,
        &input_tensor,
        &d_val_tensor,
        kernel_tensor.shape[0..],
        stride,
    );
    defer d_weights.deinit();
}

test "convolution_backward_weights() small" {
    std.debug.print("\n     test: convolution_backward_weights \n", .{});

    const allocator = pkgAllocator.allocator;

    var input_shape: [4]usize = [_]usize{ 2, 2, 3, 3 };
    var inputArray: [2][2][3][3]f32 = [_][2][3][3]f32{ //batches:2, channels:2, rows:3, cols:3
        //First Batch
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
    var input_tensor = try Tensor(f32).fromArray(&allocator, &inputArray, &input_shape);
    defer input_tensor.deinit();

    // Kernel tensor
    var kernel_shape: [4]usize = [_]usize{ 2, 2, 2, 2 };
    var kernelArray: [2][2][2][2]f32 = [_][2][2][2]f32{ //filters:2, channels:2, rows:2, cols:2
        //first filter
        [_][2][2]f32{
            //first channel
            [_][2]f32{
                [_]f32{ -1.0, 0.0 },
                [_]f32{ 0.0, 1.0 },
            },
            //second channel
            [_][2]f32{
                [_]f32{ 1.0, -1.0 },
                [_]f32{ -1.0, 1.0 },
            },
        },
        //second filter
        [_][2][2]f32{
            //first channel
            [_][2]f32{
                [_]f32{ 0.0, 0.0 },
                [_]f32{ 0.0, 0.0 },
            },
            //second channel
            [_][2]f32{
                [_]f32{ 0.0, 0.0 },
                [_]f32{ 0.0, 0.0 },
            },
        },
    };
    var kernel_tensor = try Tensor(f32).fromArray(&allocator, &kernelArray, &kernel_shape);
    defer kernel_tensor.deinit();

    // d_val
    var d_val_shape: [4]usize = [_]usize{ 2, 2, 2, 2 };
    var d_val_array: [2][2][2][2]f32 = [_][2][2][2]f32{
        // Primo batch
        [_][2][2]f32{
            [_][2]f32{
                [_]f32{ 0, 0 },
                [_]f32{ 0, 0 },
            },
            [_][2]f32{
                [_]f32{ 0, 0 },
                [_]f32{ 0, 0 },
            },
        },
        // Secondo batch
        [_][2][2]f32{
            [_][2]f32{
                [_]f32{ 1.0, 1.0 },
                [_]f32{ 1.0, 1.0 },
            },
            [_][2]f32{
                [_]f32{ 1.0, 1.0 },
                [_]f32{ 1.0, 1.0 },
            },
        },
    };

    var d_val_tensor = try Tensor(f32).fromArray(&allocator, &d_val_array, &d_val_shape);
    defer d_val_tensor.deinit();

    //create stride
    const stride: [2]usize = [_]usize{ 1, 1 };

    //creating all zero bias
    var bias_array: [2]f32 = [_]f32{ 1, 1 }; //batches: 2, filters:2
    var bias_shape: [1]usize = [_]usize{2};
    var bias = try Tensor(f32).fromArray(&allocator, &bias_array, &bias_shape);
    defer bias.deinit();

    //generating an output
    var output_tensor = try TensMath.convolution_backward_weights(
        f32,
        &input_tensor,
        &d_val_tensor,
        kernel_tensor.shape[0..],
        stride,
    );
    defer output_tensor.deinit();

    output_tensor.info();
    output_tensor.print();

    //compute bias derivate
    var d_weights = try TensMath.convolution_backward_weights(
        f32,
        &input_tensor,
        &d_val_tensor,
        kernel_tensor.shape[0..],
        stride,
    );
    defer d_weights.deinit();

    // d_weights.info();
    // d_weights.print();
}

test "convolution_backward_input() " {
    std.debug.print("\n     test: convolution_backward_input() \n", .{});

    const allocator = pkgAllocator.allocator;

    var input_shape: [4]usize = [_]usize{ 2, 2, 3, 3 };
    var inputArray: [2][2][3][3]f32 = [_][2][3][3]f32{ //batches:2, channels:2, rows:3, cols:3
        //First Batch
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
    var input_tensor = try Tensor(f32).fromArray(&allocator, &inputArray, &input_shape);
    defer input_tensor.deinit();

    // Kernel tensor
    var kernel_shape: [4]usize = [_]usize{ 2, 2, 2, 2 };
    var kernelArray: [2][2][2][2]f32 = [_][2][2][2]f32{ //filters:2, channels:2, rows:2, cols:2
        //first filter
        [_][2][2]f32{
            //first channel
            [_][2]f32{
                [_]f32{ -1.0, 0.0 },
                [_]f32{ 0.0, 1.0 },
            },
            //second channel
            [_][2]f32{
                [_]f32{ 1.0, -1.0 },
                [_]f32{ -1.0, 1.0 },
            },
        },
        //second filter
        [_][2][2]f32{
            //first channel
            [_][2]f32{
                [_]f32{ 0.0, 0.0 },
                [_]f32{ 0.0, 0.0 },
            },
            //second channel
            [_][2]f32{
                [_]f32{ 0.0, 0.0 },
                [_]f32{ 0.0, 0.0 },
            },
        },
    };
    var kernel_tensor = try Tensor(f32).fromArray(&allocator, &kernelArray, &kernel_shape);
    defer kernel_tensor.deinit();

    // d_val
    var d_val_shape: [4]usize = [_]usize{ 2, 2, 2, 2 };
    var d_val_array: [2][2][2][2]f32 = [_][2][2][2]f32{
        // Primo batch
        [_][2][2]f32{
            [_][2]f32{
                [_]f32{ 1.0, 1.0 },
                [_]f32{ 1.0, 1.0 },
            },
            [_][2]f32{
                [_]f32{ 1.0, 1.0 },
                [_]f32{ 1.0, 1.0 },
            },
        },
        // Secondo batch
        [_][2][2]f32{
            [_][2]f32{
                [_]f32{ 0, 0 },
                [_]f32{ 0, 0 },
            },
            [_][2]f32{
                [_]f32{ 0, 0 },
                [_]f32{ 0, 0 },
            },
        },
    };

    var d_val_tensor = try Tensor(f32).fromArray(&allocator, &d_val_array, &d_val_shape);
    defer d_val_tensor.deinit();

    //create stride
    const stride: [2]usize = [_]usize{ 1, 1 };

    //creating all zero bias
    var bias_array: [2][2]f32 = [_][2]f32{ //batches: 2, filters:2
        [_]f32{ 1, 1 },
        [_]f32{ 10, 10 },
    };
    var bias_shape: [2]usize = [_]usize{ 2, 2 };
    var bias = try Tensor(f32).fromArray(&allocator, &bias_array, &bias_shape);
    defer bias.deinit();

    //generating an output
    var output_tensor = try TensMath.convolution_backward_input(
        f32,
        &d_val_tensor,
        &kernel_tensor,
        input_tensor.shape[0..],
        stride,
    );
    defer output_tensor.deinit();

    output_tensor.info();
    output_tensor.print();

    //compute bias derivate
    var d_input = try TensMath.convolution_backward_input(
        f32,
        &d_val_tensor,
        &kernel_tensor,
        input_tensor.shape[0..],
        stride,
    );
    defer d_input.deinit();

    std.debug.print("\n   ----------------------------------------------  \n", .{});
    d_input.print();
    d_input.info();
}

// test "get_convolution_output_shape()" {
//     std.debug.print("\n     test: get_convolution_output_shape \n", .{});

//     var input_shape = [_]usize{ 2, 2, 5, 5 }; // batch=2, channels=2, height=5, width=5
//     var kernel_shape = [_]usize{ 3, 2, 3, 3 }; // filters=3, channels=2, height=3, width=3
//     var stride = [_]usize{ 1, 1 };

//     var output_shape = try TensMath.get_convolution_output_shape(&input_shape, &kernel_shape, &stride);

//     try std.testing.expectEqual(@as(usize, 2), output_shape[0]); // batch size
//     try std.testing.expectEqual(@as(usize, 3), output_shape[1]); // num filters
//     try std.testing.expectEqual(@as(usize, 3), output_shape[2]); // output height
//     try std.testing.expectEqual(@as(usize, 3), output_shape[3]); // output width

//     // Test with different stride
//     stride = [_]usize{ 2, 2 };
//     output_shape = try TensMath.get_convolution_output_shape(&input_shape, &kernel_shape, &stride);

//     try std.testing.expectEqual(@as(usize, 2), output_shape[0]); // batch size
//     try std.testing.expectEqual(@as(usize, 3), output_shape[1]); // num filters
//     try std.testing.expectEqual(@as(usize, 2), output_shape[2]); // output height
//     try std.testing.expectEqual(@as(usize, 2), output_shape[3]); // output width

//     // Test invalid dimensions
//     var invalid_input_shape = [_]usize{ 2, 2, 5 };
//     try std.testing.expectError(TensorMathError.InvalidDimensions, TensMath.get_convolution_output_shape(&invalid_input_shape, &kernel_shape, &stride));

//     // Test invalid stride
//     var invalid_stride = [_]usize{ 0, 1 };
//     try std.testing.expectError(TensorMathError.WrongStride, TensMath.get_convolution_output_shape(&input_shape, &kernel_shape, &invalid_stride));
// }

test "OnnxConvLean - NOTSET padding" {
    std.debug.print("\n     test: OnnxConvLean - NOTSET padding\n", .{});

    const allocator = pkgAllocator.allocator;

    // Input tensor
    var input_shape: [4]usize = [_]usize{ 1, 1, 5, 5 };
    var inputArray: [1][1][5][5]f32 = [_][1][5][5]f32{
        [_][5][5]f32{
            [_][5]f32{
                [_]f32{ 1, 1, 1, 1, 1 },
                [_]f32{ 1, 1, 1, 1, 1 },
                [_]f32{ 1, 1, 1, 1, 1 },
                [_]f32{ 1, 1, 1, 1, 1 },
                [_]f32{ 1, 1, 1, 1, 1 },
            },
        },
    };

    // Kernel tensor
    var kernel_shape: [4]usize = [_]usize{ 1, 1, 3, 3 };
    var kernelArray: [1][1][3][3]f32 = [_][1][3][3]f32{
        [_][3][3]f32{
            [_][3]f32{
                [_]f32{ 1, 1, 1 },
                [_]f32{ 1, 1, 1 },
                [_]f32{ 1, 1, 1 },
            },
        },
    };

    var input_tensor = try Tensor(f32).fromArray(&allocator, &inputArray, &input_shape);
    defer input_tensor.deinit();
    var kernel_tensor = try Tensor(f32).fromArray(&allocator, &kernelArray, &kernel_shape);
    defer kernel_tensor.deinit();

    const stride = [_]usize{1};
    const pads = [_]usize{ 0, 0, 0, 0 };

    // Create output tensor with correct shape
    var output_shape = [_]usize{ 1, 1, 3, 3 };
    var output_tensor = try Tensor(f32).fromShape(&allocator, &output_shape);
    defer output_tensor.deinit();

    try TensMath.conv_lean(f32, &input_tensor, &kernel_tensor, &output_tensor, null, &stride, &pads, null, null, null);

    try std.testing.expectEqual(@as(usize, 1), output_tensor.shape[0]); // batch
    try std.testing.expectEqual(@as(usize, 1), output_tensor.shape[1]); // channels
    try std.testing.expectEqual(@as(usize, 3), output_tensor.shape[2]); // height
    try std.testing.expectEqual(@as(usize, 3), output_tensor.shape[3]); // width

    // Each output should be 9 (sum of 3x3 kernel of ones)
    for (output_tensor.data) |val| {
        try std.testing.expectEqual(@as(f32, 9), val);
    }
}

test "OnnxConvLean - SAME_UPPER padding" {
    std.debug.print("\n     test: OnnxConvLean - SAME_UPPER padding\n", .{});

    const allocator = pkgAllocator.allocator;

    // Input tensor
    var input_shape: [4]usize = [_]usize{ 1, 1, 5, 5 };
    var inputArray: [1][1][5][5]f32 = [_][1][5][5]f32{
        [_][5][5]f32{
            [_][5]f32{
                [_]f32{ 1, 1, 1, 1, 1 },
                [_]f32{ 1, 1, 1, 1, 1 },
                [_]f32{ 1, 1, 1, 1, 1 },
                [_]f32{ 1, 1, 1, 1, 1 },
                [_]f32{ 1, 1, 1, 1, 1 },
            },
        },
    };

    // Kernel tensor
    var kernel_shape: [4]usize = [_]usize{ 1, 1, 3, 3 };
    var kernelArray: [1][1][3][3]f32 = [_][1][3][3]f32{
        [_][3][3]f32{
            [_][3]f32{
                [_]f32{ 1, 1, 1 },
                [_]f32{ 1, 1, 1 },
                [_]f32{ 1, 1, 1 },
            },
        },
    };

    var input_tensor = try Tensor(f32).fromArray(&allocator, &inputArray, &input_shape);
    defer input_tensor.deinit();
    var kernel_tensor = try Tensor(f32).fromArray(&allocator, &kernelArray, &kernel_shape);
    defer kernel_tensor.deinit();

    const stride = [_]usize{1};
    const auto_pad = "SAME_UPPER";

    // Create output tensor with correct shape (same as input for SAME_UPPER)
    var output_shape = [_]usize{ 1, 1, 5, 5 };
    var output_tensor = try Tensor(f32).fromShape(&allocator, &output_shape);
    defer output_tensor.deinit();

    try TensMath.conv_lean(f32, &input_tensor, &kernel_tensor, &output_tensor, null, &stride, null, null, null, auto_pad);

    // Add debug prints for padded input
    std.debug.print("\nKernel values:\n", .{});
    var k_row: usize = 0;
    while (k_row < 3) : (k_row += 1) {
        var k_col: usize = 0;
        while (k_col < 3) : (k_col += 1) {
            const idx = k_row * 3 + k_col;
            std.debug.print("{d:4.1} ", .{kernel_tensor.data[idx]});
        }
        std.debug.print("\n", .{});
    }

    try std.testing.expectEqual(@as(usize, 1), output_tensor.shape[0]); // batch
    try std.testing.expectEqual(@as(usize, 1), output_tensor.shape[1]); // channels
    try std.testing.expectEqual(@as(usize, 5), output_tensor.shape[2]); // height
    try std.testing.expectEqual(@as(usize, 5), output_tensor.shape[3]); // width

    // Center values should be 9, edge values less due to padding
    const expected_values = [_]f32{
        4, 6, 6, 6, 4,
        6, 9, 9, 9, 6,
        6, 9, 9, 9, 6,
        6, 9, 9, 9, 6,
        4, 6, 6, 6, 4,
    };

    std.debug.print("\nResult shape: {any}\n", .{output_tensor.shape});
    std.debug.print("\nActual values:\n", .{});
    var row: usize = 0;
    while (row < 5) : (row += 1) {
        var col: usize = 0;
        while (col < 5) : (col += 1) {
            const idx = row * 5 + col;
            std.debug.print("{d:4.1} ", .{output_tensor.data[idx]});
        }
        std.debug.print("\n", .{});
    }

    std.debug.print("\nExpected values:\n", .{});
    row = 0;
    while (row < 5) : (row += 1) {
        var col: usize = 0;
        while (col < 5) : (col += 1) {
            const idx = row * 5 + col;
            std.debug.print("{d:4.1} ", .{expected_values[idx]});
        }
        std.debug.print("\n", .{});
    }

    for (output_tensor.data, 0..) |val, i| {
        if (val != expected_values[i]) {
            std.debug.print("\nMismatch at index {d}: expected {d}, got {d}\n", .{ i, expected_values[i], val });
        }
        try std.testing.expectEqual(expected_values[i], val);
    }
}

test "OnnxConvLean - with bias and dilation" {
    std.debug.print("\n     test: OnnxConvLean - with bias and dilation\n", .{});

    const allocator = pkgAllocator.allocator;

    // Input tensor
    var input_shape: [4]usize = [_]usize{ 1, 1, 5, 5 };
    var inputArray: [1][1][5][5]f32 = [_][1][5][5]f32{
        [_][5][5]f32{
            [_][5]f32{
                [_]f32{ 1, 1, 1, 1, 1 },
                [_]f32{ 1, 1, 1, 1, 1 },
                [_]f32{ 1, 1, 1, 1, 1 },
                [_]f32{ 1, 1, 1, 1, 1 },
                [_]f32{ 1, 1, 1, 1, 1 },
            },
        },
    };

    // Kernel tensor
    var kernel_shape: [4]usize = [_]usize{ 1, 1, 2, 2 };
    var kernelArray: [1][1][2][2]f32 = [_][1][2][2]f32{
        [_][2][2]f32{
            [_][2]f32{
                [_]f32{ 1, 1 },
                [_]f32{ 1, 1 },
            },
        },
    };

    // Bias tensor
    var bias_shape: [1]usize = [_]usize{1};
    var biasArray: [1]f32 = [_]f32{1};

    var input_tensor = try Tensor(f32).fromArray(&allocator, &inputArray, &input_shape);
    defer input_tensor.deinit();
    var kernel_tensor = try Tensor(f32).fromArray(&allocator, &kernelArray, &kernel_shape);
    defer kernel_tensor.deinit();
    var bias_tensor = try Tensor(f32).fromArray(&allocator, &biasArray, &bias_shape);
    defer bias_tensor.deinit();

    const stride = [_]usize{1};
    const dilations = [_]usize{2};

    // Create output tensor with correct shape
    var output_shape = [_]usize{ 1, 1, 3, 3 };
    var output_tensor = try Tensor(f32).fromShape(&allocator, &output_shape);
    defer output_tensor.deinit();

    try TensMath.conv_lean(f32, &input_tensor, &kernel_tensor, &output_tensor, &bias_tensor, &stride, null, &dilations, null, null);

    try std.testing.expectEqual(@as(usize, 1), output_tensor.shape[0]); // batch
    try std.testing.expectEqual(@as(usize, 1), output_tensor.shape[1]); // channels
    try std.testing.expectEqual(@as(usize, 3), output_tensor.shape[2]); // height
    try std.testing.expectEqual(@as(usize, 3), output_tensor.shape[3]); // width

    // Each output should be 5 (4 from dilated kernel + 1 from bias)
    for (output_tensor.data) |val| {
        try std.testing.expectEqual(@as(f32, 5), val);
    }
}

test "OnnxConv - all padding modes and features" {
    std.debug.print("\n     test: OnnxConv - all padding modes and features\n", .{});

    const allocator = pkgAllocator.allocator;

    // Test 1: NOTSET padding
    {
        // Input tensor
        var input_shape: [4]usize = [_]usize{ 1, 1, 5, 5 };
        var inputArray: [1][1][5][5]f32 = [_][1][5][5]f32{
            [_][5][5]f32{
                [_][5]f32{
                    [_]f32{ 1, 1, 1, 1, 1 },
                    [_]f32{ 1, 1, 1, 1, 1 },
                    [_]f32{ 1, 1, 1, 1, 1 },
                    [_]f32{ 1, 1, 1, 1, 1 },
                    [_]f32{ 1, 1, 1, 1, 1 },
                },
            },
        };

        // Kernel tensor
        var kernel_shape: [4]usize = [_]usize{ 1, 1, 3, 3 };
        var kernelArray: [1][1][3][3]f32 = [_][1][3][3]f32{
            [_][3][3]f32{
                [_][3]f32{
                    [_]f32{ 1, 1, 1 },
                    [_]f32{ 1, 1, 1 },
                    [_]f32{ 1, 1, 1 },
                },
            },
        };

        var input_tensor = try Tensor(f32).fromArray(&allocator, &inputArray, &input_shape);
        defer input_tensor.deinit();
        var kernel_tensor = try Tensor(f32).fromArray(&allocator, &kernelArray, &kernel_shape);
        defer kernel_tensor.deinit();

        const stride = [_]usize{1};
        const pads = [_]usize{ 0, 0, 0, 0 };

        var result = try TensMath.Conv(f32, &input_tensor, &kernel_tensor, null, &stride, &pads, null, null, null);
        defer result.deinit();

        try std.testing.expectEqual(@as(usize, 1), result.shape[0]); // batch
        try std.testing.expectEqual(@as(usize, 1), result.shape[1]); // channels
        try std.testing.expectEqual(@as(usize, 3), result.shape[2]); // height
        try std.testing.expectEqual(@as(usize, 3), result.shape[3]); // width

        // Each output should be 9 (sum of 3x3 kernel of ones)
        for (result.data) |val| {
            try std.testing.expectEqual(@as(f32, 9), val);
        }
    }

    // Test 2: SAME_UPPER padding
    {
        // Input tensor
        var input_shape: [4]usize = [_]usize{ 1, 1, 5, 5 };
        var inputArray: [1][1][5][5]f32 = [_][1][5][5]f32{
            [_][5][5]f32{
                [_][5]f32{
                    [_]f32{ 1, 1, 1, 1, 1 },
                    [_]f32{ 1, 1, 1, 1, 1 },
                    [_]f32{ 1, 1, 1, 1, 1 },
                    [_]f32{ 1, 1, 1, 1, 1 },
                    [_]f32{ 1, 1, 1, 1, 1 },
                },
            },
        };

        // Kernel tensor
        var kernel_shape: [4]usize = [_]usize{ 1, 1, 3, 3 };
        var kernelArray: [1][1][3][3]f32 = [_][1][3][3]f32{
            [_][3][3]f32{
                [_][3]f32{
                    [_]f32{ 1, 1, 1 },
                    [_]f32{ 1, 1, 1 },
                    [_]f32{ 1, 1, 1 },
                },
            },
        };

        var input_tensor = try Tensor(f32).fromArray(&allocator, &inputArray, &input_shape);
        defer input_tensor.deinit();
        var kernel_tensor = try Tensor(f32).fromArray(&allocator, &kernelArray, &kernel_shape);
        defer kernel_tensor.deinit();

        const stride = [_]usize{1};
        const auto_pad = "SAME_UPPER";

        var result = try TensMath.Conv(f32, &input_tensor, &kernel_tensor, null, &stride, null, null, null, auto_pad);
        defer result.deinit();

        try std.testing.expectEqual(@as(usize, 1), result.shape[0]); // batch
        try std.testing.expectEqual(@as(usize, 1), result.shape[1]); // channels
        try std.testing.expectEqual(@as(usize, 5), result.shape[2]); // height
        try std.testing.expectEqual(@as(usize, 5), result.shape[3]); // width

        // Center values should be 9, edge values less due to padding
        const expected_values = [_]f32{
            4, 6, 6, 6, 4,
            6, 9, 9, 9, 6,
            6, 9, 9, 9, 6,
            6, 9, 9, 9, 6,
            4, 6, 6, 6, 4,
        };

        for (result.data, 0..) |val, i| {
            try std.testing.expectEqual(expected_values[i], val);
        }
    }

    // Test 3: With bias and dilation
    {
        // Input tensor
        var input_shape: [4]usize = [_]usize{ 1, 1, 5, 5 };
        var inputArray: [1][1][5][5]f32 = [_][1][5][5]f32{
            [_][5][5]f32{
                [_][5]f32{
                    [_]f32{ 1, 1, 1, 1, 1 },
                    [_]f32{ 1, 1, 1, 1, 1 },
                    [_]f32{ 1, 1, 1, 1, 1 },
                    [_]f32{ 1, 1, 1, 1, 1 },
                    [_]f32{ 1, 1, 1, 1, 1 },
                },
            },
        };

        // Kernel tensor
        var kernel_shape: [4]usize = [_]usize{ 1, 1, 2, 2 };
        var kernelArray: [1][1][2][2]f32 = [_][1][2][2]f32{
            [_][2][2]f32{
                [_][2]f32{
                    [_]f32{ 1, 1 },
                    [_]f32{ 1, 1 },
                },
            },
        };

        // Bias tensor
        var bias_shape: [1]usize = [_]usize{1};
        var biasArray: [1]f32 = [_]f32{1};

        var input_tensor = try Tensor(f32).fromArray(&allocator, &inputArray, &input_shape);
        defer input_tensor.deinit();
        var kernel_tensor = try Tensor(f32).fromArray(&allocator, &kernelArray, &kernel_shape);
        defer kernel_tensor.deinit();
        var bias_tensor = try Tensor(f32).fromArray(&allocator, &biasArray, &bias_shape);
        defer bias_tensor.deinit();

        const stride = [_]usize{1};
        const dilations = [_]usize{2};

        var result = try TensMath.Conv(f32, &input_tensor, &kernel_tensor, &bias_tensor, &stride, null, &dilations, null, null);
        defer result.deinit();

        try std.testing.expectEqual(@as(usize, 1), result.shape[0]); // batch
        try std.testing.expectEqual(@as(usize, 1), result.shape[1]); // channels
        try std.testing.expectEqual(@as(usize, 3), result.shape[2]); // height
        try std.testing.expectEqual(@as(usize, 3), result.shape[3]); // width

        // Each output should be 5 (4 from dilated kernel + 1 from bias)
        for (result.data) |val| {
            try std.testing.expectEqual(@as(f32, 5), val);
        }
    }
}
