const std = @import("std");
const Tensor = @import("tensor").Tensor;
const ActivFun = @import("activation_function");
const ActivType = @import("activation_function").ActivationType;
const pkgAllocator = @import("pkgAllocator");

test "tests description" {
    std.debug.print("\n--- Running activation_function tests\n", .{});
}

//*********************************************** ReLU ***********************************************
test "ReLU from ActivationFunction()" {
    std.debug.print("\n     test: ReLU from ActivationFunction()", .{});

    const allocator = pkgAllocator.allocator;

    var inputArray: [2][2]f32 = [_][2]f32{
        [_]f32{ -1.0, -2.0 },
        [_]f32{ -4.0, -5.0 },
    };

    var shape: [2]usize = [_]usize{ 2, 2 }; // 2x2 matrix

    var t1 = try Tensor(f32).fromArray(&allocator, &inputArray, &shape);
    defer t1.deinit();

    var relu = ActivFun.ActivationFunction(f32, ActivType.ReLU){};
    //var relu = act_type{};

    try relu.forward(&t1);

    for (t1.data) |*val| {
        try std.testing.expect(0.0 == val.*);
    }
}

test "ReLU all negative" {
    std.debug.print("\n     test: ReLU all negative", .{});

    const allocator = pkgAllocator.allocator;

    var inputArray: [2][2]f32 = [_][2]f32{
        [_]f32{ -1.0, -2.0 },
        [_]f32{ -4.0, -5.0 },
    };

    var shape: [2]usize = [_]usize{ 2, 2 }; // 2x2 matrix

    var t1 = try Tensor(f32).fromArray(&allocator, &inputArray, &shape);
    defer t1.deinit();

    var relu = ActivFun.ActivationFunction(f32, ActivType.ReLU){};
    //var relu = act_type{};

    try relu.forward(&t1);

    for (t1.data) |*val| {
        try std.testing.expect(0.0 == val.*);
    }
}

test "ReLU all positive" {
    std.debug.print("\n     test: ReLU all positive", .{});

    const allocator = pkgAllocator.allocator;

    var inputArray: [2][2]f32 = [_][2]f32{
        [_]f32{ 1.0, 2.0 },
        [_]f32{ 4.0, 5.0 },
    };

    var shape: [2]usize = [_]usize{ 2, 2 }; // 2x2 matrix

    var t1 = try Tensor(f32).fromArray(&allocator, &inputArray, &shape);
    defer t1.deinit();

    var relu = ActivFun.ActivationFunction(f32, ActivType.ReLU){};
    try relu.forward(&t1);

    for (t1.data) |*val| {
        try std.testing.expect(val.* >= 0);
    }
}

//*********************************************** Softmax ***********************************************

test "Softmax from ActivationFunction()" {
    std.debug.print("\n     test: Softmax from ActivationFunction()", .{});
    const allocator = pkgAllocator.allocator;

    var inputArray: [2][2]f32 = [_][2]f32{
        [_]f32{ 0, 0 },
        [_]f32{ 0, 0 },
    };

    var shape: [2]usize = [_]usize{ 2, 2 }; // 2x2 matrix

    var t1 = try Tensor(f32).fromArray(&allocator, &inputArray, &shape);
    defer t1.deinit();

    const soft_type = ActivFun.ActivationFunction(f32, ActivType.Softmax);
    _ = soft_type{ .allocator = &allocator };
}

test "Softmax all positive" {
    std.debug.print("\n     test: Softmax all positive", .{});

    const allocator = pkgAllocator.allocator;

    var inputArray: [2][2]f32 = [_][2]f32{
        [_]f32{ 1.0, 2.0 },
        [_]f32{ 4.0, 5.0 },
    };

    var shape: [2]usize = [_]usize{ 2, 2 }; // 2x2 matrix

    var t1 = try Tensor(f32).fromArray(&allocator, &inputArray, &shape);
    defer t1.deinit();

    var soft = ActivFun.ActivationFunction(f32, ActivType.Softmax){ .allocator = &allocator };
    try soft.forward(&t1);
    //now data is:
    //{ 0.2689414,  0.7310586  }
    //{ 0.2689414,  0.73105854 }

    t1.info();

    try std.testing.expect(t1.data[0] + t1.data[1] > 0.9);
    try std.testing.expect(t1.data[0] + t1.data[1] < 1.1);

    try std.testing.expect(t1.data[2] + t1.data[3] > 0.9);
    try std.testing.expect(t1.data[2] + t1.data[3] < 1.1);
}

test "Softmax all 0" {
    std.debug.print("\n     test: Softmax all 0", .{});

    const allocator = pkgAllocator.allocator;

    var inputArray: [2][2]f32 = [_][2]f32{
        [_]f32{ 0, 0 },
        [_]f32{ 0, 0 },
    };

    var shape: [2]usize = [_]usize{ 2, 2 }; // 2x2 matrix

    var t1 = try Tensor(f32).fromArray(&allocator, &inputArray, &shape);
    defer t1.deinit();

    var soft = ActivFun.ActivationFunction(f32, ActivType.Softmax){ .allocator = &allocator };
    try soft.forward(&t1);

    //t1.info();

    try std.testing.expect(t1.data[0] == t1.data[1]);
    try std.testing.expect(t1.data[2] == t1.data[3]);
}

test "Softmax derivate" {
    std.debug.print("\n     test: Softmax derivate", .{});

    const allocator = pkgAllocator.allocator;

    var inputArray: [2][2]f32 = [_][2]f32{
        [_]f32{ 1.0, 2.0 },
        [_]f32{ 4.0, 5.0 },
    };

    var shape: [2]usize = [_]usize{ 2, 2 }; // 2x2 matrix

    var t1 = try Tensor(f32).fromArray(&allocator, &inputArray, &shape);
    defer t1.deinit();

    var soft = ActivFun.ActivationFunction(f32, ActivType.Softmax){ .allocator = &allocator };
    try soft.forward(&t1);
    //now data is:
    //{ 0.2689414,  0.7310586  }
    //{ 0.2689414,  0.73105854 }
    //t1.info();

    try std.testing.expect(t1.data[0] + t1.data[1] > 0.9);
    try std.testing.expect(t1.data[0] + t1.data[1] < 1.1);

    try std.testing.expect(t1.data[2] + t1.data[3] > 0.9);
    try std.testing.expect(t1.data[2] + t1.data[3] < 1.1);

    try soft.derivate(&t1, &t1);
}

//*********************************************** Sigmoid ***********************************************

test "Sigmoid forward" {
    std.debug.print("\n     test: Sigmoid forward ", .{});

    const allocator = pkgAllocator.allocator;

    const input_data = [_]f64{ 0.0, 2.0, -2.0 }; // input data for the tensor
    var shape: [1]usize = [_]usize{3};
    var input_tensor = try Tensor(f64).fromArray(&allocator, &input_data, &shape); // create tensor from input data
    defer input_tensor.deinit();

    var sigmoid = ActivFun.ActivationFunction(f64, ActivType.Sigmoid){};
    // Test forward pass
    try sigmoid.forward(&input_tensor);
    const expected_forward_output = [_]f64{ 0.5, 0.880797, 0.119203 }; // expected sigmoid output for each input value
    for (input_tensor.data, 0..) |*data, i| {
        try std.testing.expect(@abs(data.* - expected_forward_output[i]) < 1e-6);
    }
}

test "Sigmoid derivate" {
    std.debug.print("\n     test: Sigmoid derivate ", .{});

    const allocator = pkgAllocator.allocator;

    // Setup the gradient and act_forward_out tensors
    var gradient_data = [_]f64{ 0.2, 0.4, 0.6, 0.8 };
    var shape_grad: [1]usize = [_]usize{4};

    var act_forward_out_data = [_]f64{ 0.5, 0.7, 0.3, 0.9 };
    var shape_forw: [1]usize = [_]usize{4};

    var gradient_tensor = try Tensor(f64).fromArray(&allocator, &gradient_data, &shape_grad);
    defer gradient_tensor.deinit();
    var act_forward_out_tensor = try Tensor(f64).fromArray(&allocator, &act_forward_out_data, &shape_forw);
    defer act_forward_out_tensor.deinit();

    // Call the derivate function
    var sigmoid = ActivFun.ActivationFunction(f64, ActivType.Sigmoid){};
    try sigmoid.derivate(&gradient_tensor, &act_forward_out_tensor);

    // Expected values after applying the derivative
    const expected_values = [_]f64{
        0.2 * 0.5 * (1.0 - 0.5),
        0.4 * 0.7 * (1.0 - 0.7),
        0.6 * 0.3 * (1.0 - 0.3),
        0.8 * 0.9 * (1.0 - 0.9),
    };

    // Verify the result
    for (0..gradient_tensor.data.len) |i| {
        try std.testing.expect(gradient_tensor.data[i] - expected_values[i] < 0.0001);
    }
}
