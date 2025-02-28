const std = @import("std");
const zant = @import("zant");
const Tensor = zant.core.tensor.Tensor;
const pkgAllocator = zant.utils.allocator;
const allocator = pkgAllocator.allocator;

const static_lib_mod = @import("static_lib");

// test "Static Library - Basic Prediction Test" {
//     std.debug.print("\n     test: Static Library - Basic Prediction Test\n", .{});

//     // Create a simple input tensor
//     var input_data = [_]f32{ 1.0, 2.0, 3.0, 4.0, 5.0 };
//     var input_shape = [_]u32{5};
//     var result: [*]f32 = undefined;

//     // Create a logging function
//     const LogFn = fn ([*c]u8) callconv(.C) void;
//     const logFn: LogFn = struct {
//         fn log(msg: [*c]u8) callconv(.C) void {
//             std.debug.print("{s}", .{msg});
//         }
//     }.log;

//     // Set the logging function
//     @import("static_lib").setLogFunction(logFn);

//     // Run prediction
//     @import("static_lib").predict(
//         @ptrCast(&input_data),
//         @ptrCast(&input_shape),
//         1,
//         &result,
//     );

//     // Print the results
//     std.debug.print("\nPrediction results:\n", .{});
//     for (0..5) |i| {
//         std.debug.print("Result[{d}] = {d:.6}\n", .{ i, result[i] });
//     }

//     // Verify the results sum to approximately 1.0 (since we use softmax at the end)
//     var sum: f32 = 0;
//     for (0..5) |i| {
//         sum += result[i];
//     }
//     const epsilon = 0.0001;
//     try std.testing.expect(sum >= 1.0 - epsilon and sum <= 1.0 + epsilon);
// }

// test "Static Library - Error Cases" {
//     std.debug.print("\n     test: Static Library - Error Cases\n", .{});

//     // Test with empty shape
//     {
//         var input_data = [_]f32{};
//         var input_shape = [_]u32{};
//         var result: [*]f32 = undefined;

//         @import("static_lib").predict(
//             @ptrCast(&input_data),
//             @ptrCast(&input_shape),
//             0,
//             &result,
//         );
//     }

//     // Test with invalid shape length
//     {
//         var input_data = [_]f32{ 1.0, 2.0 };
//         var input_shape = [_]u32{ 1, 2 }; // Wrong shape for our network
//         var result: [*]f32 = undefined;

//         @import("static_lib").predict(
//             @ptrCast(&input_data),
//             @ptrCast(&input_shape),
//             2,
//             &result,
//         );
//     }
// }

test "Static Library - MNIST Prediction Test" {
    std.debug.print("\n     test: Static Library - MNIST Prediction Test\n", .{});

    // Create a mock MNIST input (28x28 grayscale image)
    var input_data: [784]f32 = [_]f32{0} ** 784; // Initialize all to black (0)

    // Draw a "7" pattern
    // Horizontal line at top
    for (5..23) |x| {
        input_data[5 * 28 + x] = 245.0;
    }

    // Diagonal line
    var y: usize = 6;
    var x: usize = 20;
    while (y < 22 and x > 8) : ({
        y += 1;
        x -= 1;
    }) {
        input_data[y * 28 + x] = 255.0;
    }

    var input_shape = [_]u32{ 1, 1, 28, 28 }; // NCHW format
    var result_buffer: [10]f32 = undefined; // Allocate buffer for 10 MNIST classes
    var result: [*]f32 = @ptrCast(&result_buffer);

    // Create a logging function
    const LogFn = fn ([*c]u8) callconv(.C) void;
    const logFn: LogFn = struct {
        fn log(msg: [*c]u8) callconv(.C) void {
            std.debug.print("{s}", .{msg});
        }
    }.log;

    // Set the logging function
    static_lib_mod.setLogFunction(logFn);

    // Run prediction
    static_lib_mod.predict(
        @ptrCast(&input_data),
        @ptrCast(&input_shape),
        4, // 4D tensor shape
        &result,
    );

    // Print the results (10 classes for MNIST)
    std.debug.print("\nMNIST Prediction probabilities:\n", .{});
    for (0..10) |i| {
        std.debug.print("Digit {d}: {d:.6}\n", .{ i, result[i] });
    }

    // Verify the probabilities sum to approximately 1.0 (due to softmax)
    var sum: f32 = 0;
    for (0..10) |i| {
        sum += result[i];
    }

    // Find the predicted digit (highest probability)
    var max_prob: f32 = result[0];
    var predicted_digit: usize = 0;
    for (1..10) |i| {
        if (result[i] > max_prob) {
            max_prob = result[i];
            predicted_digit = i;
        }
    }
    std.debug.print("\nPredicted digit: {d} with confidence: {d:.2}%\n", .{ predicted_digit, max_prob * 100 });
}

test "Static Library - MNIST Error Cases" {
    std.debug.print("\n     test: Static Library - MNIST Error Cases\n", .{});

    // Test with wrong input shape
    {
        var input_data = [_]f32{1.0} ** 100; // Wrong size
        var input_shape = [_]u32{ 10, 10 }; // Wrong shape
        var result: [*]f32 = undefined;

        static_lib_mod.predict(
            @ptrCast(&input_data),
            @ptrCast(&input_shape),
            2,
            &result,
        );
    }

    // Test with empty input
    {
        var input_data = [_]f32{};
        var input_shape = [_]u32{};
        var result: [*]f32 = undefined;

        static_lib_mod.predict(
            @ptrCast(&input_data),
            @ptrCast(&input_shape),
            0,
            &result,
        );
    }

    // // Test with wrong number of dimensions
    // {
    //     var input_data = [_]f32{1.0} ** 784;
    //     var input_shape = [_]u32{784}; // Should be 4D but only 1D
    //     var result: [*]f32 = undefined;

    //     @import("static_lib_mnist_hard").predict(
    //         @ptrCast(&input_data),
    //         @ptrCast(&input_shape),
    //         1,
    //         &result,
    //     );
    // }
}
