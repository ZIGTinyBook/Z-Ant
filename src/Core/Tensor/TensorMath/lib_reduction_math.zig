//! These operations reduce a tensor to a smaller shape by aggregating values. Like:
//!    Sum: Compute the sum of elements along specific dimensions.
//!   Mean: Compute the average.
//!    Min/Max: Find the minimum or maximum value.
//!    Prod: Compute the product of elements.
//!    Standard Deviation and Variance: Statistical operations.
//!
const std = @import("std");
const zant = @import("../../../zant.zig");

const Tensor = zant.core.tensor.Tensor; // Import Tensor type
const pkg_allocator = zant.utils.allocator.allocator;
const TensorMathError = zant.utils.error_handler.TensorMathError;
const Converter = zant.utils.type_converter;

/// Performs the mean of a given tensor. It is a reduction operation, collapsing the whole tenosr into a single value.
pub fn mean(comptime T: anytype, tensor: *Tensor(T)) f32 {
    var res: f32 = 0;

    for (tensor.data) |*d| {
        res += Converter.convert(T, f32, d.*);
    }
    res = res / Converter.convert(usize, f32, tensor.size);
    return res;
}

/// Computes the mean of the input tensor's elements along the provided axes.
/// The resulting tensor has the same rank as the input if keepdims equals true.
/// If keepdims equals false, then the resulting tensor has the reduced dimension pruned.
pub fn reduce_mean(comptime T: anytype, tensor: *Tensor(T), axes: ?[]const i64, keepdims: bool, noop_with_empty_axes: bool) !Tensor(T) {
    // Calculate output shape first
    var out_shape: []usize = undefined;
    const allocator = tensor.allocator;

    if (axes == null or axes.?.len == 0) {
        if (noop_with_empty_axes) {
            return tensor.copy();
        }
        // Reduce over all dimensions
        if (keepdims) {
            out_shape = try allocator.alloc(usize, tensor.shape.len);
            @memset(out_shape, 1);
        } else {
            out_shape = try allocator.alloc(usize, 1);
            out_shape[0] = 1;
        }
    } else {
        // Mark dimensions to reduce
        var reduce_dims = try allocator.alloc(bool, tensor.shape.len);
        defer allocator.free(reduce_dims);
        @memset(reduce_dims, false);

        for (axes.?) |axis| {
            const actual_axis = if (axis < 0)
                @as(usize, @intCast(@as(i64, @intCast(tensor.shape.len)) + axis))
            else
                @as(usize, @intCast(axis));
            reduce_dims[actual_axis] = true;
        }

        // Count remaining dimensions and create output shape
        var remaining_dims: usize = 0;
        for (reduce_dims) |is_reduced| {
            if (!is_reduced or keepdims) remaining_dims += 1;
        }

        out_shape = try allocator.alloc(usize, remaining_dims);
        var out_dim: usize = 0;
        for (tensor.shape, 0..) |dim_size, i| {
            if (!reduce_dims[i] or keepdims) {
                out_shape[out_dim] = if (reduce_dims[i]) 1 else dim_size;
                out_dim += 1;
            }
        }
    }

    // Create output tensor
    var output = try Tensor(T).fromShape(allocator, out_shape);
    allocator.free(out_shape);
    errdefer output.deinit();

    try lean_reduce_mean(T, tensor, &output, axes, keepdims, noop_with_empty_axes);
    return output;
}

/// Lean version of reduce_mean that operates on pre-allocated output tensor
pub inline fn lean_reduce_mean(
    comptime T: anytype,
    input_tensor: *Tensor(T),
    output_tensor: *Tensor(T),
    axes: ?[]const i64,
    _: bool, // keepdims (unused since output shape is pre-determined)
    noop_with_empty_axes: bool,
) !void {
    //std.debug.print("\n[DEBUG] lean_reduce_mean:", .{});
    //std.debug.print("\n  Input tensor shape: ", .{});
    //for (input_tensor.shape) |s| std.debug.print("{d} ", .{s});
    //std.debug.print("\n  Output tensor shape: ", .{});
    //for (output_tensor.shape) |s| std.debug.print("{d} ", .{s});
    //std.debug.print("\n  Axes: ", .{});
    //if (axes) |a| {
    //    if (a.len == 0) {
    //        std.debug.print("empty", .{});
    //    } else {
    //        for (a) |axis| std.debug.print("{d} ", .{axis});
    //    }
    //} else {
    //    std.debug.print("null", .{});
    //}
    //std.debug.print("\n  noop_with_empty_axes: {}", .{noop_with_empty_axes});

    // Handle empty axes case
    if (axes == null or axes.?.len == 0) {
        if (noop_with_empty_axes) {
            //std.debug.print("\n  Performing identity operation (noop)", .{});
            // Act as identity operation
            @memcpy(output_tensor.data, input_tensor.data);
            return;
        }
        // Reduce over all dimensions
        //std.debug.print("\n  Reducing over all dimensions", .{});
        var sum: T = 0;
        for (input_tensor.data) |val| {
            sum += val;
        }
        output_tensor.data[0] = sum / @as(T, @floatFromInt(input_tensor.size));
        //std.debug.print("\n  Final result (all dims): {d}", .{output_tensor.data[0]});
        return;
    }

    // Validate axes
    if (axes.?.len == 0) return TensorMathError.InvalidAxes;
    for (axes.?) |axis| {
        const abs_axis = if (axis < 0)
            @as(i64, @intCast(input_tensor.shape.len)) + axis
        else
            axis;
        if (abs_axis < 0 or abs_axis >= input_tensor.shape.len) {
            //std.debug.print("\n  Error: Axis {d} out of bounds for tensor of rank {d}", .{ axis, input_tensor.shape.len });
            return TensorMathError.InvalidAxes;
        }
    }

    // Convert negative axes to positive
    var actual_axes = try pkg_allocator.alloc(usize, axes.?.len);
    defer pkg_allocator.free(actual_axes);

    //std.debug.print("\n  Converting axes to positive indices:", .{});
    for (axes.?, 0..) |axis, i| {
        actual_axes[i] = if (axis < 0)
            @intCast(@as(i64, @intCast(input_tensor.shape.len)) + axis)
        else
            @intCast(axis);
        //std.debug.print("\n    axis {d} -> {d}", .{ axis, actual_axes[i] });
    }

    // Calculate the size of dimensions being reduced
    var reduce_size: usize = 1;
    for (actual_axes) |axis| {
        reduce_size *= input_tensor.shape[axis];
    }
    //std.debug.print("\n  Total elements to reduce per output: {d}", .{reduce_size});

    // Initialize output values to 0
    @memset(output_tensor.data, 0);

    // Calculate strides for input tensor
    var input_strides = try pkg_allocator.alloc(usize, input_tensor.shape.len);
    defer pkg_allocator.free(input_strides);

    var stride: usize = 1;
    var i = input_tensor.shape.len;
    while (i > 0) {
        i -= 1;
        input_strides[i] = stride;
        stride *= input_tensor.shape[i];
    }

    // For each output element
    for (0..output_tensor.size) |out_idx| {
        //std.debug.print("\n\nProcessing output element {d}:", .{out_idx});

        // Calculate input indices for non-reduced dimensions
        var remaining = out_idx;
        var base_idx: usize = 0;

        // Calculate base index from non-reduced dimensions
        var non_reduced_dim: usize = 0;
        var output_dim: usize = 0;
        var remaining_dims = try pkg_allocator.alloc(usize, input_tensor.shape.len);
        defer pkg_allocator.free(remaining_dims);
        var num_remaining: usize = 0;

        // First pass: collect non-reduced dimensions in order
        for (0..input_tensor.shape.len) |dim_idx| {
            if (std.mem.indexOfScalar(usize, actual_axes, dim_idx) == null) {
                remaining_dims[num_remaining] = dim_idx;
                num_remaining += 1;
            }
        }

        // Calculate output strides
        var output_strides = try pkg_allocator.alloc(usize, num_remaining);
        defer pkg_allocator.free(output_strides);
        var out_stride: usize = 1;
        var stride_idx = num_remaining;
        while (stride_idx > 0) {
            stride_idx -= 1;
            output_strides[stride_idx] = out_stride;
            out_stride *= input_tensor.shape[remaining_dims[stride_idx]];
        }

        // Second pass: calculate base_idx
        for (0..num_remaining) |dim_idx| {
            const dim = remaining_dims[dim_idx];
            const dim_stride = input_strides[dim];
            const current_dim = remaining / output_strides[dim_idx];
            remaining %= output_strides[dim_idx];

            base_idx += current_dim * dim_stride;
            //std.debug.print("\n  Non-reduced dim {d}: current_dim={d}, stride={d}, base_idx={d}", .{ dim, current_dim, dim_stride, base_idx });
            non_reduced_dim += 1;
            output_dim += 1;
        }

        // Sum up all values in the reduction dimensions
        var sum: T = 0;
        const count = reduce_size;

        //std.debug.print("\n  Reduction info: count={d}, base_idx={d}", .{ count, base_idx });

        // Calculate the size and stride for each reduced dimension
        var reduced_sizes = try pkg_allocator.alloc(usize, actual_axes.len);
        defer pkg_allocator.free(reduced_sizes);
        var reduced_strides = try pkg_allocator.alloc(usize, actual_axes.len);
        defer pkg_allocator.free(reduced_strides);

        // Store sizes and strides in reverse order to match the memory layout
        for (actual_axes, 0..) |axis, axis_idx| {
            const rev_idx = actual_axes.len - 1 - axis_idx;
            reduced_sizes[rev_idx] = input_tensor.shape[axis];
            reduced_strides[rev_idx] = input_strides[axis];
            //std.debug.print("\n  Reduced axis {d}: size={d}, stride={d}", .{ axis, input_tensor.shape[axis], input_strides[axis] });
        }

        // Iterate over all combinations of reduced dimensions
        var idx: usize = 0;
        while (idx < count) : (idx += 1) {
            var temp_idx = base_idx;
            var temp = idx;

            // Process dimensions from innermost to outermost
            for (0..actual_axes.len) |dim_idx| {
                const axis_idx = temp % reduced_sizes[dim_idx];
                temp /= reduced_sizes[dim_idx];
                temp_idx += axis_idx * reduced_strides[dim_idx];
            }

            sum += input_tensor.data[temp_idx];
            //if (idx < 5 or idx > count - 5) {
            //std.debug.print("\n    idx={d}: temp_idx={d}, value={d}", .{ idx, temp_idx, input_tensor.data[temp_idx] });
            //}
        }

        //std.debug.print("\n  Final sum={d}, mean={d}", .{ sum, sum / @as(T, @floatFromInt(count)) });
        output_tensor.data[out_idx] = sum / @as(T, @floatFromInt(count));
    }
}
