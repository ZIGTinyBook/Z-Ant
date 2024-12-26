const std = @import("std");
const tensor = @import("Tensor");
const TensMath = @import("tensor_m");
const Layer = @import("Layer");
const Architectures = @import("architectures").Architectures;
const LayerError = @import("errorHandler").LayerError;
const TensorError = @import("errorHandler").TensorError;

pub const PoolingType = enum {
    Max,
    Min,
    Avg,
};

/// TODO: implement padding
pub fn PoolingLayer(comptime T: type) type {
    return struct {
        input: tensor.Tensor(T),
        output: tensor.Tensor(T),
        used_input: tensor.Tensor(u8),
        kernel: [2]usize, // [rows, cols]
        stride: [2]usize, // [rows, cols]
        poolingType: PoolingType,
        allocator: *const std.mem.Allocator,

        const Self = @This();

        pub fn create(self: *Self) !Layer.Layer(T) {
            return Layer.Layer(T){
                .layer_type = Layer.LayerType.DenseLayer,
                .layer_ptr = self,
                .layer_impl = &.{
                    .init = init,
                    .deinit = deinit,
                    .forward = forward,
                    .backward = backward,
                    .printLayer = printLayer,
                    .get_n_inputs = get_n_inputs,
                    .get_n_neurons = get_n_neurons,
                    .get_input = get_input,
                    .get_output = get_output,
                },
            };
        }

        /// Initialize the layer
        pub fn init(ctx: *anyopaque, alloc: *const std.mem.Allocator, args: *anyopaque) !void {
            const self: *Self = @ptrCast(@alignCast(ctx));
            const argsStruct: *const struct { kernel: [2]usize, stride: [2]usize, poolingType: PoolingType } = @ptrCast(@alignCast(args));

            self.allocator = alloc;

            // Assign kernel and stride arrays directly
            self.kernel = argsStruct.kernel;
            self.stride = argsStruct.stride;
            self.poolingType = argsStruct.poolingType;

            std.log.debug("\nInit Pooling Layer", .{});

            // Only 2D supported
            if (self.kernel.len != 2 or self.stride.len != 2) {
                return LayerError.Only2DSupported;
            }

            // Check Kernel != 0
            for (self.kernel) |val| {
                if (val <= 0) return LayerError.ZeroValueKernel;
            }

            // Check stride != 0
            for (self.stride) |val| {
                if (val <= 0) return LayerError.ZeroValueStride;
            }
        }

        /// Deallocate the layer
        pub fn deinit(ctx: *anyopaque) void {
            const self: *Self = @ptrCast(@alignCast(ctx));

            if (self.output.data.len > 0) {
                self.output.deinit();
            }

            if (self.input.data.len > 0) {
                self.input.deinit();
            }

            if (self.used_input.data.len > 0) {
                self.used_input.deinit();
            }

            std.log.debug("\nPooling layer resources deallocated.", .{});
        }

        /// Forward pass of the pooling layer
        pub fn forward(ctx: *anyopaque, input: *tensor.Tensor(T)) !tensor.Tensor(T) {
            const self: *Self = @ptrCast(@alignCast(ctx));

            if (self.input.data.len > 0) {
                self.input.deinit();
            }
            self.input = try input.copy();

            const input_rows = self.input.shape[0];
            const input_cols = self.input.shape[1];

            const out_rows = (input_rows - self.kernel[0] + 1) / self.stride[0];
            const out_cols = (input_cols - self.kernel[1] + 1) / self.stride[1];
            const W = out_rows * out_cols;

            var used_windows_shape = [_]usize{ W, input_rows, input_cols };
            if (self.used_input.data.len > 0) {
                self.used_input.deinit();
            }
            self.used_input = try tensor.Tensor(u8).fromShape(self.allocator, used_windows_shape[0..]);

            for (self.used_input.data) |*v| v.* = 0;

            self.output = try TensMath.pool_tensor(T, self.allocator, &self.input, &self.used_input, &self.kernel, &self.stride, self.poolingType);

            return self.output;
        }

        /// Backward pass of the pooling layer
        pub fn backward(ctx: *anyopaque, dValues: *tensor.Tensor(T)) !tensor.Tensor(T) {
            const self: *Self = @ptrCast(@alignCast(ctx));

            var dInput = try tensor.Tensor(T).fromShape(self.allocator, self.input.shape);

            for (dInput.data) |*val| val.* = 0;

            const input_rows = self.input.shape[0];
            const input_cols = self.input.shape[1];

            const output_rows = self.output.shape[0];
            const output_cols = self.output.shape[1];

            //const W = output_rows * output_cols; // window num

            // used window shape: [W, input_rows, input_cols]
            // used_windows.data[w * input_rows * input_cols + r * input_cols + c]

            for (0..output_rows) |out_r| {
                for (0..output_cols) |out_c| {
                    const grad = dValues.data[out_r * output_cols + out_c];
                    const r_start = out_r * self.stride[0];
                    const c_start = out_c * self.stride[1];

                    const w = out_r * output_cols + out_c; // current window

                    switch (self.poolingType) {
                        .Max, .Min => {
                            for (0..self.kernel[0]) |kr| {
                                for (0..self.kernel[1]) |kc| {
                                    const in_r = r_start + kr;
                                    const in_c = c_start + kc;
                                    if (in_r < input_rows and in_c < input_cols) {
                                        const mask_val = self.used_input.data[w * (input_rows * input_cols) + in_r * input_cols + in_c];
                                        if (mask_val == 1) {
                                            dInput.data[in_r * input_cols + in_c] += grad;
                                        }
                                    }
                                }
                            }
                        },

                        .Avg => {
                            const kernel_area = self.kernel[0] * self.kernel[1];
                            const distributed_grad = grad / @as(T, @floatFromInt(kernel_area));
                            for (0..self.kernel[0]) |kr| {
                                for (0..self.kernel[1]) |kc| {
                                    const in_r = r_start + kr;
                                    const in_c = c_start + kc;
                                    if (in_r < input_rows and in_c < input_cols) {
                                        dInput.data[in_r * input_cols + in_c] += distributed_grad;
                                    }
                                }
                            }
                        },
                    }
                }
            }

            return dInput;
        }

        /// Print the layer (debug)
        pub fn printLayer(ctx: *anyopaque, choice: u8) void {
            const self: *Self = @ptrCast(@alignCast(ctx));

            std.log.debug("\n ************************Pooling layer*********************", .{});
            std.log.debug("\n kernel: {any}  stride:{any}", .{ self.kernel, self.stride });
            if (choice == 0) {
                std.log.debug("\n \n************input", .{});
                self.input.printMultidim();
                std.log.debug("\n \n************output", .{});
                self.output.printMultidim();
            }
            if (choice == 1) {
                std.log.debug("\n   input: [", .{});
                for (0..self.input.shape.len) |i| {
                    std.log.debug("{}", .{self.input.shape[i]});
                    if (i == self.input.shape.len - 1) {
                        std.log.debug("]", .{});
                    } else {
                        std.log.debug(" x ", .{});
                    }
                }
                std.log.debug("\n   output: [", .{});
                for (0..self.output.shape.len) |i| {
                    std.log.debug("{}", .{self.output.shape[i]});
                    if (i == self.output.shape.len - 1) {
                        std.log.debug("]", .{});
                    } else {
                        std.log.debug(" x ", .{});
                    }
                }
                std.log.debug("\n ", .{});
            }
        }

        //---------------------------------------------------------------
        //----------------------------getters----------------------------
        //---------------------------------------------------------------
        pub fn get_n_inputs(ctx: *anyopaque) usize {
            // Return a dummy value since not supported
            _ = ctx;
            return 0;
        }

        pub fn get_n_neurons(ctx: *anyopaque) usize {
            // Return a dummy value since not supported
            _ = ctx;
            return 0;
        }

        pub fn get_input(ctx: *anyopaque) *const tensor.Tensor(T) {
            const self: *Self = @ptrCast(@alignCast(ctx));
            return &self.input;
        }

        pub fn get_output(ctx: *anyopaque) *tensor.Tensor(T) {
            const self: *Self = @ptrCast(@alignCast(ctx));
            return &self.output;
        }
    };
}
