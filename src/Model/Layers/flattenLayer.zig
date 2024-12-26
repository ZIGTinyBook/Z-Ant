const std = @import("std");
const Tensor = @import("Tensor");
const TensMath = @import("tensor_m");
const Layer = @import("Layer");
const Architectures = @import("architectures").Architectures;
const LayerError = @import("errorHandler").LayerError;

pub fn FlattenLayer(comptime T: type) type {
    return struct {
        // Flatten layer parameters
        input: Tensor.Tensor(T), // Stored input for backward pass
        output: Tensor.Tensor(T), // Flattened output
        allocator: *const std.mem.Allocator,

        const Self = @This();

        // Placeholder struct for init arguments
        pub const FlattenInitArgs = struct {
            placeholder: bool,
        };

        pub fn create(self: *Self) Layer.Layer(T) {
            return Layer.Layer(T){
                .layer_type = Layer.LayerType.FlattenLayer,
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

        /// Initialize the Flatten layer (just store allocator, no weights)
        pub fn init(ctx: *anyopaque, alloc: *const std.mem.Allocator, args: *anyopaque) !void {
            const self: *Self = @ptrCast(@alignCast(ctx));
            const argsStruct: *const FlattenInitArgs = @ptrCast(@alignCast(args));
            _ = argsStruct; // We don't really need the placeholder

            self.allocator = alloc;

            return; // No errors expected
        }

        /// Deallocate the Flatten layer resources
        pub fn deinit(ctx: *anyopaque) void {
            const self: *Self = @ptrCast(@alignCast(ctx));

            if (self.input.data.len > 0) {
                self.input.deinit();
            }

            if (self.output.data.len > 0) {
                self.output.deinit();
            }
        }

        /// Forward pass: Flatten all dimensions except the first (batch) dimension
        /// Input: [N, D1, D2, ..., Dk]
        /// Output: [N, D1*D2*...*Dk]
        pub fn forward(ctx: *anyopaque, input: *Tensor.Tensor(T)) !Tensor.Tensor(T) {
            const self: *Self = @ptrCast(@alignCast(ctx));

            if (input.shape.len < 2) {
                return LayerError.InvalidParameters;
            }

            // Save the input for backward pass
            if (self.input.data.len > 0) {
                self.input.deinit();
            }
            self.input = try input.copy();

            const batch_size = input.shape[0];

            // Compute total size of the rest dimensions
            var total_size: usize = 1;
            for (input.shape[1..]) |dim| {
                total_size *= dim;
            }

            // New shape: [N, total_size]
            var output_shape: [2]usize = .{ batch_size, total_size };

            if (self.output.data.len > 0) {
                self.output.deinit();
            }
            self.output = try Tensor.Tensor(T).fromArray(self.allocator, input.data, output_shape[0..]);

            return self.output;
        }

        /// Backward pass: Reshape the gradients to the original input shape
        /// If forward input shape = [N, D1, D2, ..., Dk]
        /// backward receives dValues of shape [N, D1*D2*...*Dk]
        /// We must reshape back to [N, D1, D2, ..., Dk].
        pub fn backward(ctx: *anyopaque, dValues: *Tensor.Tensor(T)) !Tensor.Tensor(T) {
            const self: *Self = @ptrCast(@alignCast(ctx));

            const input_shape_const = self.input.shape;

            // Just reshape the dValues to original input shape
            var input_shape = try self.allocator.alloc(usize, input_shape_const.len);
            _ = &input_shape;
            defer self.allocator.free(input_shape);

            // Copy shape
            @memcpy(input_shape, input_shape_const);

            var dInput = try Tensor.Tensor(T).fromArray(self.allocator, dValues.data, input_shape);
            _ = &dInput; // Unused
            return dInput;
        }

        /// Print the flatten layer information
        pub fn printLayer(ctx: *anyopaque, choice: u8) void {
            const self: *Self = @ptrCast(@alignCast(ctx));
            switch (choice) {
                0 => std.log.debug("Flatten Layer\n", .{}),
                1 => std.log.debug("Input shape: {any}, Output shape: {any}\n", .{ self.input.shape, self.output.shape }),
                else => {},
            }
        }

        //---------------------------------------------------------------
        //---------------------------- Getters --------------------------
        //---------------------------------------------------------------
        /// Get the number of inputs = product of all dimensions except the first is combined into one
        pub fn get_n_inputs(ctx: *anyopaque) usize {
            const self: *Self = @ptrCast(@alignCast(ctx));

            if (self.input.shape.len < 2) return 0;

            var total: usize = 1;
            for (self.input.shape[1..]) |dim| {
                total *= dim;
            }
            return total;
        }

        /// For Flatten layer, number of neurons = number of inputs after the first dimension
        pub fn get_n_neurons(ctx: *anyopaque) usize {
            return get_n_inputs(ctx);
        }

        pub fn get_input(ctx: *anyopaque) *const Tensor.Tensor(T) {
            const self: *Self = @ptrCast(@alignCast(ctx));
            return &self.input;
        }

        pub fn get_output(ctx: *anyopaque) *Tensor.Tensor(T) {
            const self: *Self = @ptrCast(@alignCast(ctx));
            return &self.output;
        }
    };
}
