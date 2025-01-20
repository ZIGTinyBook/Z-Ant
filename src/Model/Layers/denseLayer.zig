const std = @import("std");
const tensor = @import("tensor");
const TensMath = @import("tensor_m");
const Layer = @import("Layer");
const Architectures = @import("architectures").Architectures;
const LayerError = @import("errorHandler").LayerError;
const Codegen = @import("codegen");

/// Function to create a DenseLayer struct in future it will be possible to create other types of layers like convolutional, LSTM etc.
/// The DenseLayer is a fully connected layer, it has a weight matrix and a bias vector.
/// It has also an activation function that can be applied to the output, it can even be none.
pub fn DenseLayer(comptime T: type) type {
    return struct {
        //          | w11   w12  w13 |
        // weight = | w21   w22  w23 | , where Wij, i= neuron i-th and j=input j-th
        //          | w31   w32  w33 |
        weights: tensor.Tensor(T), //each row represent a neuron, where each weight is associated to an input
        bias: tensor.Tensor(T), //a bias for each neuron
        input: tensor.Tensor(T), //is saved for semplicity, it can be sobstituted
        output: tensor.Tensor(T), // output = dot(input, weight.transposed) + bias
        //layer shape --------------------
        n_inputs: usize,
        n_neurons: usize,
        //gradients-----------------------
        w_gradients: tensor.Tensor(T),
        b_gradients: tensor.Tensor(T),
        //utils---------------------------
        allocator: *const std.mem.Allocator,

        const Self = @This();

        pub fn create(self: *Self) Layer.Layer(T) {
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
                    .codegen = codegen,
                },
            };
        }

        ///Initilize the layer with random weights and biases
        /// also for the gradients
        pub fn init(ctx: *anyopaque, alloc: *const std.mem.Allocator, args: *anyopaque) !void {
            const self: *Self = @ptrCast(@alignCast(ctx));
            const argsStruct: *const struct { n_inputs: usize, n_neurons: usize } = @ptrCast(@alignCast(args));
            const n_inputs = argsStruct.n_inputs;
            const n_neurons = argsStruct.n_neurons;

            std.log.debug("\nInit DenseLayer: n_inputs = {}, n_neurons = {}, Type = {}", .{ n_inputs, n_neurons, @TypeOf(T) });

            //check on parameters
            if (n_inputs <= 0 or n_neurons <= 0) return LayerError.InvalidParameters;

            //initializing number of neurons and inputs----------------------------------
            self.n_inputs = n_inputs;
            self.n_neurons = n_neurons;

            var weight_shape: [2]usize = [_]usize{ n_inputs, n_neurons };
            var bias_shape: [1]usize = [_]usize{n_neurons};
            self.allocator = alloc;

            //std.log.debug("Generating random weights...\n", .{});
            const weight_matrix = try Layer.randn(T, self.allocator, n_inputs, n_neurons);
            defer self.allocator.free(weight_matrix);
            const bias_matrix = try Layer.randn(T, self.allocator, 1, n_neurons);
            defer self.allocator.free(bias_matrix);

            //initializing weights and biases--------------------------------------------
            self.weights = try tensor.Tensor(T).fromArray(alloc, weight_matrix, &weight_shape);
            self.bias = try tensor.Tensor(T).fromArray(alloc, bias_matrix, &bias_shape);

            //initializing gradients to all zeros----------------------------------------
            self.w_gradients = try tensor.Tensor(T).fromShape(self.allocator, &weight_shape);
            self.b_gradients = try tensor.Tensor(T).fromShape(self.allocator, &bias_shape);
        }

        ///Deallocate the layer
        pub fn deinit(ctx: *anyopaque) void {
            const self: *Self = @ptrCast(@alignCast(ctx));

            //std.log.debug("Deallocating DenseLayer resources...\n", .{});

            // Dealloc tensors of weights, bias and output if allocated
            if (self.weights.data.len > 0) {
                self.weights.deinit();
            }

            if (self.bias.data.len > 0) {
                self.bias.deinit();
            }

            if (self.output.data.len > 0) {
                self.output.deinit();
            }

            if (self.w_gradients.data.len > 0) {
                self.w_gradients.deinit();
            }

            if (self.b_gradients.data.len > 0) {
                self.b_gradients.deinit();
            }

            if (self.input.data.len > 0) {
                self.input.deinit();
            }

            std.log.debug("\nDenseLayer resources deallocated.", .{});
        }

        ///Forward pass of the layer if present it applies the activation function
        /// We can improve it removing as much as possibile all the copy operations
        pub fn forward(ctx: *anyopaque, input: *tensor.Tensor(T)) !tensor.Tensor(T) {
            const self: *Self = @ptrCast(@alignCast(ctx));

            // Dealloca self.input se già allocato
            if (self.input.data.len > 0) {
                self.input.deinit();
            }
            self.input = try input.copy();

            // Dealloca self.output se già allocato prima di riassegnarlo
            if (self.output.data.len > 0) {
                self.output.deinit();
            }

            self.output = try TensMath.compute_dot_product(T, self.allocator, &self.input, &self.weights);
            try TensMath.add_bias(T, self.allocator, &self.output, &self.bias);

            return self.output;
        }

        /// Backward pass of the layer It takes the dValues from the next layer and computes the gradients
        pub fn backward(ctx: *anyopaque, dValues: *tensor.Tensor(T)) !tensor.Tensor(T) {
            const self: *Self = @ptrCast(@alignCast(ctx));

            //---- Key Steps: -----
            // 2. Compute weight gradients (w_gradients)
            var input_transposed = try self.input.transpose2D();

            defer input_transposed.deinit();

            self.w_gradients.deinit();
            self.w_gradients = try TensMath.dot_product_tensor(Architectures.CPU, T, T, self.allocator, &input_transposed, dValues);
            // 3. Compute bias gradients (b_gradients)
            // Equivalent of np.sum(dL_dOutput, axis=0, keepdims=True)
            var sum: T = 0;
            //summing all the values in each column(neuron) of dValue and putting it into b_gradint[neuron]
            for (0..dValues.shape[1]) |neuron| {
                //scanning all the inputs
                sum = 0;
                for (0..dValues.shape[0]) |input| {
                    sum += dValues.data[input * self.n_neurons + neuron];
                }
                self.b_gradients.data[neuron] = sum;
            }

            var weights_transposed = try self.weights.transpose2D();
            defer weights_transposed.deinit();

            var dL_dInput: tensor.Tensor(T) = try TensMath.dot_product_tensor(Architectures.CPU, T, T, self.allocator, dValues, &weights_transposed);
            _ = &dL_dInput;
            return dL_dInput;
        }

        ///Print the layer used for debug purposes it has 2 different verbosity levels
        pub fn printLayer(ctx: *anyopaque, choice: u8) void {
            const self: *Self = @ptrCast(@alignCast(ctx));

            std.log.debug("\n ************************Dense layer*********************", .{});
            //MENU choice:
            // 0 -> full details layer
            // 1 -> shape schema
            if (choice == 0) {
                std.log.debug("\n neurons:{}  inputs:{}", .{ self.n_neurons, self.n_inputs });
                std.log.debug("\n \n************input", .{});
                self.input.printMultidim();

                std.log.debug("\n \n************weights", .{});
                self.weights.printMultidim();
                std.log.debug("\n \n************bias", .{});
                std.log.debug("\n {any}", .{self.bias.data});
                std.log.debug("\n \n************output", .{});
                self.output.printMultidim();
                std.log.debug("\n \n************w_gradients", .{});
                self.w_gradients.printMultidim();
                std.log.debug("\n \n************b_gradients", .{});
                std.log.debug("\n {any}", .{self.b_gradients.data});
            }
            if (choice == 1) {
                std.log.debug("\n   input       weight   bias  output", .{});
                std.log.debug("\n [{} x {}] * [{} x {}] + {} = [{} x {}] ", .{
                    self.input.shape[0],
                    self.input.shape[1],
                    self.weights.shape[0],
                    self.weights.shape[1],
                    self.bias.shape[0],
                    self.output.shape[0],
                    self.output.shape[1],
                });
                std.log.debug("\n ", .{});
            }
        }

        //---------------------------------------------------------------
        //----------------------------getters----------------------------
        //---------------------------------------------------------------
        pub fn get_n_inputs(ctx: *anyopaque) usize {
            const self: *Self = @ptrCast(@alignCast(ctx));

            return self.n_inputs;
        }

        pub fn get_n_neurons(ctx: *anyopaque) usize {
            const self: *Self = @ptrCast(@alignCast(ctx));

            return self.n_neurons;
        }

        pub fn get_weights(ctx: *anyopaque) *const tensor.Tensor(T) {
            const self: *Self = @ptrCast(@alignCast(ctx));

            return &self.weights;
        }

        pub fn get_bias(ctx: *anyopaque) *const tensor.Tensor(T) {
            const self: *Self = @ptrCast(@alignCast(ctx));

            return &self.bias;
        }

        pub fn get_input(ctx: *anyopaque) *const tensor.Tensor(T) {
            const self: *Self = @ptrCast(@alignCast(ctx));

            return &self.input;
        }

        pub fn get_output(ctx: *anyopaque) *tensor.Tensor(T) {
            const self: *Self = @ptrCast(@alignCast(ctx));

            return &self.output;
        }

        pub fn get_weightGradients(ctx: *anyopaque) *const tensor.Tensor(T) {
            const self: *Self = @ptrCast(@alignCast(ctx));

            return &self.w_gradients;
        }

        pub fn get_biasGradients(ctx: *anyopaque) *const tensor.Tensor(T) {
            const self: *Self = @ptrCast(@alignCast(ctx));

            return &self.b_gradients;
        }

        pub fn codegen(ctx: *anyopaque, writer: std.fs.File.Writer) !void {
            const self: *Self = @ptrCast(@alignCast(ctx));

            try writer.print("const {s} = blk: {{\n", .{"test"});
            try Codegen.writeAllocator(writer, "allocator", self.n_inputs * self.n_neurons);
            try writer.print(
                \\const input_ptr: [*]const f64 = @ptrCast(input);
                \\break :blk try tensor.Tensor(f64).fromArray(allocator, input_ptr, &.{{1, {}}});
            , .{self.n_inputs});
            _ = try writer.write("};\n");

            _ = try writer.write(
                \\input_tensor.print();
                \\
            );

            unreachable;
        }
    };
}
