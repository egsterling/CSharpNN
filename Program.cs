public static class GlobalFuncs {
    public static void print3d(double[,,] array) {
        for(int i = 0; i < array.GetLength(0); ++i) {
            Console.WriteLine("Layer " + i);
            for(int j = 0; j < array.GetLength(1); ++j) {
                for(int k = 0; k < array.GetLength(2); ++k) {
                    Console.Write(array[i,j,k] + " ");
                }
                Console.WriteLine();
            }
            Console.WriteLine();
        }
    }
}

// public static class Init {
//     public static void normal(double mean, double std) {

//     }

//     public static void uniform(double min, double max) {
//         Random r = new Random();
//         for(int l = 0; l < conv.Length; ++l) {
//             // foreach Layer
//             for(int c = 0; c < channels; ++c) {
//                 for(int h = 0; h < height; ++h) {
//                     for(int w = 0; w < width; ++w) {
//                         double d = (r.NextDouble() - 0.5) * 0.02;
//                         conv[l].getLayer[c,h,w] = d;
//                     }
//                 }
//             }
//         }
//     }
// }


public abstract class Init {
    protected Random r = new Random();

    protected abstract double operation(params double[] args);

    public void initialize(ref double[,,,] input, params double[] args) {
        int numLayers = input.GetLength(0);
        int channels = input.GetLength(1);
        int height = input.GetLength(2);
        int width = input.GetLength(3);
        for(int l = 0; l < numLayers; ++l) {
            for(int c = 0; c < channels; ++c) {
                for(int h = 0; h < height; ++h) {
                    for(int w = 0; w < width; ++w) {
                        input[l,c,h,w] = operation(args);
                    }
                }
            }
        }
    }

    public void initialize(ref double[,] input, params double[] args) {
        int inNeurons = input.GetLength(0);
        int outNeurons = input.GetLength(1);
        for(int i = 0; i < inNeurons; ++i) {
            for(int j = 0; j < outNeurons; ++j) {
                input[i,j] = operation(args);
            }
        }
    }

    public void initialize(ref double[] input, params double[] args) {
        int size = input.Length;
        for(int i = 0; i < size; ++i) {
            input[i] = operation(args);
        }
    }
}

public class Uniform : Init {
    protected override double operation(params double[] args) {
        // args[0]: min
        // args[1]: max    (0, 1]
        return (1-r.NextDouble()) * (args[1] - args[0]) + args[0];
    }
}

public class Gaussian : Init {
    // from stackoverflow
    protected override double operation(params double[] args) {
        // args[0]: mean
        // args[1]: std
        double u1 = 1.0-r.NextDouble(); //uniform(0,1] random doubles
        double u2 = 1.0-r.NextDouble();
        double randStdNormal = Math.Sqrt(-2.0 * Math.Log(u1)) *
                    Math.Sin(2.0 * Math.PI * u2); //random normal(0,1)
        return args[0] + args[1] * randStdNormal; //random normal(mean,stdDev^2)
    }
}

public class Constant : Init {
    protected override double operation(params double[] args) {
        return args[0];
    }
}

public abstract class CNNLayer {
    protected readonly int height;
    protected readonly int width;
    protected readonly (int, int) stride;
    protected readonly (int, int) padding;

    public CNNLayer(int H_in, int W_in)
        : this(H_in, W_in, (1, 1), (0, 0)) {}

    public CNNLayer(int H_in, int W_in, (int, int) stride_in, (int, int) padding_in) {
        height = H_in;
        width = W_in;
        stride = stride_in;
        padding = padding_in;
    }

    protected int getDim(int inputDim, int filterDim, int strideDim) {
        return ((inputDim - filterDim) / strideDim) + 1;
    }

    protected double[,,] padded(double[,,] inp) {
        if(padding == (0, 0)) {
            return inp;
        }
        int channels = inp.GetLength(0);
        int inpHeight = inp.GetLength(1);
        int inpWidth = inp.GetLength(2);
        var (vertPad, horPad) = padding;
        double[,,] padded = new double[channels, inpHeight + 2*vertPad, inpWidth + 2*horPad];
        int paddedHeight = padded.GetLength(1);
        int paddedWidth = padded.GetLength(2);
        for(int c = 0; c < channels; ++c) {
            for(int h = 0; h < paddedHeight; ++h) {
                for(int w = 0; w < paddedWidth; ++w) {
                    if(h < vertPad || h >= paddedHeight - vertPad || w < horPad || w >= paddedWidth - horPad) {
                        padded[c,h,w] = 0;
                    }
                    else {
                        padded[c,h,w] = inp[c,h-vertPad,w-horPad];
                    }
                }
            }
        }
        return padded;
    }
    
}


public class ConvolutionalLayer : CNNLayer {
    // Layer[] conv;
    double[,,,] conv;
    double[] bias;
    readonly int channels;
    readonly int numLayers;

    // Default stride: (1, 1) Default padding: 0
    public ConvolutionalLayer(int in_channels, int H_in, int W_in, int out_channels)
        : this(in_channels, H_in, W_in, out_channels, (1, 1), (0, 0)) {}

    public ConvolutionalLayer(int in_channels, int H_in, int W_in, int out_channels, (int, int) stride_in, (int, int) padding_in) 
        : base(H_in, W_in, stride_in, padding_in) {
            // conv = new Layer[out_channels];
            conv = new double[out_channels, in_channels, H_in, W_in];
            bias = new double[out_channels];
            channels = in_channels;
            numLayers = out_channels;
            for(int i = 0; i < out_channels; ++i) {
                // this.conv[i] = new Layer(in_channels, H_in, W_in);
            }
        }

    double innerProduct(int layerInd, double[,,] input, int hInd, int wInd) {
        double total = 0;
        for(int c = 0; c < channels; ++c) {
            for(int h = 0; h < height; ++h) {
                for(int w = 0; w < width; ++w) {
                    total += conv[layerInd,c,h,w] * input[c,h+hInd,w+wInd];
                }
            }
        }
        return total + bias[layerInd];
    }

    public double[,,] convolute(double[,,] input) {
        int inpChannels = input.GetLength(0);
        if(inpChannels != channels) {
            throw new ArgumentException(String.Format("in_channels and input channels don't match; {0} in_channels and {1} input channels", 
                                        channels, inpChannels));
        }
        input = padded(input);
        int inpHeight = input.GetLength(1);
        int inpWidth = input.GetLength(2);
        var (vertStride, horStride) = stride;
        if((inpHeight - height) % vertStride != 0 || (inpWidth - width) % horStride != 0) {
            throw new ArgumentException("stride length does not work with input size");
        }
        int outputHeight = getDim(inpHeight, height, vertStride);
        int outputWidth = getDim(inpWidth, width, horStride);
        double[,,] output = new double[numLayers, outputHeight, outputWidth];
        for(int o = 0; o < numLayers; ++o) {
            for(int h = 0; h < outputHeight; ++h) {
                for(int w = 0; w < outputWidth; ++w) {
                    output[o,h,w] = innerProduct(o, input, h*vertStride, w*horStride);
                }
            }
        }
        return output;
    }
    
    public void printLayers() {
        for(int l = 0; l < numLayers; ++l) {
            Console.WriteLine("Layer " + l);
            for(int h = 0; h < height; ++h) {
                for(int c = 0; c < channels; ++c) {
                    for(int w = 0; w < width; ++w) {
                        Console.Write(Math.Round(conv[l,c,h,w], 8) + " ");
                    }
                    Console.Write("    ");
                }
                Console.WriteLine();
            }
            Console.WriteLine();
        } 
    }
}

public abstract class PoolingLayer : CNNLayer {
    public PoolingLayer(int H_in, int W_in)
        : this(H_in, W_in, (1, 1), (0, 0)) {}

    public PoolingLayer(int H_in, int W_in, (int, int) stride_in, (int, int) padding_in) 
        : base(H_in, W_in, stride_in, padding_in) {}

    protected abstract double poolOperation(double[,,] input, int cInd, int hInd, int wInd);

    public double[,,] Pool(double[,,] input) {
        input = padded(input);
        int inpChannels = input.GetLength(0);
        int inpHeight = input.GetLength(1);
        int inpWidth = input.GetLength(2);
        var (vertStride, horStride) = stride;
        if((inpHeight - height) % vertStride != 0 || (inpWidth - width) % horStride != 0) {
            throw new ArgumentException("stride length does not work with input size");
        }
        int outputHeight = getDim(inpHeight, height, vertStride);
        int outputWidth = getDim(inpWidth, width, horStride);
        double[,,] output = new double[inpChannels, outputHeight, outputWidth];
        for(int c = 0; c < inpChannels; ++c) {
            for(int h = 0; h < outputHeight; ++h) {
                for(int w = 0; w < outputWidth; ++w) {
                    output[c,h,w] = poolOperation(input, c, h*vertStride, w*horStride);
                }
            }
        }
        return output;
    }
}

public class MaxPool : PoolingLayer {
    public MaxPool(int H_in, int W_in)
        : this(H_in, W_in, (1, 1), (0, 0)) {}

    public MaxPool(int H_in, int W_in, (int, int) stride_in, (int, int) padding_in) 
        : base(H_in, W_in, stride_in, padding_in) {}

    protected override double poolOperation(double[,,] input, int cInd, int hInd, int wInd) {
        double maxim = input[cInd, hInd, wInd];
        for(int i = hInd; i < hInd + height; ++i) {
            for(int j = wInd; j < wInd + width; ++j) {
                maxim = Math.Max(maxim, input[cInd,i,j]);
            }
        }
        return maxim;
    }
}

public class AveragePool : PoolingLayer {
    public AveragePool(int H_in, int W_in)
        : this(H_in, W_in, (1, 1), (0, 0)) {}

    public AveragePool(int H_in, int W_in, (int, int) stride_in, (int, int) padding_in) 
        : base(H_in, W_in, stride_in, padding_in) {}

    protected override double poolOperation(double[,,] input, int cInd, int hInd, int wInd) {
        double total = 0;
        for(int i = hInd; i < hInd + height; ++i) {
            for(int j = wInd; j < wInd + width; ++j) {
                total += input[cInd,i,j];
            }
        }
        return total / (height * width);
    }
}

public class FullyConnected {
    readonly int in_features;
    readonly int out_features;
    // weight[a,b] --> weight from neuron a to neuron b
    double[,] weights;
    double[] biases;

    public FullyConnected(int num_in_features, int num_out_features) {
        in_features = num_in_features;
        out_features = num_out_features;
        weights = new double[num_in_features, num_out_features];
        biases = new double[num_out_features];
    }

    public double[] propagate(double[] input) {
        double[] output = new double[out_features];
        for(int o = 0; o < out_features; ++o) {
            double total = 0;
            for(int i = 0; i < in_features; ++i) {
                total += input[i] * weights[i,o];
            }
            output[o] = total + biases[o];
        }
        return output;
    }
}

public abstract class ActivationFunc {
    protected abstract double activation(double d);

    public double[] apply(double[] input) {
        double[] output = new double[input.Length];
        for(int i = 0; i < input.Length; ++i) {
            output[i] = activation(input[i]);
        }
        return output;
    }

    public double[,,] apply(double[,,] input) {
        int channels = input.GetLength(0);
        int height = input.GetLength(1);
        int width = input.GetLength(2);
        double[,,] output = new double[channels, height, width];
        for(int c = 0; c < channels; ++c) {
            for(int h = 0; h < height; ++h) {
                for(int w = 0; w < width; ++w) {
                    output[c,h,w] = activation(input[c,h,w]);
                }
            }
        }
        return output;
    }
}

public class ReLU : ActivationFunc {
    protected override double activation(double d) {
        return Math.Max(0, d);
    }
}

public class Sigmoid : ActivationFunc {
    protected override double activation(double d) {
        return 1 / (1 + Math.Exp(-d));
    }
}


public class Run {
    static void Main() {
        // double[,,] inp = {{{1, 2, 3, 4}, {1, 2, 3, 4}, {1, 2, 3, 4}, {1, 2, 3, 4}},
        //                   {{1, 2, 3, 4}, {1, 2, 3, 4}, {1, 2, 3, 4}, {1, 2, 3, 4}},};
        double[,,] inp = {{{1, 2}, {2, 1}}, {{1, 2}, {2, 1}}};
        // GlobalFuncs.print3d(inp);
        var cn = new ConvolutionalLayer(2, 3, 3, 2, (1, 1), (1, 1));
        // GlobalFuncs.print3d(cn.padded(inp));

        // cn.uniform(-1, 1);
        // cn.printLayers();
        GlobalFuncs.print3d(inp);
        var j = cn.convolute(inp);
        GlobalFuncs.print3d(inp);
        // GlobalFuncs.print3d(inp);
        // var pool = new MaxPool(2, 2, (2, 2), (1, 1));
        // var pj = pool.Pool(inp);
        // GlobalFuncs.print3d(pj);

    }
}