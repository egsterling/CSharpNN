public class Layer {
    double[,,] layer;
    double bias;

    public Layer(int in_channels, int H_in, int W_in) {
        this.layer = new double[in_channels, H_in, W_in];
    }

    public double getBias {
        get => bias;
        set => bias = value;
    }

    public double[,,] getLayer {
        get => layer;
    }
}

public class ConvolutionalLayer {
    Layer[] conv;
    readonly int channels;
    readonly int height;
    readonly int width;
    readonly (int, int) stride;
    readonly (int, int) padding;

    // Default stride: (1, 1) Default padding: 0
    public ConvolutionalLayer(int in_channels, int H_in, int W_in, int out_channels)
        : this(in_channels, H_in, W_in, out_channels, (1, 1), (0, 0)) {}

    public ConvolutionalLayer(int in_channels, int H_in, int W_in, int out_channels, (int, int) stride_in, (int, int) padding_in) {
        this.conv = new Layer[out_channels];
        this.channels = in_channels; this.height = H_in; this.width = W_in;
        for(int i = 0; i < out_channels; ++i) {
            this.conv[i] = new Layer(H_in, W_in, in_channels);
        }
        this.stride = stride_in;
        this.padding = padding_in;
    }

    public double[,,] padded(double[,,] inp) {
        if(padding == (0, 0)) {
            return inp;
        }
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

    public double innerProduct(Layer convLayer, double[,,] section) {
        double[,,] layerVals = convLayer.getLayer;
        double total = 0;
        for(int c = 0; c < channels; ++c) {
            for(int w = 0; w < width; ++w) {
                for(int h = 0; h < height; ++h) {
                    total += layerVals[c,w,h] * section[c,w,h];
                }
            }
        }
        return total + convLayer.getBias;
    }


    public double[,] convolute(double[,,] input) {
        int inpChannels = input.GetLength(0);
        if(inpChannels != channels) {
            throw new ArgumentException(String.Format("in_channels and input channels don't match; {0} in_channels and {1} input channels", 
                                        channels, inpChannels));
        }
        input = padded(input);
        int inpHeight = input.GetLength(1);
        int inpWidth = input.GetLength(2);
        // if(check exceptions, like check if stride length works with padded input size, maybe otherget)
        

    }

}

public class PoolingLayer {
    

}

public class FullyConnected {

}

public class Run {
    static void Main() {
        double[,,] inp = {{{1, 2}, {1, 2}},
                          {{1, 2}, {1, 2}},};
        var cn = new ConvolutionalLayer(2, 2, 2, 2, (1, 1), (2, 1));
        var pp = cn.padded(inp);
        for(int i = 0; i < pp.GetLength(0); ++i) {
            Console.WriteLine("Layer " + i);
            for(int j = 0; j < pp.GetLength(1); ++j) {
                for(int k = 0; k < pp.GetLength(2); ++k) {
                    Console.Write(pp[i,j,k] + " ");
                }
                Console.WriteLine();
            }
            Console.WriteLine();
        }
    }
}