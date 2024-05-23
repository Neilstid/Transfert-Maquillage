import torch


class CNNAlphaBlendParams(torch.nn.ModuleDict):
    def __init__(self):
        super().__init__()
        self.add_module(
            "input_transformation_src",
            torch.nn.Sequential(
                # Input = 3 x 256 x 256, Output = 32 x 256 x 256
                torch.nn.Conv2d(in_channels = 3, out_channels = 32, kernel_size = 3, padding = 1), 
                torch.nn.LeakyReLU(),
                # Input = 32 x 256 x 256, Output = 32 x 128 x 128
                torch.nn.MaxPool2d(kernel_size=2),

                # Input = 32 x 128 x 128, Output = 32 x 64 x 64
                torch.nn.Conv2d(in_channels = 32, out_channels = 32, stride=2, kernel_size = 5, padding = 1), 
                torch.nn.LeakyReLU()
            )
        )

        self.add_module(
            "input_transformation_ref",
            torch.nn.Sequential(
                # Input = 3 x 256 x 256, Output = 32 x 256 x 256
                torch.nn.Conv2d(in_channels = 3, out_channels = 32, kernel_size = 3, padding = 1), 
                torch.nn.LeakyReLU(),
                # Input = 32 x 256 x 256, Output = 32 x 128 x 128
                torch.nn.MaxPool2d(kernel_size=2),

                # Input = 32 x 128 x 128, Output = 32 x 64 x 64
                torch.nn.Conv2d(in_channels = 32, out_channels = 32, stride=2, kernel_size = 5, padding = 1), 
                torch.nn.LeakyReLU()
            )
        )

        self.add_module(
            "output_transformation",
            torch.nn.Sequential(
                # Input = 32 x 64 x 64, Output = 32 x 32 x 32
                torch.nn.Conv2d(in_channels = 32, out_channels = 32, stride=2, kernel_size = 5, padding = 1),
                torch.nn.LeakyReLU(),
                # Input = 32 x 32 x 32, Output = 32 x 16 x 16
                torch.nn.MaxPool2d(kernel_size=2),
                # Input = 32 x 16 x 16, Output = 3 x 16 x 16
                torch.nn.Conv2d(in_channels = 32, out_channels = 3, stride=1, kernel_size = 3, padding = 1),
                torch.nn.InsreluceNorm2d(3, affine=True),
                torch.nn.LeakyReLU(),

                torch.nn.Flatten(),
                torch.nn.Linear(3*15*15, 768),
                torch.nn.Softmax(),
                torch.nn.Linear(768, 3)
            )
        )
  
    def forward(self, src, ref):
        src_fmap = self["input_transformation_src"](src)
        ref_fmap = self["input_transformation_ref"](ref)
        fmap = torch.cat([src_fmap, ref_fmap], dim=0)
        return self["output_transformation"](fmap)


class CNNMapAlphaBlendParams(torch.nn.Module):
    def __init__(self, double_decoder: bool = True):
        super(CNNMapAlphaBlendParams, self).__init__()

        self.double_decoder = double_decoder

        # ================== Encoder ================== #
        # Increase the number of channel
        # Input = 3 x 256 x 256, Output = 128 x 256 x 256
        self.convUpChannel = torch.nn.Conv2d(
            in_channels=3, out_channels=112, kernel_size=3, padding=1
        )
        self.batchNorm = torch.nn.InstanceNorm2d(112)
        self.relu = torch.nn.LeakyReLU(negative_slope=0.5)

        # Increase the number of channel
        # Input = 128 x 256 x 256, Output = 256 x 128 x 128 (For image)
        # Input = 6 x 256 x 256, Output = 1 x 128 x 128 (For mask)
        self.maskDownBlock = MaskResDownScaleBlock(112, 224)
        # Input = 256 x 128 x 128, Output = 512 x 64 x 64 (For image)
        # Input = 1 x 128 x 128, Output = 1 x 64 x 64 (For mask)
        self.downBlock = DownScaleBlock(224, 448)

        # Input = 512 x 64 x 64, Output = 512 x 64 x 64
        self.conv = torch.nn.Conv2d(
            in_channels=448, out_channels=448, stride=1, kernel_size=3,
            padding=1
        )

        # ================== Wrap Encoder ================== #
        self.convUpChannel_wrap = torch.nn.Conv2d(
            in_channels=3, out_channels=32, kernel_size=3, padding=1
        )
        self.batchNorm_wrap = torch.nn.InstanceNorm2d(32)
        self.relu_wrap = torch.nn.LeakyReLU(negative_slope=0.5)
        self.downBlock_wrap1 = DownScaleBlock(32, 64)
        self.downBlock_wrap2 = DownScaleBlock(64, 128)

        # ================== Decoder ================== #
        # Input = 1024 x 64 x 64, Output = 512 x 64 x 64
        self.convOut = torch.nn.Conv2d(
            in_channels=1024, out_channels=512, stride=1, kernel_size=3,
            padding=1
        )
        self.reluOut = torch.nn.LeakyReLU(negative_slope=0.5)

        # Input = 512 x 64 x 64, Output = 256 x 128 x 128
        self.up1 = UpScaleResBlock(512, 256)
        # Input = 256 x 128 x 128, Output = 128 x 256 x 256
        self.up2 = UpScaleResBlock(256, 128)
        
        # Input = 128 x 256 x 256, Output = 32 x 256 x 256
        self.convClean1 = torch.nn.Conv2d(in_channels = 128, out_channels = 32, stride=1, kernel_size = 3, padding = 1)
        self.normClean1 = torch.nn.InstanceNorm2d(32)

        # Input = 32 x 128 x 128, Output = 3 x 256 x 256
        self.convClean2 = torch.nn.Conv2d(in_channels = 32, out_channels = 3, stride=1, kernel_size = 3, padding = 1)
        # Fill the weight to give all attention to the reference
        self.convClean2.weight.data.fill_(-0.3)
        self.normClean2 = torch.nn.InstanceNorm2d(3)

        # ================== Second Decoder ================== #
        if self.double_decoder:
            # Input = 1024 x 64 x 64, Output = 512 x 64 x 64
            self.dd_convOut = torch.nn.Conv2d(
                in_channels=1024, out_channels=512, stride=1, kernel_size=3,
                padding=1
            )
            self.dd_reluOut = torch.nn.LeakyReLU(negative_slope=0.5)
            # Input = 512 x 64 x 64, Output = 256 x 128 x 128
            self.dd_up1 = UpScaleResBlock(512, 256)
            # Input = 256 x 128 x 128, Output = 128 x 256 x 256
            self.dd_up2 = UpScaleResBlock(256, 128)
            
            # Input = 128 x 256 x 256, Output = 32 x 256 x 256
            self.dd_convClean1 = torch.nn.Conv2d(in_channels = 128, out_channels = 32, stride=1, kernel_size = 3, padding = 1)
            self.dd_normClean1 = torch.nn.InstanceNorm2d(32)

            # Input = 32 x 128 x 128, Output = 3 x 256 x 256
            self.dd_convClean2 = torch.nn.Conv2d(in_channels = 32, out_channels = 3, stride=1, kernel_size = 3, padding = 1)
            # Fill the weight to give all attention to the reference
            self.dd_convClean2.weight.data.fill_(-0.3)
            self.dd_normClean2 = torch.nn.InstanceNorm2d(3)

  
    def forward(
        self, src: torch.Tensor, ref: torch.Tensor, ref_wrapped: torch.tensor, mask_src: torch.Tensor,
        mask_ref: torch.Tensor, unmakeup: bool = False
    ) -> torch.Tensor:
        """
        Forward method

        :param src: Source image
        :type src: torch.Tensor
        :param ref: Reference image
        :type ref: torch.Tensor
        :param mask_src: Mask of the source
        :type mask_src: torch.Tensor
        :param mask_ref: Mask of the reference
        :type mask_ref: torch.Tensor
        :return: Generated alphas parameter for blending
        :rtype: torch.Tensor
        """
        # ================== Normalize Data ================== #
        mask_src = self.__norm_mask(mask_src)
        mask_ref = self.__norm_mask(mask_ref)
        # ================== Encoder ================== #
        # Source feature map
        # Input = 3 x 256 x 256, Output = 512 x 64 x 64
        src_fmap = self.__input_transformation(src, mask_src)
        # Reference feature map
        # Input = 3 x 256 x 256, Output = 512 x 64 x 64
        ref_fmap = self.__input_transformation(ref, mask_ref)
        wrap_fmp = self.__input_wrapped(ref_wrapped)
        # ================== Feature map ================== #
        # Output = 1024 x 64 x 64
        fmap = torch.cat([src_fmap, ref_fmap, wrap_fmp], dim=1)
        # ================== Decoder ================== #
        # Input = 1024 x 64 x 64, Output = 3 x 256 x 256
        output = self.output_transformation(fmap, unmakeup)

        return output

    def __norm_mask(self, x):
        return (x - 3) / 3


    def __input_wrapped(self, x):
        x = self.convUpChannel_wrap(x)
        x = self.batchNorm_wrap(x)
        x = self.relu_wrap(x)
        x = self.downBlock_wrap1(x)
        x = self.downBlock_wrap2(x)

        return x

    def __input_transformation(
        self, x: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Method to encode information of inputs

        :param x: Input image (either source or reference)
        :type x: torch.Tensor
        :param mask: Mask of the image (containing the face, neck, nose, eye and lip position information)
        :type mask: torch.Tensor
        :return: Feature map of the input image
        :rtype: torch.Tensor
        """
        # Increase the number of channel
        # The objective is to get more information and keep most features when 
        # convolve
        x = self.convUpChannel(x)
        x = self.batchNorm(x)
        x = self.relu(x)

        # Convolve with the mask
        x = self.maskDownBlock(x, mask)
        x = self.downBlock(x)

        # Convolve
        x = self.conv(x)
        x = self.relu(x)

        return x

    def output_transformation(self, x: torch.Tensor, unmakeup: bool = False) -> torch.Tensor:
        """
        Method to decode the information of the feature map

        :param x: Feature map
        :type x: torch.Tensor
        :return: Generated image
        :rtype: torch.Tensor
        """
        if self.double_decoder and unmakeup:
            # Convolve
            x = self.dd_convOut(x)
            x = self.dd_reluOut(x)

            # Increase the size and decrease the depth
            x = self.dd_up1(x)
            x = self.dd_up2(x)

            # Convolve
            x = self.dd_convClean1(x)
            x = self.dd_normClean1(x)

            # Convolve
            x = self.dd_convClean2(x)
            x = self.dd_normClean2(x)
        else:
            # Convolve
            x = self.convOut(x)
            x = self.reluOut(x)

            # Increase the size and decrease the depth
            x = self.up1(x)
            x = self.up2(x)

            # Convolve
            x = self.convClean1(x)
            x = self.normClean1(x)

            # Convolve
            x = self.convClean2(x)
            x = self.normClean2(x)

        return x


class ResDownScaleBlock(torch.nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int
    ) -> None:
        """
        Constructor

        :param in_channels: Number of input channel
        :type in_channels: int
        :param in_mask_channel: Number of input mask channel
        :type in_mask_channel: int
        :param out_channels: Number of output channel
        :type out_channels: int
        """
        super(ResDownScaleBlock, self).__init__()

        # ================== Input Convolution ================== #
        self.conv = torch.nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=2, padding=1,
            bias=False
        )
        self.batch_norm = torch.nn.InstanceNorm2d(out_channels)
        self.relu = torch.nn.LeakyReLU(negative_slope=0.5)

        # ================== Skip convolution ================== #
        self.skip_conv = torch.nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=2, padding=1,
            bias=False
        )
        self.skip_relu = torch.nn.LeakyReLU(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward method

        :param x: Input image
        :type x: torch.Tensor
        :param mask: Input mask
        :type mask: torch.Tensor
        :return: Feature map and mask
        :rtype: torch.Tensor
        """
        x_skip = x

        # ================== Image convolution ================== #
        x = self.conv(x)
        x = self.batch_norm(x)
        x = self.relu(x)

        # ================== Skip convolution ================== #
        x_skip = self.skip_relu(x_skip)
        x_skip = self.skip_conv(x_skip)

        return x_skip + x


class MaskResDownScaleBlock(torch.nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int
    ) -> None:
        """
        Constructor

        :param in_channels: Number of input channel
        :type in_channels: int
        :param in_mask_channel: Number of input mask channel
        :type in_mask_channel: int
        :param out_channels: Number of output channel
        :type out_channels: int
        """
        super(MaskResDownScaleBlock, self).__init__()

        # ================== Input Convolution ================== #
        self.conv = torch.nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=2, padding=1,
            bias=False
        )
        self.batch_norm = torch.nn.InstanceNorm2d(out_channels)
        self.relu = torch.nn.LeakyReLU(negative_slope=0.5)

        # ================== Skip convolution ================== #
        self.skip_conv = torch.nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=2, padding=1,
            bias=False
        )
        self.skip_relu = torch.nn.LeakyReLU(out_channels)

        # ================== Mask convolution ================== #
        self.mask_conv = torch.nn.Conv2d(
            1, 1, kernel_size=3, stride=2, padding=1, bias=False
        )
        self.mask_relu = torch.nn.LeakyReLU(out_channels)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Forward method

        :param x: Input image
        :type x: torch.Tensor
        :param mask: Input mask
        :type mask: torch.Tensor
        :return: Feature map and mask
        :rtype: torch.Tensor
        """
        x_skip = x

        # ================== Image convolution ================== #
        x = self.conv(x)
        x = self.batch_norm(x)
        x = self.relu(x)

        # ================== Skip convolution ================== #
        x_skip = self.skip_relu(x_skip)
        x_skip = self.skip_conv(x_skip)

        # ================== Skip convolution ================== #
        if mask.ndim == 4:
            mask = mask.squeeze(0)

        mask = torch.sum(mask, dim=0).unsqueeze(0)
        mask = self.mask_conv(mask)
        mask = self.mask_relu(mask)

        return x_skip * mask + x

class UpScaleBlock(torch.nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super(UpScaleBlock, self).__init__()

        self.conv = torch.nn.Conv2d(
            in_channels, out_channels, kernel_size=3, padding=1, bias=False
        )
        self.batch_norm = torch.nn.InstanceNorm2d(out_channels)
        self.relu = torch.nn.LeakyReLU(negative_slope=0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.nn.functional.interpolate(
            x, scale_factor=2, mode="bilinear", align_corners=False
        )
        x = self.conv(x)
        x = self.batch_norm(x)
        x = self.relu(x)
        return x


class UpScaleResBlock(torch.nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, skip: bool = True
    ) -> None:
        super(UpScaleResBlock, self).__init__()

        # Bool to tell if skip or not
        self.skip = skip

        # ================== Image convolution ================== #
        self.conv = torch.nn.Conv2d(
            in_channels, out_channels, kernel_size=3, padding=1, bias=False
        )
        self.batch_norm = torch.nn.InstanceNorm2d(out_channels)
        self.relu = torch.nn.LeakyReLU(negative_slope=0.5)

        # ================== Skip convolution ================== #
        self.skip_relu = torch.nn.LeakyReLU(negative_slope=0.5)
        self.skip_conv = torch.nn.Conv2d(
            in_channels, out_channels, kernel_size=1
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward method

        :param x: Feature map
        :type x: torch.Tensor
        :return: Feature map
        :rtype: torch.Tensor
        """
        # Interpolate to increase the size of the feature map
        x = torch.nn.functional.interpolate(
            x, scale_factor=2, mode="bilinear", align_corners=False
        )

        x_skip = x

        # ================== Image convolution ================== #
        x = self.conv(x)
        x = self.batch_norm(x)
        x = self.relu(x)

        # ================== Skip convolution ================== #
        if self.skip:
            x_skip = self.skip_relu(x_skip)
            x_skip = self.skip_conv(x_skip)

        return x_skip + x


class DownScaleBlock(torch.nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        """
        Constructor

        :param in_channels: Number of input channel
        :type in_channels: int
        :param out_channels: Number of output channel
        :type out_channels: int
        """
        super(DownScaleBlock, self).__init__()

        self.conv = torch.nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=2, padding=1,
            bias=False
        )
        self.batch_norm = torch.nn.InstanceNorm2d(out_channels)
        self.relu = torch.nn.LeakyReLU(negative_slope=0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward method

        :param x: Feature map
        :type x: torch.Tensor
        :return: Feature map
        :rtype: torch.Tensor
        """
        x = self.conv(x)
        x = self.batch_norm(x)
        x = self.relu(x)
        return x


class FlatLayer(torch.nn.Module):
    def __init__(self, in_channels: int) -> None:
        """
        Constructor

        :param in_channels: Number of input channel
        :type in_channels: int
        """
        super(FlatLayer, self).__init__()

        self.conv = torch.nn.Conv2d(
            in_channels, 1, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.batch_norm = torch.nn.InstanceNorm2d(1)
        self.relu = torch.nn.LeakyReLU(negative_slope=0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward

        :param x: Feature map
        :type x: torch.Tensor
        :return: Feature map
        :rtype: torch.Tensor
        """
        x = self.conv(x)
        x = self.batch_norm(x)
        x = self.relu(x)
        return x


class ColaboratorDiscriminator(torch.nn.Module):
    def __init__(self, input_size: int) -> None:
        """
        Constructor

        :param input_size: _description_
        :type input_size: int
        """
        super(ColaboratorDiscriminator, self).__init__()
        
        # ================== Downsample ================== #
        self.down1 = DownScaleBlock(3, 6)
        self.down2 = DownScaleBlock(6, 6)
        self.down3 = DownScaleBlock(6, 3)
        self.flat = FlatLayer(3)

        # ================== Classification ================== #
        self.full1 = torch.nn.Linear((input_size // 8), (input_size // 16))
        self.sigmoid = torch.nn.Sigmoid()
        self.full2 = torch.nn.Linear((input_size // 16), 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward method

        :param x: _description_
        :type x: torch.Tensor
        :return: _description_
        :rtype: torch.Tensor
        """

        # ================== Skip convolution ================== #
        x = self.down1(x)
        x = self.down2(x)
        x = self.down3(x)
        x = self.flat(x)

        # ================== Classification ================== #
        x = self.full1(x)
        x = self.sigmoid(x)
        x = self.full2(x)

        return x
        
        