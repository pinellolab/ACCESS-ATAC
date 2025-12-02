import torch
import torch.nn as nn
import torch.nn.functional as F


class BPNet(torch.nn.Module):
    """A basic BPNet model with stranded profile and total count prediction.

    This is a reference implementation for BPNet. The model takes in
    one-hot encoded sequence, runs it through: 

    (1) a single wide convolution operation 

    THEN 

    (2) a user-defined number of dilated residual convolutions

    THEN

    (3a) profile predictions done using a very wide convolution layer 
    that also takes in stranded control tracks 

    AND

    (3b) total count prediction done using an average pooling on the output
    from 2 followed by concatenation with the log1p of the sum of the
    stranded control tracks and then run through a dense layer.

    This implementation differs from the original BPNet implementation in
    two ways:


    (1) The model concatenates stranded control tracks for profile
    prediction as opposed to adding the two strands together and also then
    smoothing that track 

    (2) The control input for the count prediction task is the log1p of
    the strand-wise sum of the control tracks, as opposed to the raw
    counts themselves.

    (3) A single log softmax is applied across both strands such that
    the logsumexp of both strands together is 0. Put another way, the
    two strands are concatenated together, a log softmax is applied,
    and the MNLL loss is calculated on the concatenation. 

    (4) The count prediction task is predicting the total counts across
    both strands. The counts are then distributed across strands according
    to the single log softmax from 3.

    Note that this model is also used as components in the ChromBPNet model,
    as both the bias model and the accessibility model. Both components are
    the same BPNet architecture but trained on different loci.


    Parameters
    ----------
    n_filters: int, optional
        The number of filters to use per convolution. Default is 64.

    n_layers: int, optional
        The number of dilated residual layers to include in the model.
        Default is 8.

    n_outputs: int, optional
        The number of profile outputs from the model. Generally either 1 or 2 
        depending on if the data is unstranded or stranded. Default is 2.

    n_control_tracks: int, optional
        The number of control tracks to feed into the model. When predicting
        TFs, this is usually 2. When predicting accessibility, this is usualy
        0. When 0, this input is removed from the model. Default is 2.

    alpha: float, optional
        The weight to put on the count loss.

    profile_output_bias: bool, optional
        Whether to include a bias term in the final profile convolution.
        Removing this term can help with attribution stability and will usually
        not affect performance. Default is True.

    count_output_bias: bool, optional
        Whether to include a bias term in the linear layer used to predict
        counts. Removing this term can help with attribution stability but
        may affect performance. Default is True.

    name: str or None, optional
        The name to save the model to during training.

    trimming: int or None, optional
        The amount to trim from both sides of the input window to get the
        output window. This value is removed from both sides, so the total
        number of positions removed is 2*trimming.

    verbose: bool, optional
        Whether to display statistics during training. Setting this to False
        will still save the file at the end, but does not print anything to
        screen during training. Default is True.
    """

    def __init__(
        self, 
        out_dim=1,
        n_filters=64, 
        n_layers=8, 
        rconvs_kernel_size=3,
        conv1_kernel_size=21,
        profile_kernel_size=75,
        n_outputs=1, 
        name=None, 
        verbose=False,
    ):
        super().__init__()

        self.out_dim = out_dim
        self.n_filters = n_filters
        self.n_layers = n_layers
        self.n_outputs = n_outputs
        self.verbose = verbose
        
        self.name = name or "bpnet.{}.{}".format(n_filters, n_layers)

        # first convolution without dilation
        self.iconv = torch.nn.Conv1d(4, n_filters, kernel_size=conv1_kernel_size, padding='valid')
        self.irelu = torch.nn.ReLU()

        # residual dilated convolutions
        self.rconvs = torch.nn.ModuleList([
            torch.nn.Conv1d(n_filters, n_filters, kernel_size=rconvs_kernel_size, padding='valid', 
                dilation=2**i) for i in range(1, self.n_layers+1)
        ])

        self.rrelus = torch.nn.ModuleList([
			torch.nn.ReLU() for i in range(1, self.n_layers+1)
		])

        # profile prediction
        self.fconv = torch.nn.Conv1d(n_filters+n_control_tracks, n_outputs, 
            kernel_size=profile_kernel_size, padding='valid', bias=profile_output_bias)
        
        # count prediction
        n_count_control = 1 if n_control_tracks > 0 else 0
        self.global_avg_pool = torch.nn.AdaptiveAvgPool1d(1)

        self.linear = torch.nn.Linear(n_filters+n_count_control, 1, 
            bias=count_output_bias)


    def forward(self, x, x_ctl=None):
        """A forward pass of the model.

        This method takes in a nucleotide sequence x, a corresponding
        per-position value from a control track, and a per-locus value
        from the control track and makes predictions for the profile 
        and for the counts. This per-locus value is usually the
        log(sum(X_ctl_profile)+1) when the control is an experimental
        read track but can also be the -output from another model.

        Parameters
        ----------
        x: torch.tensor, shape=(batch_size, 4, length)
            The one-hot encoded batch of sequences.

        X_ctl: torch.tensor or None, shape=(batch_size, n_strands, length)
            A value representing the signal of the control at each position in 
            the sequence. If no controls, pass in None. Default is None.

        Returns
        -------
        pred_profile: torch.tensor, shape=(batch_size, n_strands, out_length)
            The output predictions for each strand trimmed to the output
            length.
        pred_count: torch.tensor, shape=(batch_size, 1)
        """
        if x.shape[1] != 4:
            x = x.permute(0, 2, 1)
        x = self.get_embs_after_crop(x)

        if self.verbose: print(f'trunk shape: {x.shape}')

        if x_ctl is not None:
            crop_size = (x_ctl.shape[2] - x.shape[2]) // 2
            if self.verbose: print(f'crop_size: {crop_size}')
            if crop_size > 0:
                x_ctl = x_ctl[:, :, crop_size:-crop_size]
            else:
                x_ctl = F.pad(x_ctl, (-crop_size, -crop_size))

        pred_profile = self.profile_head(x, x_ctl=x_ctl) # before log_softmax
        pred_count = self.count_head(x, x_ctl=x_ctl) #.squeeze(-1) # (batch_size, 1)

        return pred_profile, pred_count
    

    def get_embs_after_crop(self, x):
        x = self.irelu(self.iconv(x))
        for i in range(self.n_layers):
            conv_x = self.rrelus[i](self.rconvs[i](x))
            crop_len = (x.shape[2] - conv_x.shape[2]) // 2
            if crop_len > 0:
                x = x[:, :, crop_len:-crop_len]
            x = torch.add(x, conv_x)
        
        return x
    
    def count_head(self, x, x_ctl=None):

        # pred_count = torch.mean(x, dim=2)
        pred_count = self.global_avg_pool(x).squeeze(-1)
        if x_ctl is not None:
            x_ctl = torch.sum(x_ctl, dim=(1, 2)).unsqueeze(-1)
            pred_count = torch.cat([pred_count, torch.log1p(x_ctl)], dim=-1)
        pred_count = self.linear(pred_count)
        return pred_count
    

if __name__ == "__main__":
    
    import torch
    import numpy as np
    from utils import random_seq, one_hot_encode

    seq1 = random_seq(256)
    seq2 = random_seq(256)
    x1 = one_hot_encode(seq1)
    x2 = one_hot_encode(seq2)

    signal_raw = np.expand_dims(np.random.rand(256), axis=1)
    x1 = np.concatenate([x1, signal_raw], axis=1)
    x2 = np.concatenate([x2, signal_raw], axis=1)
    
    x1 = torch.tensor(x1).float()
    x2 = torch.tensor(x2).float()

    # # add ATAC-seq signal
    x1 = x1.unsqueeze(dim=0)
    x2 = x2.unsqueeze(dim=0)

    x = torch.cat((x1, x2), dim=0)

    print(x.shape)

    model = MaxATACCNN(in_ch=5)

    x = model(x)

    print(x)
