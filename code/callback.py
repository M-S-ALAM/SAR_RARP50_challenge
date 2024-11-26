class EarlyStopping:
    def __init__(self, patience=10, verbose=False, delta=0, mode='max'):
        """
        Args:
            patience (int): How long to wait after the last time the monitored metric improved.
                            Default: 10
            verbose (bool): If True, prints a message for each improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                           Default: 0
            mode (str): One of ['min', 'max']. If 'min', stops when the metric stops decreasing;
                        if 'max', stops when the metric stops increasing.
                        Default: 'max'
        """
        self.patience = patience
        self.verbose = verbose
        self.delta = delta
        self.mode = mode
        self.best_score = None
        self.epochs_no_improve = 0  # Correctly initialize to 0
        self.early_stop = False

    def __call__(self, metric):
        if self.mode == 'max':
            score = metric
        else:
            score = -metric

        if self.best_score is None:
            self.best_score = score
        elif (self.mode == 'max' and score <= self.best_score + self.delta) or \
             (self.mode == 'min' and score >= self.best_score - self.delta):
            self.epochs_no_improve += 1
            if self.epochs_no_improve >= self.patience:
                if self.verbose:
                    print("Early stopping")
                self.early_stop = True
        else:
            self.best_score = score
            self.epochs_no_improve = 0  # Reset to 0 on improvement

