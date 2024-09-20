# numRNN

GazeRNN architecture applied to dot counting task. 

You can train the dot count network with function train_dot() or the saccade network with train(). Most of the time, train() needs at least about 50,000 to 100,000 iterations to perform well.

Here are my immediate ideas for what's important going forward:

**Fix the task / stimuli parameters.** I’ve been using a very simplified version of the original dot task, mainly for faster training as I don’t have access to computing resources. If you’re able to increase the size of the input image without making training too slow, it will allow much more flexibility with dot size, field size, etc. Also, I probably didn’t do a great job of ensuring that the training stimuli set is balanced for all dimensions — it’s a good idea to check this so that the model is unable to “cheat”. 

**Choose reasonable values for tau and noise level.** This should ideally be done to match what a human sees for a given fixation. High tau and noise level are a lot more difficult to train, but help prevent model cheating. 

**Discuss the model architecture.** For instance, do we need two separate fovea types? Does the Gaussian blurring + subtraction make sense? Also, carefully normalizing the inputs will be important to 1) help the training go smoother and 2) prevent the model from cheating. 

**Make sure gradients are flowing.** Right now in the main loss_fn() I’m using MSELoss with the stacked dot count outputs from each saccade, which are obtained via an argmax of the dot count logits. This is very bad because argmax is not differentiable, which means the network can’t effectively use the total dot count error to learn. I began to implement a Straight-Through Estimator (commented out in loss_fn()) to circumvent this, but didn’t finish. You might also choose to switch back to a continuous dot count output (instead of discrete), which doesn’t have this argmax problem. I have the code for the continuous version, so just let me know if you ever want to work from that. 

**Train a better local dot count network.** I didn’t spend too much time trying to train a great dot count network, but it’s a good idea to do that so that it can be plugged into the saccade network with confidence.

**Learning how to stop counting.** We currently have 2 ideas for this. First, we can train (end-to-end) a separate EstimateCNN that is first run on each trial to estimate the total number of saccades needed; then, that many saccades are performed on the trial. The second idea is that at each timestep we can output a “done” signal (either from the SaccadeCNN or a separate network); if it is 1 then the network stops counting on that trial. Both of these need to be implemented carefully to avoid differentiability issues. I started playing around with the second idea since it feels more realistic, but the first idea is probably easier to train. We can also look in the RNN/LSTM literature for more inspiration.

**Thoughts for far future**
An interesting question is how to balance the weights on the different loss terms. One idea is to calculate what are the human ratios for these loss terms based on eye movement data. For example, if there is a single far-away dot, will humans look at it and count it, or accept a small error to save saccade energy? How does this change based on trial duration?
This was an idea from the original GazeRNN project, but adding duration into the network would allow us to ask: does the amount of time allotted to an agent influence their counting strategy? For example, at each timestep the RNN can output a “fixation duration”; long durations increase the foveal radius and short durations limit it. For this to work, we would need to train the dot count network to work with a variable foveal radius.
