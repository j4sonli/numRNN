import os
import time
import math
import cv2
import numpy as np
import torch
from torch import nn, optim
import torch.nn.functional as F
import pickle
import matplotlib.pyplot as plt
from scipy.ndimage import distance_transform_edt
from scipy.stats import norm
import cv2

from torchvision import models, transforms

IMG_DIM = 40
SPATIAL_BASIS = torch.as_tensor(np.meshgrid(np.arange(IMG_DIM), np.arange(IMG_DIM))).permute(1, 2, 0).to('cpu')

np.random.seed(1033)

MAX_DOTS = 10
TAU = 0.4
NOISE_STDEV = 0.65
N_SACCADES = 10
MEM_CHANNELS = 8
BATCH_SIZE = 16
LEARNING_RATE = 0.001
AMP_COST = 0.001
DOT_COST = 1
DOT_COUNTER = 'simplenet'  # one of 'visionnet', 'simplenet', 'ideal'

# below is a calculation of the radius from the fovea at which
# it becomes difficult to discern the original value of the pixel,
# given the chosen tau and noise_stdev hyperparameters
# p = probability of correct classification; ideally as close to 0.5 as possible (random chance)
p = 0.55
# discriminability d'
d_prime = norm.ppf(p) * np.sqrt(2)
y = 2 * d_prime * NOISE_STDEV ** 2
x_p = (y + 1 + np.sqrt(2 * y + 1)) / y
x_n = (y + 1 - np.sqrt(2 * y + 1)) / y
fovea_radius = -1 / TAU * np.log(np.min((x_n, x_p)))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

SPATIAL_BASIS = SPATIAL_BASIS.to(device)

TRAIN_SAVES_DIR = 'train_saves'
SAVED_MODELS_DIR = 'saved_models'

version = 'd2,10r1,2tau0.4noise0.65disc'

if not os.path.exists(TRAIN_SAVES_DIR):
    os.makedirs(TRAIN_SAVES_DIR)
if not os.path.exists(SAVED_MODELS_DIR):
    os.makedirs(SAVED_MODELS_DIR)


def dotField2GKA(radii, fieldRadius):
    # Field coordinates
    xx, yy = np.meshgrid(np.arange(-fieldRadius, fieldRadius + 1), np.arange(-fieldRadius, fieldRadius + 1))

    # Distance to nearest object
    d = fieldRadius - np.sqrt(xx ** 2 + yy ** 2)

    # avoid (0,0) to prevent overlap with fixation
    new_d = np.sqrt((0 - xx) ** 2 + (0 - yy) ** 2) - radii[1]
    d = np.minimum(new_d, d)

    # Coordinates of each point
    x = np.zeros(len(radii))
    y = np.zeros(len(radii))

    for i in range(len(radii)):
        is_valid = d > radii[i]
        xo = xx[is_valid]
        if len(xo) == 0:
            # print('Ran out of space before placing all dots')
            err = 1
            break
        else:
            err = 0
        yo = yy[is_valid]
        ind = np.random.randint(0, len(xo))
        x[i] = xo[ind]
        y[i] = yo[ind]

        new_d = np.sqrt((x[i] - xx) ** 2 + (y[i] - yy) ** 2) - radii[i]
        d = np.minimum(new_d, d)

    pts = np.column_stack((x, y))

    return pts, err


def generate_stimulus(dot_min=2, dot_max=MAX_DOTS,
                      dot_radius_min=1, dot_radius_max=2,
                      image_field_radius_min=IMG_DIM // 2, image_field_radius_max=IMG_DIM):
    num_dots = np.random.randint(dot_min, dot_max + 1)
    buffer = 10
    while True:
        radii = np.ones(num_dots) * np.random.randint(dot_radius_min, dot_radius_max + 1)
        field_radius = 100
        dts, err = dotField2GKA(radii * buffer, field_radius)
        if not err:
            break
    image_field_radius = np.random.randint(image_field_radius_min, image_field_radius_max + 1)
    # create a background image that's "mulfac" times larger (for anti-aliasing)
    mulfac = np.random.randint(5, 9)
    M = np.zeros((mulfac * IMG_DIM, mulfac * IMG_DIM))
    dts_aa = mulfac * np.round(
        (dts + field_radius) * image_field_radius / (field_radius * 2) + (IMG_DIM - image_field_radius) / 2
    ).astype('int')
    M[dts_aa[:, 0], dts_aa[:, 1]] = 1
    img = (distance_transform_edt(M == 0) <= radii[0] * mulfac).astype('float')
    img = cv2.resize(img, None, fx=1 / mulfac, fy=1 / mulfac)
    dots = np.flip(dts_aa, axis=1) / mulfac - 0.5
    return img, dots, num_dots


def create_masks(spatial_basis, centers, tau):
    d = (torch.tile(spatial_basis, (centers.shape[0], 1, 1, 1)) - torch.reshape(centers, (-1, 1, 1, 2))).square().sum(dim=3).sqrt()
    return torch.exp(-d * tau)


def as_minutes(seconds):
    return '{:d}m {:d}s'.format(math.floor(seconds / 60), int(seconds - 60 * math.floor(seconds / 60)))


def blur(images, kernel_size=20):
    # define constant kernel and apply to batch
    # kernel_1d = cv2.getGaussianKernel(IMG_DIM // 4 + 1, 0)
    kernel_1d = cv2.getGaussianKernel(kernel_size, 0)
    kernel_2d = np.outer(kernel_1d, kernel_1d.T)
    blurred = torch.tensor([cv2.filter2D(i.numpy(), -1, kernel_2d) for i in images])
    # minmax normalize so that every pixel in the blurred images is between 0 and 1
    blurred = blurred / torch.amax(blurred, dim=(1, 2)).view(BATCH_SIZE, 1, 1)
    return blurred


#=============== Dot Network training functions ================#

def generate_batch_dot(batch_size):
    images, dots_in_foveas, centers = [], [], []
    while len(images) < batch_size:
        img, dots, _ = generate_stimulus()
        # choose a random fixation point, with higher weights in the middle of the image
        center_weights = np.zeros(IMG_DIM) + 0.01
        center_weights[(IMG_DIM//4):(3*IMG_DIM//4)] += 0.6 / (IMG_DIM / 2)
        x = np.random.choice(np.arange(IMG_DIM), p=center_weights)
        y = np.random.choice(np.arange(IMG_DIM), p=center_weights)
        center = np.array([y, x])
        # how many dot centers are in the discriminable fovea?
        dots_in_fovea = np.where(np.linalg.norm(dots - center, axis=1) < fovea_radius)[0].size

        # weighted frequencies of dots_in_fovea to re-balance the training distribution
        dot_count_weights = [0.12, 0.1, 0.2, 0.4, 0.9, 1, 1, 1, 1, 1, 1]
        if np.random.rand() < dot_count_weights[dots_in_fovea]:
            images.append(img)
            dots_in_foveas.append([dots_in_fovea])
            centers.append(center)

    images = torch.as_tensor(images).float().to(device)
    dots_in_foveas = torch.as_tensor(dots_in_foveas).float().to(device)
    centers = torch.as_tensor(centers).float().to(device)
    return images, dots_in_foveas, centers


class DotCNN(nn.Module):
    def __init__(self):
        super(DotCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, stride=2)
        self.conv2 = nn.Conv2d(32, 32, 3, stride=2)
        self.conv3 = nn.Conv2d(32, 64, 3, stride=2)
        self.fc1 = nn.Linear(64 * 4 ** 2, 64)
        self.fc2 = nn.Linear(64, 16)
        self.fc3 = nn.Linear(16, 16)
        self.fc4 = nn.Linear(16, MAX_DOTS+1)

    def forward(self, img):
        x = F.relu(self.conv1(img))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x


def loss_fn_dot(dot_counts, dots_in_foveas):
    return F.cross_entropy(dot_counts, dots_in_foveas)


def run_batch_dot(action, dotcnn, images=None, dots_in_foveas=None, centers=None, dotcnn_optimizer=None):
    """Train or test on a single batch."""
    with torch.set_grad_enabled(action == 'train'):
        if images is None:
            images, dots_in_foveas, centers = generate_batch_dot(BATCH_SIZE)

        if action == 'train':
            dotcnn_optimizer.zero_grad()
        # initialize network to randomly selected fixation points
        curr_masks = create_masks(SPATIAL_BASIS, centers, TAU).to(device)
        noise = torch.normal(0, NOISE_STDEV, size=(IMG_DIM, IMG_DIM)).to(device)
        # predict dot count
        masked_img = images.unsqueeze(1) * curr_masks.unsqueeze(1) + noise * (1 - curr_masks.unsqueeze(1))
        if DOT_COUNTER == 'visionnet':
            dot_counts = dotcnn(transforms.Resize((224, 224))(masked_img.repeat(1, 3, 1, 1)))
        elif DOT_COUNTER == 'simplenet':
            dot_counts = dotcnn(masked_img)

        # compute and backprop the loss
        loss = loss_fn_dot(dot_counts, dots_in_foveas.flatten().long())
        if action == 'train':
            loss.backward()
            dotcnn_optimizer.step()

        dot_counts = dot_counts.detach().cpu().numpy()
        return loss.item(), dot_counts


def train_dot(dotcnn, version, max_iters=1e7, print_every=100, save_every=1e3):
    dotcnn_optimizer = optim.Adam(dotcnn.parameters(), lr=LEARNING_RATE)
    loss_buffer = 0
    print('Training model version: {}'.format(version))
    start_time = time.time()
    for curr_iter in range(1, int(max_iters) + 1):
        train_loss, _ = run_batch_dot('train', dotcnn, dotcnn_optimizer=dotcnn_optimizer)
        loss_buffer += train_loss
        if curr_iter % print_every == 0:
            loss_avg = loss_buffer / print_every
            loss_buffer = 0
            time_elapsed = as_minutes(time.time() - start_time)
            print('{} ({:d} {:d}%) {:.4f}'.format(time_elapsed, curr_iter, round(curr_iter / max_iters * 100), loss_avg))
        if curr_iter % save_every == 0:
            torch.save(dotcnn.state_dict(), '{}/dotcnn-{}-it{}.pt'.format(SAVED_MODELS_DIR, version, curr_iter))

#================ Saccade Network training functions ===============#

def generate_batch(batch_size):
    images, dotss, true_dot_counts = [], [], []
    for _ in range(batch_size):
        img, dots, num_dots = generate_stimulus()
        images.append(img)
        dotss.append(dots)
        true_dot_counts.append([num_dots])
    images = torch.as_tensor(images).float().to(device)
    true_dot_counts = torch.as_tensor(true_dot_counts).float().to(device)
    return images, true_dot_counts, dotss


class SaccadeCNN(nn.Module):
    def __init__(self, mem_channels):
        super(SaccadeCNN, self).__init__()
        self.mem_channels = mem_channels
        self.conv1 = nn.Conv2d(1 + self.mem_channels + 1, 32, 3, stride=2)
        self.conv2 = nn.Conv2d(32, 32, 3, stride=2)
        self.conv3 = nn.Conv2d(32, 64, 3, stride=2)
        self.fc1 = nn.Linear(64 * 4 ** 2, 64)
        self.eye_pos_fc1 = nn.Linear(64, 16)
        self.eye_pos_fc2 = nn.Linear(16, 16)
        self.eye_pos_fc3 = nn.Linear(16, 2)

    def forward(self, img, memory, curr_mask, noise, blurred):
        x = torch.cat([img * curr_mask + (noise + blurred) * (1 - curr_mask), memory, curr_mask.detach()], dim=1)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))

        eye_pos = F.relu(self.eye_pos_fc1(x))
        eye_pos = F.relu(self.eye_pos_fc2(eye_pos))
        eye_pos = self.eye_pos_fc3(eye_pos)

        return eye_pos


class MemCNN(nn.Module):
    def __init__(self, mem_channels):
        super(MemCNN, self).__init__()
        self.mem_channels = mem_channels
        self.conv1 = nn.Conv2d(1 + self.mem_channels + 1, 8, 3, padding='same')
        self.conv2 = nn.Conv2d(8, 8, 3, padding='same')
        self.conv3 = nn.Conv2d(8, self.mem_channels, 3, padding='same')

    def forward(self, img, memory, curr_mask, noise, blurred):
        x = torch.cat([img * curr_mask + (noise + blurred) * (1 - curr_mask), memory, curr_mask.detach()], dim=1)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.conv3(x)
        return x


class DoneCNN(nn.Module):
    def __init__(self, mem_channels):
        super(DoneCNN, self).__init__()
        self.mem_channels = mem_channels
        self.conv1 = nn.Conv2d(1 + self.mem_channels + 1, 32, 3, stride=2)
        self.conv2 = nn.Conv2d(32, 32, 3, stride=2)
        self.conv3 = nn.Conv2d(32, 64, 3, stride=2)
        self.fc1 = nn.Linear(64 * 4 ** 2, 64)
        self.fc2 = nn.Linear(64, 16)
        self.fc3 = nn.Linear(16, 1)

    def forward(self, img, memory, curr_mask, noise, blurred):
        x = torch.cat([img * curr_mask + (noise + blurred) * (1 - curr_mask), memory, curr_mask.detach()], dim=1)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = torch.flatten(x, 1)

        done = F.relu(self.fc1(x))
        done = F.relu(self.fc2(done))
        done = F.sigmoid(self.fc3(done))

        return done


def loss_fn(saccade_i, eye_positionss, blurred, dot_count_logitss, dot_countss, true_dot_counts):
    amplitude = torch.linalg.norm(eye_positionss[-1] - eye_positionss[-2], dim=1)
    if saccade_i == N_SACCADES - 1:
        # argmax_indices = torch.argmax(torch.stack(dot_count_logitss), dim=2)
        # # Straight-Through Estimator: pretend argmax is differentiable by using softmax
        # softmax_output = F.softmax(torch.stack(dot_count_logitss), dim=-1)
        # # gradient flows through softmax
        # dot_sums_w_grad = (softmax_output * torch.arange(11)).sum(dim=(0,2))
        # dot_err = F.mse_loss(true_dot_counts, dot_sums_w_grad.unsqueeze(1))
        dot_err = F.mse_loss(true_dot_counts, torch.sum(torch.stack(dot_countss), dim=0).unsqueeze(1))
        return torch.mean(torch.sum(blurred, dim=(1,2))) + dot_err * DOT_COST + torch.mean(amplitude) * AMP_COST
    return torch.mean(amplitude) * AMP_COST


def run_batch(action, saccadecnn, memcnn, dotcnn, images=None, true_dot_counts=None, dotss=None,
              saccadecnn_optimizer=None, memcnn_optimizer=None, donecnn_optimizer=None):
    """Train or test on a single batch."""
    with torch.set_grad_enabled(action == 'train'):
        if images is None:
            images, true_dot_counts, dotss = generate_batch(BATCH_SIZE)

        eye_positionss = []
        dot_countss = []
        dot_count_logitss = []

        if action == 'train':
            saccadecnn_optimizer.zero_grad()
            memcnn_optimizer.zero_grad()
            donecnn_optimizer.zero_grad()
        loss = 0

        batch_memory = None
        # fixation begins at center
        centers = torch.ones((BATCH_SIZE, 2)).to(device) * IMG_DIM/2
        eye_positionss.append(centers.detach().cpu())
        curr_masks = create_masks(SPATIAL_BASIS, centers, TAU).to(device)
        blurred = blur(images)

        # indicates whether to keep or discard dot count outputs
        done_mask = torch.ones((BATCH_SIZE, N_SACCADES+1), requires_grad=False)
        for i in range(N_SACCADES):
            noise = torch.normal(0, NOISE_STDEV, size=(IMG_DIM, IMG_DIM)).to(device)
            # update internal memory; initialize all 8 memory channels to the masked and noised image
            if batch_memory is None:
                batch_memory = (images * curr_masks + (noise + blurred) * (1 - curr_masks)).unsqueeze(1).repeat(1,
                                                                                                    saccadecnn.mem_channels,
                                                                                                    1, 1).to(device)
            else:
                batch_memory = memcnn(images.unsqueeze(1), batch_memory, curr_masks.unsqueeze(1), noise, blurred.unsqueeze(1))
            # predict number of dots
            masked_img = images.unsqueeze(1) * curr_masks.unsqueeze(1) + noise * (1 - curr_masks.unsqueeze(1))
            if DOT_COUNTER == 'ideal':
                # calculate how many dot centers are within fovea_radius from the current eye position
                dot_distances = [np.linalg.norm(dotss[t] - eye_positionss[-1][t].detach().numpy(), axis=1) for t in range(BATCH_SIZE)]
                dot_counts = torch.tensor([np.where(dot_distances[t] < fovea_radius)[0].size for t in range(BATCH_SIZE)])
                dot_count_logits = F.one_hot(dot_counts, num_classes=MAX_DOTS+1).float()
            else:
                if DOT_COUNTER == 'visionnet':
                    dot_count_logits = dotcnn(transforms.Resize((224, 224))(masked_img.repeat(1, 3, 1, 1)))
                elif DOT_COUNTER == 'simplenet':
                    dot_count_logits = dotcnn(masked_img)
                dot_counts = torch.argmax(dot_count_logits, dim=1).unsqueeze(1)
            dot_count_logits = dot_count_logits * done_mask[:, i].view(-1, 1)
            dot_counts = dot_counts * done_mask[:, i]
            # find next eye position
            eye_delta = saccadecnn(images.unsqueeze(1), batch_memory, curr_masks.unsqueeze(1), noise, blurred.unsqueeze(1)) * done_mask[:, i].view(-1, 1)
            eye_positions = eye_positionss[-1] + eye_delta
            # make a saccade
            curr_masks = create_masks(SPATIAL_BASIS, eye_positions, TAU).to(device) * done_mask[:, i].view(-1, 1, 1)
            # subtract fovea from blurred image
            blurred = torch.clamp(blurred - curr_masks, min=0)

            # are we done? zero out dot mask from this saccade onwards
            done_logit = donecnn(images.unsqueeze(1), batch_memory, curr_masks.unsqueeze(1), noise, blurred.unsqueeze(1))
            done_mask = torch.where((done_logit.flatten().unsqueeze(1) < 0.5) & (torch.arange(done_mask.shape[1]) >= i), 0, done_mask)

            eye_positionss.append(eye_positions)
            dot_countss.append(dot_counts)
            dot_count_logitss.append(dot_count_logits)

            # add dot count from last eye fixation position
            if i == N_SACCADES - 1:
                # predict number of dots
                masked_img = images.unsqueeze(1) * curr_masks.unsqueeze(1) + noise * (1 - curr_masks.unsqueeze(1))
                if DOT_COUNTER == 'ideal':
                    # calculate how many dot centers are within fovea_radius from the current eye position
                    dot_distances = [np.linalg.norm(dotss[t] - eye_positionss[-1][t].detach().numpy(), axis=1) for t in range(BATCH_SIZE)]
                    dot_counts = torch.tensor([np.where(dot_distances[t] < fovea_radius)[0].size for t in range(BATCH_SIZE)])
                    dot_count_logits = F.one_hot(dot_counts, num_classes=MAX_DOTS+1).float()
                else:
                    if DOT_COUNTER == 'visionnet':
                        dot_count_logits = dotcnn(transforms.Resize((224, 224))(masked_img.repeat(1, 3, 1, 1)))
                    elif DOT_COUNTER == 'simplenet':
                        dot_count_logits = dotcnn(masked_img)
                    dot_counts = torch.argmax(dot_count_logits, dim=1).unsqueeze(1)
                dot_count_logits = dot_count_logits * done_mask[:, i].view(-1, 1)
                dot_counts = dot_counts * done_mask[:, i]

                dot_countss.append(dot_counts)
                dot_count_logitss.append(dot_count_logits)

            # compute the loss
            loss += loss_fn(i, eye_positionss, blurred, dot_count_logitss, dot_countss, true_dot_counts)

        if action == 'train':
            loss.backward()
            saccadecnn_optimizer.step()
            memcnn_optimizer.step()

        # reshape network outputs
        eye_positionss = np.array([e.detach().cpu().numpy() for e in eye_positionss])
        dot_count_logitss = np.array([d.detach().cpu().numpy() for d in dot_count_logitss])
        eye_positionss = np.transpose(eye_positionss, axes=(1, 0, 2))
        dot_count_logitss = np.transpose(dot_count_logitss, axes=(1, 0, 2))

        return loss.item(), eye_positionss, dot_count_logitss


def train(saccadecnn, memcnn, donecnn, dotcnn, version, max_iters=5e6, print_every=1e2, save_every=1e3):
    train_losses, test_losses = [], []
    loss_buffer = 0
    saccadecnn_optimizer = optim.Adam(saccadecnn.parameters(), lr=LEARNING_RATE)
    memcnn_optimizer = optim.Adam(memcnn.parameters(), lr=LEARNING_RATE)
    donecnn_optimizer = optim.Adam(donecnn.parameters(), lr=LEARNING_RATE)
    print('Training model version: {}'.format(version))
    start_time = time.time()
    for curr_iter in range(1, int(max_iters) + 1):
        train_loss, _, _ = run_batch('train', saccadecnn, memcnn, donecnn, dotcnn,
                                     saccadecnn_optimizer=saccadecnn_optimizer,
                                     memcnn_optimizer=memcnn_optimizer,
                                     donecnn_optimizer=donecnn_optimizer)
        loss_buffer += train_loss
        train_losses.append(train_loss)

        if curr_iter % print_every == 0:
            loss_avg = loss_buffer / print_every
            loss_buffer = 0
            time_elapsed = as_minutes(time.time() - start_time)
            print('{} ({:d} {:d}%) {:.4f}'.format(time_elapsed, curr_iter, round(curr_iter / max_iters * 100), loss_avg))

        if curr_iter % save_every == 0:
            with open('{}/train_losses-{}.pickle'.format(TRAIN_SAVES_DIR, version), 'wb') as f:
                pickle.dump(train_losses, f)
            with open('{}/val_losses-{}.pickle'.format(TRAIN_SAVES_DIR, version), 'wb') as f:
                pickle.dump(test_losses, f)
            torch.save(saccadecnn.state_dict(), '{}/saccadecnn-{}-it{}.pt'.format(SAVED_MODELS_DIR, version, curr_iter))
            torch.save(memcnn.state_dict(), '{}/memcnn-{}-it{}.pt'.format(SAVED_MODELS_DIR, version, curr_iter))


########################################################################################################################

if DOT_COUNTER == 'simplenet':
    dotcnn = DotCNN().to(device)
elif DOT_COUNTER == 'visionnet':
    dotcnn = models.efficientnet_b0(weights='IMAGENET1K_V1').to(device)
    for param in dotcnn.parameters():
        param.requires_grad = False
    dotcnn.classifier = nn.Linear(dotcnn.classifier[1].in_features, MAX_DOTS+1).to(device)

# train_dot(dotcnn, version, print_every=100, save_every=1000)

#=========#

saccadecnn = SaccadeCNN(MEM_CHANNELS).to(device)
memcnn = MemCNN(MEM_CHANNELS).to(device)
donecnn = DoneCNN(MEM_CHANNELS).to(device)
# load pretrained dot count CNN and freeze all parameters
if DOT_COUNTER == 'ideal':
    dotcnn = None
else:
    if DOT_COUNTER == 'simplenet':
        dotcnn = DotCNN().to(device)
        dotcnn.load_state_dict(torch.load('saved_models/dotcnn-d2,10r1,2tau0.4noise0.65disc-it13000.pt'))
    elif DOT_COUNTER == 'visionnet':
        dotcnn = models.efficientnet_b0(weights='IMAGENET1K_V1').to(device)
        dotcnn.classifier = nn.Linear(dotcnn.classifier[1].in_features, MAX_DOTS+1).to(device)
        dotcnn.load_state_dict(torch.load('saved_models/dotcnn-d2,10r1,2tau0.4noise0.65disc-it13000.pt'))
    for param in dotcnn.parameters():
        param.requires_grad = False

train(saccadecnn, memcnn, donecnn, dotcnn, version, print_every=100, save_every=1000)





#########################################   test dot count network   ###################################################

test_iteration = 50900
dotcnn.load_state_dict(torch.load('saved_models/dotcnn-{}-it{}.pt'.format(version, test_iteration)))
dotcnn.eval()

images, dots_in_foveas, centers = generate_batch_dot(BATCH_SIZE)

test_loss, dot_counts = run_batch_dot('test', dotcnn, images, dots_in_foveas, centers)

dot_counts = torch.argmax(F.softmax(torch.tensor(dot_counts)), axis=1).unsqueeze(1).numpy()

def plot_trial(idx, axs, i, j):
    axs[i, j].imshow(images[idx], cmap='Greys_r')
    # curr_masks = create_masks(SPATIAL_BASIS, centers[idx].unsqueeze(0), TAU)[0].numpy()
    # noise = torch.normal(0, NOISE_STDEV, size=(IMG_DIM, IMG_DIM)).to(device)
    # axs[i, j].imshow(images[idx] * curr_masks + noise * (1-curr_masks))
    axs[i, j].plot(*centers[idx], c='r', marker='o', ls='--')
    axs[i, j].set_yticks([])
    axs[i, j+1].axhline(dots_in_foveas[idx][0], c='k', ls='--')
    axs[i, j+1].plot(0, dot_counts[idx][0], c='r', marker='o', ls='--')
    axs[i, j+1].set_ylabel('Dot count')
    axs[i, j+1].yaxis.grid(True, which='both', color='k', alpha=0.2)
    axs[i, j+1].minorticks_on()
    axs[i, j].set_xticks([])
    axs[i, j+1].set_ylim(-1, 7)


fig, axs = plt.subplots(4, 6, figsize=(16, 8))
idxs = np.arange(12) #np.random.choice(BATCH_SIZE, size=12, replace=False)
for i in range(4):
    for j in (0, 2, 4):
        plot_trial(idxs[i*3+j//2], axs, i, j)


######### make violinplot for dot count network ########

dot_pred, dot_true = [], []
for i in range(1000):
    print(i)
    images, dots_in_foveas, centers = generate_batch_dot(BATCH_SIZE)
    test_loss, dot_counts = run_batch_dot('test', dotcnn, images, dots_in_foveas, centers)
    dot_pred.extend(np.argmax(dot_counts, axis=1).tolist())
    dot_true.extend(dots_in_foveas.flatten().tolist())

for i in np.unique(dot_true):
    plt.violinplot([dot_pred[j] for j,_ in enumerate(dot_pred) if dot_true[j] == i], positions=[i])
plt.xlabel('True dot count')
plt.ylabel('Predicted dot count')

##############################################  test saccade network  ##################################################

test_iteration = 51000
saccadecnn.load_state_dict(torch.load('saved_models/saccadecnn-{}-it{}.pt'.format(version, test_iteration)))
memcnn.load_state_dict(torch.load('saved_models/memcnn-{}-it{}.pt'.format(version, test_iteration)))
saccadecnn.eval()
memcnn.eval()

test_images, test_true_dot_counts, test_dotss = generate_batch(BATCH_SIZE)

test_loss, eye_positions, dot_counts = run_batch('test', saccadecnn, memcnn, dotcnn, test_images, test_true_dot_counts, test_dotss)

dot_counts = np.argmax(dot_counts, axis=2)[:, :, np.newaxis]

def plot_trial(idx, axs, i, j):
    axs[i, j].imshow(test_images[idx], cmap='Greys_r')
    axs[i, j].set_xticks([])
    axs[i, j].set_yticks([])
    cmap = plt.cm.get_cmap('autumn_r')
    axs[i, j].plot(*eye_positions[idx].T, c='r')
    for k, (x, y) in enumerate(eye_positions[idx]):
        axs[i, j].plot(x, y, marker='o', c=cmap(k/len(eye_positions[idx])))
    axs[i, j+1].axhline(test_true_dot_counts[idx][0], c='k', ls='--')
    axs[i, j+1].plot(np.arange(0, len(dot_counts[idx])), *np.abs(dot_counts[idx].T), c='r')
    axs[i, j+1].plot(np.arange(0, len(dot_counts[idx])), *np.cumsum(np.abs(dot_counts[idx].T), axis=1), c='b')
    for k, ct in enumerate(dot_counts[idx]):
        axs[i, j+1].plot(k, np.abs(ct), marker='o', c=cmap((k)/len(dot_counts[idx])))
    axs[i, j+1].set_xlabel('Fixation')
    axs[i, j+1].set_ylabel('Dot count')
    axs[i, j+1].yaxis.grid(True, which='both', color='k', alpha=0.2)
    axs[i, j+1].minorticks_on()
    axs[i, j+1].set_ylim(-1, 22)


fig, axs = plt.subplots(4, 6, figsize=(16, 8))
idxs = np.random.choice(BATCH_SIZE, size=12, replace=False)
for i in range(4):
    for j in (0, 2, 4):
        plot_trial(idxs[i*3+j//2], axs, i, j)
