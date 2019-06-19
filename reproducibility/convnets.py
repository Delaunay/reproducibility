from track import TrackClient

import sys
import torch
import torch.nn as nn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import torch.nn.functional as F
import argparse
import os

from apex import amp


sys.stderr = sys.stdout

parser = argparse.ArgumentParser(description='Convnet training for torchvision models')

parser.add_argument('--batch-size', '-b', type=int, help='batch size', default=128)
parser.add_argument('--cuda', action='store_true', dest='cuda', default=True, help='enable cuda')
parser.add_argument('--no-cuda', action='store_false', dest='cuda', help='disable cuda')

parser.add_argument('--workers', '-j', type=int, default=4, help='number of workers/processors to use')
parser.add_argument('--seed', '-s', type=int, default=0, help='seed to use')
parser.add_argument('--epochs', '-e', type=int, default=30, help='number of epochs')

parser.add_argument('--arch', '-a', metavar='ARCH', default='convnet')
parser.add_argument('--lr', '--learning-rate', default=0.001, type=float, metavar='LR')
parser.add_argument('--momentum', default=0.9, type=float, metavar='MT')
parser.add_argument('--opt-level', default='O0', type=str)
parser.add_argument('--shape', nargs='*', default=(3, 32, 32))

parser.add_argument('--data', metavar='DIR', default='mnist', help='path to dataset')
parser.add_argument('--init', default=None, help='reuse an init weight', type=str)

WEIGHT_LOC = os.path.dirname(os.path.realpath(__file__)) + '/weights'

# ----
trial = TrackClient(backend='file:report.json')
trial.set_project(
    name='Reproducibility',
    description='Test NVIDIA vs AMD performance in term of loss/accuracy')

trial.new_trial()

args = trial.get_arguments(parser.parse_args(), show=True)
device = trial.get_device()

tag = 'cpu'
if torch.cuda.is_available():
    tag = torch.cuda.get_device_name(device)
    torch.cuda.manual_seed_all(args.seed)

args.shape = tuple(args.shape)
trial.add_tags(arch=args.arch, device=tag)

torch.manual_seed(args.seed)

try:
    import torch.backends.cudnn as cudnn
    cudnn.benchmark = True
except Exception:
    pass


class ConvClassifier(nn.Module):
    def __init__(self, input_shape=(1, 28, 28)):
        super(ConvClassifier, self).__init__()

        c, h, w = input_shape

        self.convs = nn.Sequential(
            nn.Conv2d(c, 10, kernel_size=5),
            nn.MaxPool2d(2),
            nn.ReLU(True),
            nn.Conv2d(10, 20, kernel_size=5),
            nn.Dropout2d(),
            nn.MaxPool2d(2)
        )

        _, c, h, w = self.convs(torch.randn(1, *input_shape)).shape
        self.conv_output_size = c * h * w

        self.fc1 = nn.Linear(self.conv_output_size, self.conv_output_size // 4)
        self.fc2 = nn.Linear(self.conv_output_size // 4, 10)

    def forward(self, x):
        x = self.convs(x)
        x = x.view(-1, self.conv_output_size)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


# ----
model = ConvClassifier(args.shape)

init_file = f'{WEIGHT_LOC}/{tag}_{args.seed}.init'
if args.init is not None:
    init = torch.load(args.init)
    model.load_state_dict(init)

elif not os.path.exists(init_file):
    torch.save(model.state_dict(), init_file)

model = model.to(device)

criterion = nn.CrossEntropyLoss().to(device)

optimizer = torch.optim.SGD(
    model.parameters(),
    args.lr,
    args.momentum
)

# ----
model, optimizer = amp.initialize(
    model,
    optimizer,
    enabled=args.opt_level != 'O0',
    cast_model_type=None,
    patch_torch_functions=True,
    keep_batchnorm_fp32=None,
    master_weights=None,
    loss_scale="dynamic",
    opt_level=args.opt_level
)


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])


train_dataset = datasets.CIFAR10('/tmp', download=True, train=True, transform=transform)
test_dataset = datasets.CIFAR10('/tmp', download=True, train=False, transform=transform)

# ----
train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=args.batch_size,
    shuffle=False,
    num_workers=4,
    pin_memory=True
)

train_img_count = len(train_dataset)
train_batch_count = len(train_loader)

test_loader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=args.batch_size,
    shuffle=False,
    num_workers=4,
    pin_memory=True
)

test_img_count = len(test_dataset)
test_batch_count = len(test_loader)


def next_batch(batch_iter):
    try:
        input, target = next(batch_iter)
        input = input.to(device)
        target = target.to(device)
        return input, target

    except StopIteration:
        return None


def do_one_epoch(train_loader):
    batch_id = 0
    epoch_loss = 0

    batch_iter = iter(train_loader)

    while True:
        with trial.chrono('train_batch_time'):

            with trial.chrono('train_batch_wait'):
                batch = next_batch(batch_iter)

            if batch is None:
                break

            with trial.chrono('train_batch_compute'):
                input, target = batch

                output = model(input)
                loss = criterion(output, target)

                batch_loss = loss.item()
                epoch_loss += batch_loss

                # trial.log_metrics(step=(epoch * train_batch_count + batch_id), train_batch_loss=loss.item())

                # compute gradient and do SGD step
                optimizer.zero_grad()

                with amp.scale_loss(loss, optimizer) as scaled_loss:
                     scaled_loss.backward()

                # loss.backward()

                optimizer.step()
                batch_id += 1

    epoch_loss /= train_batch_count
    trial.log_metrics(step=epoch, train_loss=epoch_loss)

    return epoch_loss


def eval_model(test_loader):
    batch_id = 0
    epoch_loss = 0

    batch_iter = iter(test_loader)
    test_acc = 0

    while True:
        with trial.chrono('test_batch_time'):

            with trial.chrono('test_batch_wait'):
                batch = next_batch(batch_iter)

            if batch is None:
                break

            with trial.chrono('test_batch_compute'):
                with torch.no_grad():
                    input, target = batch
                    output = model(input)

                    acc = (target.eq(output.max(dim=1)[1].long())).sum()
                    loss = criterion(output, target)

                    test_acc += acc.item()

                    batch_loss = loss.item()
                    epoch_loss += batch_loss

                optimizer.step()
                batch_id += 1

    test_acc /= test_img_count
    epoch_loss /= test_batch_count

    trial.log_metrics(step=epoch, test_loss=epoch_loss)
    trial.log_metrics(step=epoch, test_acc=test_acc * 100)

    return test_acc * 100


trial.set_eta_total(args.epochs)

with trial:
    model.train()
    for epoch in range(args.epochs):
        with trial.chrono('epoch') as epoch_time:
            with trial.chrono('epoch_time'):
                loss = do_one_epoch(train_loader)

            with trial.chrono('eval_time'):
                acc = eval_model(test_loader)

        trial.show_eta(epoch, epoch_time, f'| loss: {loss:5.2f} | acc: {acc:5.2f}')

trial.report()
trial.save()



