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
import traceback
from apex import amp
import reproducibility.resnet as resnet_cifar


sys.stderr = sys.stdout

all_models = models.__dict__
all_models.update(resnet_cifar.__dict__)
parser = argparse.ArgumentParser(description='Convnet training for torchvision models')

parser.add_argument('--batch-size', '-b', type=int, help='batch size', default=256)
parser.add_argument('--cuda', action='store_true', dest='cuda', default=True, help='enable cuda')
parser.add_argument('--no-cuda', action='store_false', dest='cuda', help='disable cuda')

parser.add_argument('--workers', '-j', type=int, default=4, help='number of workers/processors to use')
parser.add_argument('--seed', '-s', type=int, default=0, help='seed to use')
parser.add_argument('--epochs', '-e', type=int, default=30, help='number of epochs')

parser.add_argument('--warmup', default=True, action='store_true', dest='warm')
parser.add_argument('--no-warmup', action='store_false', dest='warm')
parser.add_argument('--warmup_lr', type=float, default=0.001)
parser.add_argument('--warmup_epoch', type=int, default=5, help='number of epochs')

parser.add_argument('--arch', '-a', metavar='ARCH', default='convnet', choices=all_models.keys())
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float, metavar='LR')
parser.add_argument('--momentum', default=0.9, type=float, metavar='MT')
parser.add_argument('--weight-decay', default=5e-4, type=float, metavar='MT')
parser.add_argument('--opt-level', default='O0', type=str)
parser.add_argument('--shape', nargs='*', default=(3, 32, 32))

parser.add_argument('--data', metavar='DIR', default='mnist', help='path to dataset')
parser.add_argument('--init', default=None, help='reuse an init weight', type=str)
parser.add_argument('--report', default='report.json')
parser.add_argument('--dry-run', action='store_true')

WEIGHT_LOC = os.path.dirname(os.path.realpath(__file__)) + '/weights'

# ----
args = parser.parse_args()
trial = TrackClient(backend=f'file:{args.report}')
trial.set_project(
    name='Reproducibility',
    description='Test NVIDIA vs AMD performance in term of loss/accuracy')

trial.new_trial()
args = trial.get_arguments(args, show=True)
device = trial.get_device()

tag = 'cpu'
if torch.cuda.is_available() and args.cuda:
    tag = torch.cuda.get_device_name(device)
    torch.cuda.manual_seed_all(args.seed)

args.shape = tuple(args.shape)
trial.add_tags(arch=args.arch, device=tag)

torch.manual_seed(args.seed)

try:
    import torch.backends.cudnn as cudnn
    cudnn.benchmark = True

except Exception:
    traceback.print_exc()


class ConvClassifier(nn.Module):
    def __init__(self, input_shape=(1, 28, 28)):
        super(ConvClassifier, self).__init__()

        c, h, w = input_shape

        self.convs = nn.Sequential(
            nn.Conv2d(c, 10, kernel_size=5),
            nn.MaxPool2d(2),
            nn.ReLU(True),
            nn.Conv2d(10, 20, kernel_size=5),
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
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


# ----
init_file = f'{WEIGHT_LOC}/{tag}_{args.seed}_{args.arch}.init'
if args.arch == 'convnet':
    model = ConvClassifier(args.shape)
elif args.arch.endswith('cifar'):
    args.shape = (3, 32, 32)
    model = all_models[args.arch]()
else:
    args.shape = (3, 224, 224)
    model = all_models[args.arch]()


if args.init is not None:
    init = torch.load(args.init)
    model.load_state_dict(init)

elif os.path.exists(init_file):
    init = torch.load(init_file)
    model.load_state_dict(init)

else:
    torch.save(model.state_dict(), init_file)

if args.dry_run:
    sys.exit()

model = model.to(device)
criterion = nn.CrossEntropyLoss().to(device)


def make_optimizer(model, lr):
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr,
        args.momentum,
        weight_decay=args.weight_decay
    )

    return amp.initialize(
        model,
        optimizer,
        enabled=args.opt_level != 'O0',
        opt_level=args.opt_level
    )


transform = transforms.Compose([
    transforms.RandomResizedCrop(size=args.shape[-1]),
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


def do_one_epoch(train_loader, model, optimizer):
    batch_id = 0
    batch_iter = iter(train_loader)
    epoch_items = []

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

                epoch_items.append(loss.detach())

                # compute gradient and do SGD step
                optimizer.zero_grad()

                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()

                optimizer.step()
                batch_id += 1

    count = len(epoch_items)
    epoch_loss = sum([i.item() for i in epoch_items]) / count
    trial.log_metrics(step=epoch, train_loss=epoch_loss)

    return epoch_loss


def eval_model(test_loader, model, optimizer):
    batch_id = 0
    batch_iter = iter(test_loader)

    acc_items = []
    loss_items = []

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

                    acc_items.append(acc.detach())
                    loss_items.append(loss.detach())

                optimizer.step()
                batch_id += 1

    test_acc = sum([i.item() for i in acc_items]) / test_img_count
    epoch_loss = sum([i.item() for i in loss_items]) / test_batch_count

    trial.log_metrics(step=epoch, test_loss=epoch_loss)
    trial.log_metrics(step=epoch, test_acc=test_acc * 100)

    return test_acc * 100


trial.set_eta_total(args.epochs)

with trial:
    model.train()

    if args.warmup:
        model_warm, optimizer_warm = make_optimizer(model, args.warmup_lr)
        print('warm up')
        for epoch in range(args.warmup_epoch):
            with trial.chrono('warmup_epoch') as epoch_time:
                loss = do_one_epoch(train_loader, model_warm, optimizer_warm)

            trial.show_eta(epoch, epoch_time, f'| loss: {loss:5.2f}')

    print('training')
    model, optimizer = make_optimizer(model, args.lr)
    for epoch in range(args.epochs):
        with trial.chrono('epoch') as epoch_time:

            with trial.chrono('epoch_time'):
                loss = do_one_epoch(train_loader, model, optimizer)

            with trial.chrono('eval_time'):
                acc = eval_model(test_loader, model, optimizer)

        trial.show_eta(epoch, epoch_time, f'| loss: {loss:5.2f} | acc: {acc:5.2f}')

trial.report()
trial.save()



