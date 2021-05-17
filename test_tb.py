from torch.utils.tensorboard import SummaryWriter

if __name__ == '__main__':
    writer = SummaryWriter("dataset/logs")
    for i in range(100):
        writer.add_scalar("y=x",i,i)

    writer.close()