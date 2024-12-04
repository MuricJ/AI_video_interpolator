import torchvision.transforms as transforms
import torch
from torch.utils.data import DataLoader, Subset
from model import FrameInterpolator
from dataset_handler import split_dataset
from training import train_test, infer
from dataset_handler import split_dataset, Vimeo90KDataset
import util
import losses
from video_convertor import generate_video_samples

if __name__ == "__main__":
    EPOCHS = 16
    BATCH_SIZE = 3
    LR = 8e-5
    CRITERION = losses.percLoss
    CHECKPOINT = "./checkpoints/size24_fullres.pth"

    mean, std = [util.DENORM_MEAN]*3, [util.DENORM_STD]*3
    transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.CenterCrop((992, 1440)),
                    #transforms.Resize((16, 16)),
                    transforms.Normalize(mean, std),
                    #lambda x: transforms.functional.crop(x, 25, 85, 224, 224)
                    ])
   
    full_dataset = Vimeo90KDataset("./vimeo90k", transform=transform)
    train_dataset, val_dataset, test_dataset = split_dataset(full_dataset)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu") 
    model = FrameInterpolator(device).to(device)  
    if CHECKPOINT:
        print("Loading model from:", CHECKPOINT)
        checkpoint = torch.load(CHECKPOINT)
        model.load_state_dict(checkpoint['model_state_dict'])

    #model.train()
    #sample_frame1, sample_frame2, sample_true_frame = next(iter(train_loader))
    #torchinfo.summary(model, input_data=(sample_frame1.to(device), sample_frame2.to(device)), device=device)
            
    #train_test(model, device, train_loader, val_loader, test_loader, EPOCHS, LR, CRITERION)
    #infer([model], test_loader, device)
    generate_video_samples(model, device, transform)
    pass
