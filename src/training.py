import torch
import os, time
from util import denormalize, warp
from torchvision.utils import save_image
from visual import show_images2

def train_model(model, train_loader, val_loader, optimizer, criterion, device, epoch_count, save_dir="./checkpoints"):
    model.train()
    os.makedirs(save_dir, exist_ok=True)
    try:
        for epoch in range(1, epoch_count + 1):
            print("Epoch:", epoch)
            epoch_loss = 0

            running_loss = 0
            print_interval = 10
            num_batches = len(train_loader)
            start_time = time.time()
            for batch_ind, data in enumerate(train_loader):
                frame1, frame2, true_frame = data
                frame1, frame2, true_frame = frame1.to(device), frame2.to(device), true_frame.to(device)
                optimizer.zero_grad()
                pred_frame, flows_1, flows_2, feature_pyramid1, feature_pyramid2 = model(frame1, frame2)
                loss = criterion(pred_frame, true_frame, flows_1, flows_2, feature_pyramid1, feature_pyramid2)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                running_loss += loss.item()
                if (batch_ind % print_interval == print_interval-1):
                    new_time = time.time()
                    time_per_batch = (new_time-start_time)/print_interval
                    start_time = new_time
                    time_left_string = time.strftime('%H:%M:%S', time.gmtime((num_batches-batch_ind-1)*time_per_batch))
                    print(f"\n[{batch_ind+1} / {num_batches}] Mean batch loss: {running_loss/print_interval} | Epoch ETA: {time_left_string}")
                    running_loss = 0

            mean_train_loss = epoch_loss / len(train_loader)
            print(f"Epoch [{epoch}/{epoch_count}] Training Loss: {mean_train_loss:.4f}")
            
            model_data = {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'gradients': {name: param.grad.clone() for name, param in model.named_parameters() if param.grad is not None}
                }

            torch.save(model_data, os.path.join(save_dir, f"model_data_epoch_{epoch}.pth"))
            print("Epoch done. Validating...")
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for val_batch_idx, (frame1, frame2, true_frame) in enumerate(val_loader):
                    frame1, frame2, true_frame = frame1.to(device), frame2.to(device), true_frame.to(device)

                    pred_frame, flows_1, flows_2, feature_pyramid1, feature_pyramid2 = model(frame1, frame2)
                    batch_loss = criterion(pred_frame, true_frame, flows_1, flows_2, feature_pyramid1, feature_pyramid2)
                    val_loss += batch_loss.item()

                    print(f"\rValidation Batch {val_batch_idx + 1}/{len(val_loader)} Loss: {batch_loss.item():.4f}", end="")

            mean_val_loss = val_loss / len(val_loader)
            print(f"Epoch [{epoch}/{epoch_count}] Validation Loss: {mean_val_loss:.4f}")
            model.train()

    except KeyboardInterrupt:
        model_data = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'gradients': {name: param.grad.clone() for name, param in model.named_parameters() if param.grad is not None}
            }

        torch.save(model_data, os.path.join(save_dir, f"interrupted_data_{epoch}.pth"))
        print("\nSAVED INTERRUPT")

def test_model(model, criterion, test_loader, device, output_dir="./output"):
    model.eval()
    os.makedirs(output_dir, exist_ok=True)
    total_loss = 0
    with torch.no_grad():
        for idx, (frame1, frame2, true_frame) in enumerate(test_loader):
            frame1, frame2, true_frame = frame1.to(device), frame2.to(device), true_frame.to(device)
            pred_frame, flows_1, flows_2, feature_pyramid1, feature_pyramid2 = model(frame1, frame2)
            loss = criterion(pred_frame, true_frame, flows_1, flows_2, feature_pyramid1, feature_pyramid2)
            total_loss += loss.item()
            pred_img = denormalize(pred_frame[0].cpu().detach()).clamp(0, 1)
            save_image(pred_img, os.path.join(output_dir, f"pred_{idx}.png"))

    print(f"Test Loss: {total_loss / len(test_loader):.4f}")

def train_test(model, device, train_loader, val_loader, test_loader, epochs, lr, criterion):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    train_model(model, train_loader, val_loader, optimizer, criterion, device, epochs)
    test_model(model, criterion, test_loader, device, output_dir="./test_outputs")

def infer(model_list, infer_loader, device):
    for model in model_list:
        model.eval()

    for frame1, frame2, true_frame in infer_loader:
        frame1 = frame1.to(device)
        frame2 = frame2.to(device)
        pred_frames = []
        with torch.no_grad():
            for model in model_list:
                pred_frame, flows_1, flows_2, feature_pyramid1, feature_pyramid2 = model(frame1, frame2)
                pred_frames.append(pred_frame)

        pred_frames = list(map(lambda x: x[0].unsqueeze(0), pred_frames))
        show_images2((frame1[0].unsqueeze(0), frame2[0].unsqueeze(0), true_frame[0].unsqueeze(0), *pred_frames), 
                     ["Frame1", "Frame2", "True frame"] + ["model"]*len(model_list))